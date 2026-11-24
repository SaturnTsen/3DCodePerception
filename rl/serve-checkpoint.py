"""Serve a verl FSDP2, Hugging Face, or LoRA checkpoint with vLLM.

Example:
    pixi run serve --checkpoint-path outputs/run/global_step_46 --port 8000

The checkpoint path can also be the first positional argument. Extra options
are passed to vLLM, with an optional '--' separating them from launcher options.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from torch import Tensor


def option_value(options: list[str], flag: str) -> str | None:
    for index, option in enumerate(options):
        key, separator, value = option.partition("=")
        if key.replace("_", "-") != flag:
            continue
        if separator:
            return value
        if index + 1 == len(options) or options[index + 1].startswith("--"):
            raise ValueError(f"{flag} requires a value")
        return options[index + 1]
    return None


def fingerprint(files: list[Path], extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode())
    for path in sorted(files):
        stat = path.stat()
        digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def prepare_cache(destination: Path, populate: Callable[[Path], None]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not destination.is_dir():
            with TemporaryDirectory(dir=destination.parent) as temporary:
                staging = Path(temporary)
                populate(staging)
                staging.rename(destination)


def actor_shard_paths(actor: Path) -> list[Path]:
    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    world_sizes = {
        int(match[1])
        for path in actor.glob("model_world_size_*_rank_*.pt")
        if (match := pattern.fullmatch(path.name))
    }
    if len(world_sizes) != 1:
        raise ValueError(f"Expected model shards from a single FSDP world size in {actor}")
    world_size = world_sizes.pop()
    if world_size < 1:
        raise ValueError(f"Invalid FSDP world size in {actor}: {world_size}")
    paths = [actor / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Checkpoint in {actor} is missing model shards: {', '.join(missing)}")
    return paths


def load_actor_shards(paths: list[Path]) -> list[dict[str, Tensor]]:
    import torch
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Replicate, Shard
    from torch.distributed.tensor._dtensor_spec import DTensorSpec, ShardOrderEntry, TensorMeta

    tensor_types = [DeviceMesh, DTensor, DTensorSpec, Replicate, Shard, ShardOrderEntry, TensorMeta]
    # mmap lets the exporter materialize one safetensors shard at a time.
    with torch.serialization.safe_globals(tensor_types):
        states = [torch.load(path, map_location="cpu", mmap=True, weights_only=True) for path in paths]
    if not states[0] or any(state.keys() != states[0].keys() for state in states[1:]):
        raise ValueError("FSDP model shards must contain the same nonempty set of tensor names")
    if any(".lora_" in name for name in states[0]):
        raise ValueError("Export the verl LoRA checkpoint to PEFT format before serving it")
    return states


def merge_actor_tensor(name: str, pieces: list[Tensor]) -> Tensor:
    import torch
    from torch.distributed.tensor import DTensor, Replicate, Shard

    first = pieces[0]
    signature = (type(first), first.shape, first.dtype)
    if any((type(piece), piece.shape, piece.dtype) != signature for piece in pieces[1:]):
        raise ValueError(f"FSDP shards disagree on type, shape, or dtype for {name}")
    if not isinstance(first, DTensor):
        if any(not torch.equal(first, piece) for piece in pieces[1:]):
            raise ValueError(f"Replicated FSDP tensor differs between ranks: {name}")
        return first.contiguous()

    placements = first.placements
    ranks = list(range(len(pieces)))
    if len(placements) != 1:
        raise ValueError(f"Expected a one-dimensional FSDP2 mesh for {name}")
    local_pieces = []
    for piece in pieces:
        assert isinstance(piece, DTensor)
        if piece.placements != placements or piece.device_mesh.mesh.tolist() != ranks:
            raise ValueError(f"Unexpected FSDP2 placement or rank order for {name}")
        local_pieces.append(piece.to_local())
    placement = placements[0]
    if isinstance(placement, Shard):
        merged = torch.cat(local_pieces, dim=placement.dim)
    elif isinstance(placement, Replicate):
        merged = merge_actor_tensor(name, local_pieces)
    else:
        raise ValueError(f"Unsupported FSDP2 placement for {name}: {placement}")
    if merged.shape != first.shape:
        raise ValueError(f"Merged FSDP2 tensor {name} has shape {merged.shape}, expected {first.shape}")
    return merged.contiguous()


def export_actor(shards: list[Path], assets: Path, output: Path) -> None:
    from safetensors.torch import save_file

    print(f"Exporting {len(shards)} FSDP2 model shards to safetensors", file=sys.stderr, flush=True)
    shutil.copytree(assets, output, dirs_exist_ok=True)
    states = load_actor_shards(shards)
    pending: dict[str, Tensor] = {}
    written: list[tuple[Path, list[str]]] = []
    pending_size = total_size = 0

    def flush() -> None:
        path = output / f"weights-{len(written) + 1:05d}.tmp"
        save_file(pending, path, metadata={"format": "pt"})
        written.append((path, list(pending)))
        pending.clear()

    for name, tensor in states[0].items():
        size = tensor.numel() * tensor.element_size()
        if pending and pending_size + size > 4 * 1024**3:
            flush()
            pending_size = 0
        pending[name] = merge_actor_tensor(name, [state[name] for state in states])
        pending_size += size
        total_size += size
    flush()

    weight_map: dict[str, str] = {}
    for index, (path, names) in enumerate(written, start=1):
        filename = f"model-{index:05d}-of-{len(written):05d}.safetensors"
        path.rename(output / filename)
        weight_map.update(dict.fromkeys(names, filename))
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2)
    )


def read_adapter_config(checkpoint: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    source = checkpoint / "adapter_config.json"
    if source.is_file():
        return json.loads(source.read_text())

    from peft import LoraConfig, LoraRuntimeConfig

    if "model" not in snapshot:
        raise ValueError("LoRA checkpoint requires adapter_config.json or a model section in resolved_config.yaml")
    model = snapshot["model"]
    if model["tuning"] != "lora":
        raise ValueError("Expected a LoRA model in resolved_config.yaml")
    lora = model["lora"]
    options = dict(lora["backend"])
    options["runtime_config"] = LoraRuntimeConfig(**options["runtime_config"])
    options.update(
        r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"], task_type=lora["task_type"], bias=lora["bias"],
    )
    config = LoraConfig(**options).to_dict()
    config["base_model_name_or_path"] = model["path"]
    return config


def export_dcp_adapter(source: Path, config: dict[str, Any], output: Path) -> None:
    import torch
    from peft import LoraConfig
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.filesystem import FileSystemReader
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata
    from torch.distributed.checkpoint.state_dict_loader import load

    weights: dict[str, Tensor] = {}
    names: dict[str, str] = {}
    for key, metadata in FileSystemReader(source).read_metadata().state_dict_metadata.items():
        match = re.fullmatch(r"model\.(.+\.lora_[AB])(?:\.[^.]+)?\.weight", key)
        if match is None:
            continue
        name = match[1] + ".weight"
        if not isinstance(metadata, TensorStorageMetadata) or name in names:
            raise ValueError(f"Unsupported or duplicate LoRA tensor: {key}")
        original = key.removeprefix("model.")
        names[name] = original
        weights[original] = torch.empty(metadata.size, dtype=metadata.properties.dtype)
    if not weights:
        raise ValueError("FSDP checkpoint contains no LoRA A/B tensors")
    load({"model": weights}, checkpoint_id=source, no_dist=True)
    LoraConfig(**config).save_pretrained(str(output))
    save_file({name: weights[key].contiguous() for name, key in names.items()}, output / "adapter_model.safetensors")


def validate_adapter(base: Path, adapter: Path, config: dict[str, Any]) -> int:
    from safetensors import safe_open

    if json.loads((base / "config.json").read_text())["model_type"] != "qwen3_5":
        raise ValueError("Expected a Qwen3.5 multimodal base model")
    if config["peft_type"] != "LORA" or config["bias"] != "none" or config.get("modules_to_save"):
        raise ValueError("Expected LoRA A/B weights with bias=none and no modules_to_save")
    unsupported = ("use_dora", "alora_invocation_tokens", "use_bdlora", "rank_pattern", "alpha_pattern")
    if any(config.get(key) for key in unsupported):
        raise ValueError("The adapter uses LoRA options unsupported by this launcher")

    index = base / "model.safetensors.index.json"
    with ExitStack() as stack:
        shards: dict[str, safe_open] = {}
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
        else:
            filename = "model.safetensors"
            shards[filename] = stack.enter_context(safe_open(base / filename, framework="pt"))
            weight_map = dict.fromkeys(shards[filename].keys(), filename)
        weights = stack.enter_context(safe_open(adapter / "adapter_model.safetensors", framework="pt"))
        names = set(weights.keys())
        if not names:
            raise ValueError("Adapter weight file is empty")
        rank = int(config["r"])
        for name in sorted(names):
            match = re.fullmatch(r"base_model\.model\.(.+)\.lora_([AB])\.weight", name)
            if match is None:
                raise ValueError(f"Unsupported adapter tensor: {name}")
            module, side = match.groups()
            counterpart = f"base_model.model.{module}.lora_{'B' if side == 'A' else 'A'}.weight"
            if counterpart not in names:
                raise ValueError(f"Missing LoRA A/B counterpart for {name}")
            base_name = module + ".weight"
            if base_name not in weight_map:
                raise ValueError(f"Adapter module is absent from base model: {base_name}")
            filename = weight_map[base_name]
            if filename not in shards:
                shards[filename] = stack.enter_context(safe_open(base / filename, framework="pt"))
            shape = shards[filename].get_slice(base_name).get_shape()
            current = weights.get_slice(name).get_shape()
            if len(shape) != 2:
                raise ValueError(f"Expected a two-dimensional base weight: {base_name}")
            expected = [rank, shape[1]] if side == "A" else [shape[0], rank]
            if current != expected:
                raise ValueError(f"Incompatible adapter shape for {name}: got {current}, expected {expected}")
    return rank


def tokenizer_with_config(checkpoint: Path, base: Path, cache: Path) -> Path:
    files = [base / "config.json"]
    files.extend(
        path for path in checkpoint.iterdir()
        if path.is_file() and path.suffix in {".json", ".jinja", ".model", ".txt"}
        and path.name not in {"adapter_config.json", "trainer_state.json", "config.json"}
    )
    destination = cache / f"tokenizer-{fingerprint(files)}"

    def populate(output: Path) -> None:
        for path in files:
            shutil.copyfile(path, output / path.name)

    prepare_cache(destination, populate)
    return destination


def actor_model(checkpoint: Path, cache: Path, dry_run: bool) -> tuple[Path, Path]:
    actor = checkpoint / "actor" if (checkpoint / "actor").is_dir() else checkpoint
    hf_assets = actor / "huggingface"
    assets = hf_assets if hf_assets.is_dir() else checkpoint
    config_path = assets / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Checkpoint config.json is unavailable at {assets}")
    index = assets / "model.safetensors.index.json"
    if index.is_file():
        filenames = set(json.loads(index.read_text())["weight_map"].values())
        for filename in filenames:
            if not (assets / filename).is_file():
                raise ValueError(f"Missing Hugging Face model shard: {assets / filename}")
        return assets, assets
    if (assets / "model.safetensors").is_file():
        return assets, assets
    if assets != hf_assets:
        raise ValueError(f"Expected model.safetensors or model.safetensors.index.json at {assets}")

    shards = actor_shard_paths(actor)
    if json.loads(config_path.read_text())["model_type"] != "qwen3_5":
        raise ValueError("Expected a Qwen3.5 model for verl FSDP2 export")
    files = [path for path in assets.rglob("*") if path.is_file()]
    model = cache / f"fsdp2-{fingerprint(shards + files)}"
    if dry_run:
        load_actor_shards(shards)
    else:
        prepare_cache(model, lambda output: export_actor(shards, assets, output))
    return model, assets


def adapter_command(
    checkpoint: Path, cache: Path, snapshot: dict[str, Any], base_override: Path | None,
    name: str, options: set[str], passthrough: list[str],
) -> list[str]:
    config = read_adapter_config(checkpoint, snapshot)
    base_path = base_override or config.get("base_model_name_or_path") or snapshot.get("model", {}).get("path")
    if not base_path:
        raise ValueError("Provide --base-model-path to identify the LoRA base model")
    base = Path(base_path).expanduser().resolve()
    if not (base / "config.json").is_file():
        raise ValueError(f"Base config.json is unavailable at {base}. Set --base-model-path to the model directory.")

    adapter = checkpoint
    dcp = checkpoint / "pytorch_model_fsdp_0"
    if (dcp / ".metadata").is_file():
        config_key = json.dumps(config, sort_keys=True, default=sorted)
        adapter = cache / f"lora-{fingerprint(list(dcp.iterdir()), config_key)}"
        prepare_cache(adapter, lambda output: export_dcp_adapter(dcp, config, output))
    rank = validate_adapter(base, adapter, config)
    supported = (1, 8, 16, 32, 64, 128, 256, 320, 512)
    supplied_rank = option_value(passthrough, "--max-lora-rank")
    if supplied_rank is None:
        selected_rank = next((size for size in supported if size >= rank), 0)
    else:
        selected_rank = int(supplied_rank)
    if selected_rank not in supported or selected_rank < rank:
        raise ValueError(f"--max-lora-rank must be a vLLM-supported value at least {rank}, from {supported}")
    module = {"name": name, "path": str(adapter), "base_model_name": str(base)}
    command = ["vllm", "serve", str(base), "--enable-lora", "--lora-modules", json.dumps(module)]
    if supplied_rank is None:
        command += ["--max-lora-rank", str(selected_rank)]
    if "--tokenizer" not in options and (checkpoint / "tokenizer_config.json").is_file():
        command += ["--tokenizer", str(tokenizer_with_config(checkpoint, base, cache))]
    return command


def build_command(args: argparse.Namespace, passthrough: list[str]) -> tuple[list[str], str]:
    checkpoint = args.checkpoint_path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise ValueError(f"Checkpoint directory does not exist: {checkpoint}")
    run = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    name = args.adapter_name or f"{run.name}-{checkpoint.name}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Model name must contain only letters, digits, dots, underscores and hyphens")

    options = {argument.partition("=")[0].replace("_", "-") for argument in passthrough}
    managed = options & {"--model", "--enable-lora", "--no-enable-lora", "--lora-modules"}
    if managed:
        raise ValueError(f"Launcher-managed options: {', '.join(sorted(managed))}")

    cache = Path(__file__).resolve().parents[1] / "temp" / "serve-checkpoint"
    dcp = checkpoint / "pytorch_model_fsdp_0"
    snapshot: dict[str, Any] = {}
    assets = checkpoint
    is_adapter = (checkpoint / "adapter_model.safetensors").is_file() or (dcp / ".metadata").is_file()
    if is_adapter:
        snapshot_path = run / "resolved_config.yaml"
        if snapshot_path.is_file():
            snapshot = yaml.safe_load(snapshot_path.read_text())
        command = adapter_command(checkpoint, cache, snapshot, args.base_model_path, name, options, passthrough)
    else:
        if "--served-model-name" in options:
            raise ValueError("Use --model-name to set the served checkpoint name")
        if args.base_model_path is not None:
            raise ValueError("--base-model-path applies to LoRA checkpoints")
        model, assets = actor_model(checkpoint, cache, args.dry_run)
        command = ["vllm", "serve", str(model), "--served-model-name", name]

    template = assets / "chat_template.jinja"
    if "--chat-template" not in options and template.is_file():
        command += ["--chat-template", str(template)]
    if "--mm-processor-kwargs" not in options:
        processor = assets / "processor_config.json"
        if processor.is_file():
            image_processor = json.loads(processor.read_text()).get("image_processor")
            if image_processor is not None:
                command += ["--mm-processor-kwargs", json.dumps({"size": image_processor["size"]})]
        elif "template" in snapshot:
            template_config = snapshot["template"]
            command += ["--mm-processor-kwargs", json.dumps({
                "min_pixels": template_config["min_image_pixels"], "max_pixels": template_config["max_image_pixels"],
            })]
    return command + passthrough, name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, allow_abbrev=False, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--adapter-name", "--model-name", dest="adapter_name", help="Name in the API's model field")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print argv without starting vLLM")
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-"):
        argv.insert(0, "--checkpoint-path")
    boundary = argv.index("--") if "--" in argv else len(argv)
    args, extra = parser.parse_known_args(argv[:boundary])
    try:
        command, name = build_command(args, extra + argv[boundary + 1:])
        print(f"Client request model: {name}", flush=True)
        if args.dry_run:
            display = list(command)
            redact_keys = False
            for index, argument in enumerate(command):
                if argument.startswith("-"):
                    key, separator, _ = argument.partition("=")
                    redact_keys = key.replace("_", "-") == "--api-key"
                    if redact_keys and separator:
                        display[index] = f"{key}=<redacted>"
                elif redact_keys:
                    display[index] = "<redacted>"
            print(json.dumps(display, ensure_ascii=False, indent=2))
            return
        os.execvp(command[0], command)
    except (ValueError, OSError, KeyError, RuntimeError, yaml.YAMLError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
