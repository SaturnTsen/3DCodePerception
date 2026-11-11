"""Convert training and validation JSONL samples to multimodal Parquet for verl."""
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from rl.geometry import sample_surface


def prepare(config: DictConfig) -> None:
    """
    Transform JSONL samples to Parquet for verl.
    Run using `pixi run python -m rl.prepare --config /path/to/config.yaml`.
    Or use `pixi run prepare-rl` to run the default config.

    An example of data config yaml:

    sources:
      train: /path/to/train.jsonl
      val: /path/to/val.jsonl
    outputs:
      train: /path/to/train.parquet
      val: /path/to/val.parquet
    system_prompt: 你是一个3D重建专家。

    Each row in the input JSONL file has the following structure:
    {
      "object_id": "chair-001",
      "images": ["front.png", "side.png"],
      "instruction": "根据图片重建这个物体。",
      "target": "/path/to/chair.npz"
    }

    One output Parquet record, represented as a Python dictionary:
    {
        "data_source": "3d_mesh",
        "prompt": [
            {"role": "system", "content": "你是一个3D重建专家。"},
            {"role": "user", "content": "<image>\n<image>\n根据图片重建这个物体。"}
        ],
        "images": [
            {"bytes": b'...', "path": None},
            {"bytes": b'...', "path": None}
        ], 
        "reward_model": {
            "style": "rule",
            "ground_truth": "/path/to/chair.npz"
        },
        "extra_info": {
            "index": 0,
            "object_id": "chair-001",
            "split": "train"
        }
    }  
    """
    seen: set[str] = set() # 已完成的数据集 split 的 object_id

    for split in ('train', 'val'):
        source = config['sources'][split]
        if source is None:
            raise ValueError(f'sources.{split} must be filled before preparing data')
        source = Path(source).resolve()

        rows: list[dict[str, Any]] = [] # 存储处理后的数据行
        objects: set[str] = set() # 当前 split 中的 object_id

        for index, line in enumerate(source.read_text().splitlines()):
            item: dict[str, Any] = json.loads(line)

            # Sample sanity check
            if item['object_id'] in seen:
                raise ValueError(f"object_id {item['object_id']!r} appears in both train and val")
            objects.add(item['object_id'])
            images = [(source.parent / name).resolve() for name in item['images']]
            if not images:
                raise ValueError('VLM samples require images')
            for image in images:
                with Image.open(image) as opened:
                    opened.verify()
            target = (source.parent / item['target']).resolve()
            with np.load(target, allow_pickle=False) as mesh:
                # Check that the mesh has the required keys
                sample_surface(mesh['vertices'], mesh['faces'], 1, 0)
            
            # Make prompt
            prompt = '\n'.join(['<image>'] * len(images)) + '\n' + item['instruction']
            
            # Make sample row for verl
            rows.append(dict(
                data_source='3d_mesh', 
                prompt=[
                    dict(role='system', content=config['system_prompt']),
                    dict(role='user', content=prompt)
                ],
                images=[
                    dict(bytes=p.read_bytes(), path=None) for p in images
                ],
                reward_model=dict(
                    style='rule',
                    ground_truth=str(target)
                ),
                extra_info=dict(
                    index=index,
                    object_id=item['object_id'],
                    split=split)
            ))

        if not rows:
            raise ValueError(f'{split} has no samples')
        seen.update(objects)
        
        destination = Path(config['outputs'][split])
        destination.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(destination, index=False)
    

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if not isinstance(config, DictConfig):
        raise ValueError('data config must be a YAML mapping')
    prepare(config)


if __name__ == '__main__':
    main()
