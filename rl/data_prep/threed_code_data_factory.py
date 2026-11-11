"""Convert 3DCodeDataFactory objects to multi-view verl records."""

from collections.abc import Iterable
import math
from pathlib import Path
import random
import subprocess
import tempfile
from typing import Any

import numpy as np
from PIL import Image
from pydantic import Field

from rl.data_prep.base import DataConfig, DataPrepProcessor, PreparedSplits
from rl.geometry import sample_surface


class ThreeDCodeDataFactoryConfig(DataConfig):
    source_root: Path = Field(strict=False)
    image_paths: list[str] = Field(min_length=1)
    val_fraction: float = Field(gt=0, lt=1)
    seed: int
    target_dir: Path = Field(strict=False)
    blender: Path = Field(strict=False)
    timeout: float = Field(gt=0)


class ThreeDCodeDataFactoryProcessor(DataPrepProcessor):
    """Partition whole objects before constructing multi-view records.

    GLB targets are imported through Blender so scene transforms are preserved
    in the world-space triangle meshes consumed by the existing reward function.
    """

    config_model = ThreeDCodeDataFactoryConfig

    def process(self) -> PreparedSplits:
        config = ThreeDCodeDataFactoryConfig.model_validate(self.config.model_dump())
        objects = sorted(path for path in config.source_root.iterdir() if path.is_dir())
        if len(objects) < 2:
            raise ValueError('3DCodeDataFactory requires at least two objects for train and val')
        if len(set(config.image_paths)) != len(config.image_paths):
            raise ValueError('image_paths must contain distinct views')
        random.Random(config.seed).shuffle(objects)
        val_count = math.ceil(len(objects) * config.val_fraction)
        if not 0 < val_count < len(objects):
            raise ValueError('val_fraction must allocate at least one object to train and val')

        targets = config.target_dir.resolve()
        targets.mkdir(parents=True, exist_ok=True)
        return {
            'train': self._records(objects[val_count:], 'train', targets, config),
            'val': self._records(objects[:val_count], 'val', targets, config),
        }

    def _records(
        self, objects: list[Path], split: str, targets: Path,
        config: ThreeDCodeDataFactoryConfig,
    ) -> Iterable[dict[str, Any]]:
        for index, directory in enumerate(objects):
            images = []
            for relative_path in config.image_paths:
                path = directory / relative_path
                with Image.open(path) as opened:
                    opened.verify()
                images.append({'bytes': path.read_bytes(), 'path': None})

            source = (directory / f'{directory.name}.glb').resolve()
            target = targets / f'{directory.name}.npz'
            self._export_target(source, target, config)
            with np.load(target, allow_pickle=False) as mesh:
                sample_surface(mesh['vertices'], mesh['faces'], 1, config.seed)

            yield {
                'data_source': '3d_mesh',
                'prompt': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': '\n'.join(
                        ['<image>'] * len(images) + [self.user_prompt]
                    )},
                ],
                'images': images,
                'reward_model': {'style': 'rule', 'ground_truth': str(target)},
                'extra_info': {
                    'index': index, 'object_id': directory.name, 'split': split,
                    'source_mesh': str(source),
                },
            }

    @staticmethod
    def _export_target(
        source: Path, target: Path, config: ThreeDCodeDataFactoryConfig,
    ) -> None:
        importer = Path(__file__).with_name('import_glb.py').resolve()
        exporter = Path(__file__).resolve().parents[1] / 'export_mesh.py'
        Path('temp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            log_path = Path(directory) / 'blender.log'
            with log_path.open('w') as log:
                result = subprocess.run(
                    [str(config.blender), '--background', '--factory-startup',
                     '--python-exit-code', '1', '--python', str(importer),
                     '--python', str(exporter), '--', str(target), str(source)],
                    stdout=log, stderr=subprocess.STDOUT, timeout=config.timeout,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f'Failed to export {source}:\n{log_path.read_text(encoding="utf-8")}'
                )
