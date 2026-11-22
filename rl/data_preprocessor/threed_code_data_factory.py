"""Convert 3DCodeDataFactory objects to multi-view verl records."""

from collections.abc import Iterable
import math
from pathlib import Path
import random

from PIL import Image
from pydantic import Field

from rl.data_preprocessor.base import DataConfig, DataPrepProcessor, ImageData, PreparedSplits, VerlRecord


class ThreeDCodeDataFactoryConfig(DataConfig):
    source_root: Path = Field(strict=False)
    image_paths: list[str] = Field(min_length=1)
    val_fraction: float = Field(gt=0, lt=1)
    seed: int


class ThreeDCodeDataFactoryProcessor(DataPrepProcessor):
    """Partition whole objects before constructing multi-view records.

    Records carry the object_id (the object's directory name under source_root) as
    ground truth. The reward service reads the same source_root on its own and is
    reached only over HTTP.
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

        return {
            'train': self._records(objects[val_count:], 'train', config),
            'val': self._records(objects[:val_count], 'val', config),
        }

    def _records(
        self, objects: list[Path], split: str, config: ThreeDCodeDataFactoryConfig,
    ) -> Iterable[VerlRecord]:
        for index, directory in enumerate(objects):
            images: list[ImageData] = []
            for path in (directory / relative_path for relative_path in config.image_paths):
                with Image.open(path) as opened:
                    opened.verify()
                images.append({'bytes': path.read_bytes(), 'path': None})

            yield {
                'data_source': '3d_mesh',
                'prompt': [
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': '\n'.join(
                        ['<image>'] * len(images) + [self.user_prompt]
                    )},
                ],
                'images': images,
                'reward_model': {'style': 'rule', 'ground_truth': directory.name},
                'extra_info': {'index': index, 'object_id': directory.name, 'split': split},
            }

