"""Local checks of factory configuration, object splits and Parquet records."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image
from pydantic import ValidationError
import yaml

from rl.data_preprocessor.base import DataConfig, OutputsConfig
from rl.data_preprocessor.threed_code_data_factory import (
    ThreeDCodeDataFactoryConfig, ThreeDCodeDataFactoryProcessor,
)
from rl.prepare import parse_data_config, prepare


class FactoryTest(unittest.TestCase):
    def config(self, root: Path) -> ThreeDCodeDataFactoryConfig:
        return ThreeDCodeDataFactoryConfig(
            data_preparation='3DCodeDataFactoryPreprocessor',
            outputs=OutputsConfig(train=root / 'train.parquet', val=root / 'val.parquet'),
            system_prompt='Create a mesh.', user_prompt='Reconstruct.',
            source_root=root / 'source',
            image_paths=[f'color_renders/Image_{i:03}.png' for i in (5, 15, 25, 35)],
            val_fraction=0.5, seed=42,
        )

    def test_process_validates_base_config_before_reading_data(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            raw = self.config(Path(directory)).model_dump()
            base = DataConfig.model_validate(raw)
            splits = ThreeDCodeDataFactoryProcessor(base)
            for field, value in (('seed', 'invalid'), ('val_fraction', 0), ('image_paths', [])):
                with self.subTest(field=field):
                    invalid = base.model_copy(update={field: value})
                    with self.assertRaises(ValidationError):
                        ThreeDCodeDataFactoryProcessor(invalid).process()
            del raw['source_root']
            with self.assertRaises(ValidationError):
                ThreeDCodeDataFactoryProcessor(DataConfig.model_validate(raw)).process()
            with self.assertRaises(FileNotFoundError):
                splits.process()

    def test_dataset_fields_are_required_in_yaml(self):
        path = Path('configs/data-3d_code_data_factory.yaml')
        config = parse_data_config(path)
        self.assertIsInstance(config, ThreeDCodeDataFactoryConfig)
        for field in ('source_root', 'image_paths', 'val_fraction', 'seed'):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir='temp') as directory:
                raw = yaml.safe_load(path.read_text(encoding='utf-8'))
                del raw[field]
                missing = Path(directory) / 'config.yaml'
                missing.write_text(yaml.safe_dump(raw), encoding='utf-8')
                with self.assertRaises(ValidationError):
                    parse_data_config(missing)

    def test_invalid_partition_and_view_settings(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            config = self.config(Path(directory))
            config.source_root.mkdir()
            (config.source_root / 'A').mkdir()
            with self.assertRaisesRegex(ValueError, 'at least two objects'):
                ThreeDCodeDataFactoryProcessor(config).process()
            (config.source_root / 'B').mkdir()
            duplicate = config.model_copy(update={'image_paths': ['view.png', 'view.png']})
            with self.assertRaisesRegex(ValueError, 'distinct views'):
                ThreeDCodeDataFactoryProcessor(duplicate).process()
            unbalanced = config.model_copy(update={'val_fraction': 0.99})
            with self.assertRaisesRegex(ValueError, 'at least one object'):
                ThreeDCodeDataFactoryProcessor(unbalanced).process()

    def test_two_objects_to_parquet(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            config = self.config(Path(directory).resolve())
            colors = {'AFactory': (255, 0, 0), 'BFactory': (0, 0, 255)}
            for name, color in colors.items():
                for relative in config.image_paths:
                    path = config.source_root / name / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (16, 16), color).save(path)
            partitions = []
            for _ in range(2):
                prepare(config)
                ids = []
                for split, path in (('train', config.outputs.train), ('val', config.outputs.val)):
                    frame = pd.read_parquet(path)
                    self.assertEqual(len(frame), 1)
                    row = frame.iloc[0]
                    name = row['extra_info']['object_id']
                    ids.append(name)
                    self.assertEqual(row['extra_info']['split'], split)
                    self.assertEqual(len(row['images']), 4)
                    self.assertEqual(row['prompt'][1]['content'].count('<image>'), 4)
                    for image in row['images']:
                        with Image.open(BytesIO(image['bytes'])) as opened:
                            self.assertEqual(opened.getpixel((0, 0)), colors[name])
                    self.assertEqual(row['reward_model']['ground_truth'], name)
                self.assertEqual(set(ids), set(colors))
                partitions.append(ids)
            self.assertEqual(partitions[0], partitions[1])
