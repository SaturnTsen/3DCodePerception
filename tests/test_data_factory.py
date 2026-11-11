"""Local checks of factory configuration, object splits and GLB target export."""

from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import ValidationError
import yaml

from rl.data_prep.base import DataConfig, OutputsConfig
from rl.data_prep.threed_code_data_factory import (
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
            val_fraction=0.5, seed=42, target_dir=root / 'targets',
            blender=Path('/home/ubuntu/.local/bin/blender'), timeout=60,
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
        for field in ('source_root', 'image_paths', 'val_fraction', 'seed',
                      'target_dir', 'blender', 'timeout'):
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

    def test_two_objects_through_blender_and_parquet(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            config = self.config(Path(directory).resolve())
            colors = {'AFactory': (255, 0, 0), 'BFactory': (0, 0, 255)}
            for name, color in colors.items():
                for relative in config.image_paths:
                    path = config.source_root / name / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB', (16, 16), color).save(path)
            script = '''import bpy
from pathlib import Path
import sys
root = Path(sys.argv[sys.argv.index('--') + 1])
for name, x in [('AFactory', 0), ('BFactory', 3)]:
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(location=(x, 0, 0))
    bpy.ops.export_scene.gltf(filepath=str(root / name / (name + '.glb')), export_format='GLB')
'''
            subprocess.run(
                [str(config.blender), '--background', '--factory-startup',
                 '--python-exit-code', '1', '--python-expr', script, '--', str(config.source_root)],
                check=True, capture_output=True, text=True, timeout=config.timeout,
            )
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
                    with np.load(row['reward_model']['ground_truth'], allow_pickle=False) as mesh:
                        expected_x = 0 if name == 'AFactory' else 3
                        np.testing.assert_allclose(mesh['vertices'].mean(axis=0), [expected_x, 0, 0], atol=1e-6)
                        np.testing.assert_allclose(np.ptp(mesh['vertices'], axis=0), [2, 2, 2], atol=1e-6)
                        self.assertEqual(mesh['faces'].shape[1], 3)
                self.assertEqual(set(ids), set(colors))
                partitions.append(ids)
            self.assertEqual(partitions[0], partitions[1])
