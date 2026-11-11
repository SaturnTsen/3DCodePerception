"""Focused checks for Pydantic YAML loading and dataset preparation."""

from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest

import pandas as pd
from pydantic import ValidationError
import yaml

from rl.data_prep import (
    DataConfig, DataPrepProcessor, OutputsConfig,
)
from rl.data_prep.base import PreparedSplits
from rl.prepare import PROCESSORS, parse_data_config, prepare


class FixtureProcessor(DataPrepProcessor):
    def process(self) -> PreparedSplits:
        # Partition the complete source once; source order intentionally differs
        # from output order to check that prepare preserves processor assignment.
        source = [
            {'object_id': 'validation-object', 'split': 'val'},
            {'object_id': 'training-object', 'split': 'train'},
        ]
        result: PreparedSplits = {'train': [], 'val': []}
        for split in ('train', 'val'):
            records = []
            for item in source:
                if item['split'] != split:
                    continue
                records.append({
                    'data_source': 'fixture',
                    'prompt': [
                        {'role': 'system', 'content': self.system_prompt},
                        {'role': 'user', 'content': '<image>\n' + self.user_prompt},
                    ],
                    'images': [{'bytes': b'image-content', 'path': None}],
                    'reward_model': {'style': 'rule', 'ground_truth': 'target.npz'},
                    'extra_info': {'split': split, 'object_id': item['object_id']},
                })
            if split == 'train':
                result['train'] = records
            else:
                result['val'] = records
        return result


class EmptyProcessor(DataPrepProcessor):
    def process(self) -> PreparedSplits:
        return {'train': [], 'val': []}


PROCESSORS.update({
    'TestDataPrepProcessor': FixtureProcessor,
    'EmptyTestDataPrepProcessor': EmptyProcessor,
})


class DataPrepTest(unittest.TestCase):
    def test_builtin_processor_is_available_in_fresh_process(self):
        subprocess.run(
            [sys.executable, '-c',
             'import rl.prepare\n'
             'from rl.prepare import PROCESSORS\n'
             'processor = PROCESSORS["3DCodeDataFactoryPreprocessor"]\n'
             'assert processor.__module__ == "rl.data_prep.threed_code_data_factory"\n'],
            check=True, capture_output=True, text=True,
        )

    def config(self, root: Path) -> dict[str, Any]:
        return {
            'data_preparation': 'TestDataPrepProcessor',
            'outputs': {split: str(root / 'nested' / f'{split}.parquet')
                        for split in ('train', 'val')},
            'system_prompt': 'Create a mesh.',
            'user_prompt': 'Reconstruct.',
        }

    def parse(self, config: Any) -> DataConfig:
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            path = Path(directory) / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            return parse_data_config(path)

    def test_prepare_accepts_pydantic_model_directly(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            config = DataConfig(
                data_preparation='TestDataPrepProcessor',
                outputs=OutputsConfig(train=root / 'train.parquet', val=root / 'val.parquet'),
                system_prompt='Create a mesh.',
                user_prompt='Reconstruct.',
            )
            prepare(config)
            self.assertEqual(len(pd.read_parquet(config.outputs.train)), 1)
            self.assertEqual(len(pd.read_parquet(config.outputs.val)), 1)

    def test_yaml_loading_and_nested_parquet_records(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            config = self.parse(self.config(Path(directory)))
            self.assertIsInstance(config, DataConfig)
            self.assertIsInstance(config.outputs.train, Path)
            prepare(config)
            row = pd.read_parquet(config.outputs.val).iloc[0]
            self.assertEqual(row['prompt'][0]['content'], config.system_prompt)
            self.assertEqual(row['prompt'][1]['content'], '<image>\n' + config.user_prompt)
            self.assertEqual(row['images'][0]['bytes'], b'image-content')
            self.assertEqual(row['reward_model']['ground_truth'], 'target.npz')
            self.assertEqual(row['extra_info']['object_id'], 'validation-object')
            train = pd.read_parquet(config.outputs.train).iloc[0]
            self.assertEqual(train['extra_info']['object_id'], 'training-object')

    def test_empty_split_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            raw = self.config(Path(directory))
            raw['data_preparation'] = 'EmptyTestDataPrepProcessor'
            config = self.parse(raw)
            with self.assertRaisesRegex(ValueError, 'train has no samples'):
                prepare(config)
            self.assertFalse(config.outputs.train.exists())

    def test_duplicate_output_paths_are_rejected_before_writing(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            raw = self.config(Path(directory))
            raw['outputs']['val'] = raw['outputs']['train']
            with self.assertRaisesRegex(ValueError, 'also used by another split'):
                prepare(self.parse(raw))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_required_fields_have_no_defaults(self):
        for field in ('data_preparation', 'outputs', 'system_prompt', 'user_prompt'):
            with self.subTest(field=field):
                raw = self.config(Path('temp/unused'))
                del raw[field]
                with self.assertRaises(ValidationError):
                    self.parse(raw)
        for split in ('train', 'val'):
            with self.subTest(split=split):
                raw = self.config(Path('temp/unused'))
                del raw['outputs'][split]
                with self.assertRaises(ValidationError):
                    self.parse(raw)

    def test_schema_rejects_invalid_types_and_unknown_output_fields(self):
        for field, value in (
            ('system_prompt', None), ('user_prompt', ['prompt']),
            ('outputs', ['train.parquet']), ('data_preparation', {'class': 'processor'}),
            ('system_prompt', 123),
            ('outputs', {'train': 'train.parquet', 'val': 'val.parquet', 'test': 'test.parquet'}),
        ):
            with self.subTest(field=field, value=value):
                raw = self.config(Path('temp/unused'))
                raw[field] = value
                with self.assertRaises(ValidationError):
                    self.parse(raw)
        for raw in (None, [], 'not a mapping'):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                self.parse(raw)

    def test_outer_config_preserves_dataset_fields(self):
        raw = self.config(Path('temp/unused'))
        raw['dataset_option'] = {'value': 42}
        config = self.parse(raw)
        self.assertEqual(config.model_extra, {'dataset_option': {'value': 42}})

    def test_unknown_processor_is_rejected(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            raw = self.config(Path(directory))
            raw['data_preparation'] = 'UnknownProcessor'
            with self.assertRaisesRegex(ValueError, 'Unknown data preparation processor'):
                prepare(self.parse(raw))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_current_yaml_layout_with_fixture_processor(self):
        config = parse_data_config(Path('configs/data-3d_code_data_factory.yaml'))
        self.assertEqual(config.data_preparation, '3DCodeDataFactoryPreprocessor')
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            config = config.model_copy(update={
                'data_preparation': 'TestDataPrepProcessor',
                'outputs': OutputsConfig(train=root / 'train.parquet', val=root / 'val.parquet'),
            })
            prepare(config)
            row = pd.read_parquet(config.outputs.train).iloc[0]
            self.assertEqual(row['prompt'][0]['content'], config.system_prompt)
            self.assertEqual(row['prompt'][1]['content'], '<image>\n' + config.user_prompt)
