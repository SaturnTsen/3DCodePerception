import tempfile
from functools import partial
from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from PIL import Image

from rl.prepare import prepare
from rl.reward import compute_score


class PipelineTest(unittest.TestCase):
    def test_images_targets_and_split(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            Image.new('RGB', (16, 16), 'red').save(root / 'view.png')
            np.savez(root / 'target.npz', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
            config = DictConfig(dict(sources={}, outputs={}, system_prompt='Create a mesh.'))
            for split in ('train', 'val'):
                sample_file = root / f'{split}.jsonl'
                sample_file.write_text(json.dumps(dict(object_id=split, images=['view.png'], instruction='Reconstruct.', target='target.npz')) + '\n')
                config['sources'][split] = str(sample_file)
                config['outputs'][split] = str(root / f'{split}.parquet')
            prepare(config)
            row = pd.read_parquet(config['outputs']['train']).iloc[0]
            self.assertEqual(row.prompt[-1]['content'].count('<image>'), len(row.images))
            self.assertTrue(Path(row.reward_model['ground_truth']).is_file())
            self.assertGreater(len(row.images[0]['bytes']), 0)

    def test_blender_reward(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            target = root / 'target.npz'
            np.savez(target, vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
            code = '''import bpy
mesh = bpy.data.meshes.new('triangle')
mesh.from_pydata([(0,0,0),(1,0,0),(0,1,0)], [], [(0,1,2)])
obj = bpy.data.objects.new('triangle', mesh)
bpy.context.collection.objects.link(obj)
'''
            score = partial(
                compute_score, '3d_mesh', ground_truth=str(target), extra_info={},
                blender='/home/ubuntu/.local/bin/blender', timeout=30,
                samples=128, seed=42, work_dir=str(root),
            )
            result = score(code)
            self.assertAlmostEqual(result['score'], 1)
            self.assertEqual(score('raise RuntimeError()')['score'], 0)
