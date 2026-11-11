import tempfile
from pathlib import Path
import json
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from rl.prepare import prepare
from rl.reward import compute_score


class PipelineTest(unittest.TestCase):
    def test_images_targets_and_split(self):
        with tempfile.TemporaryDirectory(dir='temp') as directory:
            root = Path(directory)
            Image.new('RGB', (16, 16), 'red').save(root / 'view.png')
            np.savez(root / 'target.npz', vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
            config = dict(manifests={}, outputs={}, system_prompt='Create a mesh.')
            for split in ('train', 'val'):
                manifest = root / f'{split}.jsonl'
                manifest.write_text(json.dumps(dict(object_id=split, images=['view.png'], instruction='Reconstruct.', target='target.npz')) + '\n')
                config['manifests'][split] = str(manifest)
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
            kwargs = dict(blender='/home/ubuntu/.local/bin/blender', timeout=30, samples=128, seed=42, work_dir=str(root))
            result = compute_score('3d_mesh', code, str(target), {}, **kwargs)
            self.assertAlmostEqual(result['score'], 1)
            self.assertEqual(compute_score('3d_mesh', 'raise RuntimeError()', str(target), {}, **kwargs)['score'], 0)
