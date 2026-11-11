import tempfile
from functools import partial
from pathlib import Path
import unittest

import numpy as np

from rl.reward import compute_score


class PipelineTest(unittest.TestCase):
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
