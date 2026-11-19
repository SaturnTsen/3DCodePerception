"""Code extraction and reward combination of rl/reward.py (no services needed)."""
import math
import unittest

from rl.reward import COMPONENTS, combine, extract_code

METRICS = {
    'chamfer': {'rms_radius': 0.2, 'max_radius': 0.1},
    'dinov3': {'view_paired': 0.5, 'best_assignment': 0.8},
    'uni3d': {'cos_3d_3d': 0.9, 'cos_image_3d': 0.3},
}
SETTINGS = {
    'chamfer_normalization': 'rms_radius', 'chamfer_temperature': 0.5,
    'weights': {'chamfer': 0.4, 'dinov3': 0.3, 'uni3d': 0.2, 'uni3d_image': 0.1},
}


class ExtractCodeTest(unittest.TestCase):
    def test_single_block(self):
        self.assertEqual(extract_code('Plan.\n```python\nimport bpy\n```\nDone.'), 'import bpy\n')

    def test_without_block_or_with_several_blocks_uses_whole_response(self):
        self.assertEqual(extract_code('import bpy'), 'import bpy')
        several = '```python\na = 1\n```\n```python\nb = 2\n```'
        self.assertEqual(extract_code(several), several)


class CombineTest(unittest.TestCase):
    def test_weighted_sum(self):
        result = combine('ok', METRICS, **SETTINGS)
        self.assertAlmostEqual(result['chamfer'], math.exp(-0.2 / 0.5))
        expected = 0.4 * math.exp(-0.4) + 0.3 * 0.8 + 0.2 * 0.9 + 0.1 * 0.3
        self.assertAlmostEqual(result['score'], expected)
        self.assertEqual((result['success'], result['chamfer_distance']), (1.0, 0.2))

    def test_failure_scores_zero_with_the_same_keys(self):
        success, failure = combine('ok', METRICS, **SETTINGS), combine('timeout', None, **SETTINGS)
        self.assertEqual(set(success), set(failure))
        self.assertEqual(failure['score'], 0.0)
        self.assertEqual(failure['chamfer_distance'], -1.0)
        self.assertTrue(all(failure[name] == 0.0 for name in COMPONENTS))


if __name__ == '__main__':
    unittest.main()
