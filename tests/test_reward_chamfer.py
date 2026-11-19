"""Chamfer distance of reward."""
import unittest

import numpy as np

from reward.metrics.chamfer import NORMALIZATIONS, chamfer_distance
from reward.utils import sample_surface

# An irregular tetrahedron: no rotational symmetry, so orientation errors are visible.
VERTICES = np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0.3, 0.2, 3]], dtype=np.float64)
FACES = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
YAW_90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
FLIP = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)


class ChamferTest(unittest.TestCase):
    def setUp(self):
        self.points = sample_surface(VERTICES, FACES, 2048, 42)

    def test_position_scale_and_yaw_are_ignored(self):
        moved = 2.5 * self.points @ YAW_90.T + np.array([4.0, -1.0, 7.0])
        for normalization in NORMALIZATIONS:
            self.assertAlmostEqual(chamfer_distance(moved, self.points, normalization), 0, places=9)

    def test_upside_down_and_wrong_proportions_are_penalized(self):
        for normalization in NORMALIZATIONS:
            flipped = chamfer_distance(self.points @ FLIP.T, self.points, normalization)
            stretched = chamfer_distance(self.points * np.array([1, 1, 2.0]), self.points, normalization)
            self.assertGreater(flipped, 0.1)
            self.assertGreater(stretched, 0.05)

    def test_unknown_normalization_rejected(self):
        with self.assertRaises(ValueError):
            chamfer_distance(self.points, self.points, 'bounding_box')

    def test_empty_surface_rejected(self):
        with self.assertRaises(ValueError):
            sample_surface([[0, 0, 0]], np.zeros((0, 3)), 10, 42)


if __name__ == '__main__':
    unittest.main()
