import unittest
import numpy as np
from rl.geometry import sample_surface, compare


class GeometryTest(unittest.TestCase):
    def test_identical_and_translated(self):
        points = sample_surface([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], 100, 42)
        self.assertEqual(compare(points, points, 1)['score'], 1)
        self.assertLess(compare(points + 2, points, 1)['score'], 0.1)

    def test_empty_surface_rejected(self):
        with self.assertRaises(ValueError):
            sample_surface([[0, 0, 0]], [], 10, 42)
