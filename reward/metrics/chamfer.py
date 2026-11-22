"""Alignment-normalized Chamfer distance between two point clouds.

All coordinates are Blender world coordinates with +Z up.
"""
from typing import get_args

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from reward.types import ChamferNormalization

NORMALIZATIONS: tuple[ChamferNormalization, ...] = get_args(ChamferNormalization)
# Rotations about the Blender up axis (+Z) by 0, 90, 180, and 270 degrees.
YAW_ROTATIONS = tuple(
    np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for c, s in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
)


def normalize(points: NDArray[np.float64], normalization: str) -> NDArray[np.float64]:
    """Move the centroid to the origin and divide by the chosen radius.

    rms_radius: root mean square distance to the centroid.
    max_radius: distance from the centroid to the farthest point.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(f'normalization must be one of {NORMALIZATIONS}, got {normalization!r}')
    centered = points - points.mean(axis=0)
    radii = np.linalg.norm(centered, axis=1)
    radius = np.sqrt(np.mean(radii**2)) if normalization == 'rms_radius' else radii.max()
    return centered / radius


def chamfer_distance(
    prediction: NDArray[np.float64],
    target: NDArray[np.float64],
    normalization: str,
) -> float:
    """Chamfer distance after removing position, scale, and yaw differences.

    Both clouds are normalized independently. The distance is the mean of the two
    directed mean nearest-neighbor distances, minimized over four rotations about
    the Z axis. Upside-down or sideways predictions are not compensated.
    """
    prediction = normalize(prediction, normalization)
    target = normalize(target, normalization)
    prediction_tree, target_tree = cKDTree(prediction), cKDTree(target)
    # Rotations preserve distances: comparing target with R @ prediction equals
    # comparing R.T @ target with prediction, so both trees are built once.
    return float(min(
        (np.mean(prediction_tree.query(target @ rotation)[0])
         + np.mean(target_tree.query(prediction @ rotation.T)[0])) / 2
        for rotation in YAW_ROTATIONS
    ))
