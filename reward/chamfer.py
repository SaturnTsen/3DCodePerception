"""Surface sampling and alignment-normalized Chamfer distance.

All coordinates are Blender world coordinates with +Z up.
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

NORMALIZATIONS = ('rms_radius', 'max_radius')
# Rotations about the Blender up axis (+Z) by 0, 90, 180, and 270 degrees.
YAW_ROTATIONS = tuple(
    np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    for c, s in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))
)


def sample_surface(
    vertices: ArrayLike,
    faces: ArrayLike,
    count: int,
    seed: int,
) -> NDArray[np.float64]:
    """Sample count points from triangles, weighted by surface area.

    Example: points = sample_surface(vertices, faces, count=4096, seed=42)
    Input shapes: vertices (Nx3), faces (Mx3). Output shape: (count, 3).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("vertices must be finite Nx3 coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must be nonempty triangle indices")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("invalid triangle index")
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        raise ValueError("mesh has no finite surface area")
    rng = np.random.default_rng(seed)
    selected = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    u = np.sqrt(rng.random((count, 1)))
    v = rng.random((count, 1))
    return (1-u)*selected[:, 0] + u*(1-v)*selected[:, 1] + u*v*selected[:, 2]


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
