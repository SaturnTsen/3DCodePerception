"""Area-weighted point sampling on triangle mesh surfaces.

Coordinates are Blender world coordinates with +Z up.
"""
import numpy as np
from numpy.typing import ArrayLike, NDArray


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
