"""Measure one run against its target: the raw values of every metric."""
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from scipy.optimize import linear_sum_assignment

from services.reward.chamfer import NORMALIZATIONS, chamfer_distance, sample_surface
from services.reward.config import ChamferConfig
from services.reward.encoders.dinov3 import Dinov3Encoder
from services.reward.runner import mesh_path, render_paths
from services.reward.encoders.uni3d import Uni3DEncoder


def view_similarity(prediction: NDArray[np.float32], target: NDArray[np.float32]) -> dict[str, float]:
    """Compare DINOv3 view embeddings of two objects (3DHarnessBench image_similarity.py).

    view_paired: mean cosine similarity of views with the same index.
    best_assignment: mean cosine similarity under the best one-to-one view matching,
    which compensates yaw offsets between the two objects.
    """
    cosine = prediction @ target.T
    rows, columns = linear_sum_assignment(-cosine)
    return {
        'view_paired': float(np.mean(np.diag(cosine))),
        'best_assignment': float(cosine[rows, columns].mean()),
    }



def measure(
    dinov3: Dinov3Encoder, uni3d: Uni3DEncoder, chamfer: ChamferConfig, directory: Path, feats: Path,
) -> dict[str, Any]:
    """Compare the mesh and views of a successful run in directory with the target feats.

    feats is the file written by services/reward/target_feats.py with the same encoders
    and Chamfer settings. Raises ValueError when the mesh has no finite surface to sample.
    """
    images: list[Image.Image] = [Image.open(path) for path in render_paths(directory)]
    for image in images:
        image.load()
    with np.load(mesh_path(directory), allow_pickle=False) as mesh, \
            np.load(feats, allow_pickle=False) as target:
        vertices, faces = mesh['vertices'], mesh['faces']
        points = sample_surface(vertices, faces, chamfer.samples, chamfer.seed)
        embedding = uni3d.encode_mesh(vertices, faces)
        return {
            'chamfer': {
                name: chamfer_distance(points, target['chamfer_points'], name) for name in NORMALIZATIONS
            },
            'dinov3': view_similarity(dinov3.encode(images), target['dinov3']),
            'uni3d': {
                'cos_3d_3d': float(embedding @ target['uni3d']),
                'cos_image_3d': float(embedding @ target['clip_image']),
            },
        }
