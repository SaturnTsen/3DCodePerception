"""Measure one run against its target: the raw values of every metric."""
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from reward.config import RewardServiceConfig
from reward.encoders.dinov3 import Dinov3Encoder
from reward.encoders.uni3d import Uni3DEncoder
from reward.metrics.chamfer import NORMALIZATIONS, chamfer_distance
from reward.metrics.view_similarity import view_similarity
from reward.utils import sample_surface


def measure(
    dinov3: Dinov3Encoder, uni3d: Uni3DEncoder, config: RewardServiceConfig,
    mesh_file: Path, render_files: list[Path], feats: Path,
) -> dict[str, Any]:
    """Compare the exported mesh and rendered views of a run with the target feats.

    feats is the file written by reward/target_feats.py with the same encoders,
    sampling settings, and views. Raises ValueError when the mesh has no finite
    surface to sample.
    """
    images: list[Image.Image] = [Image.open(path) for path in render_files]
    for image in images:
        image.load()
    with np.load(mesh_file, allow_pickle=False) as mesh, np.load(feats, allow_pickle=False) as target:
        vertices, faces = mesh['vertices'], mesh['faces']
        points = sample_surface(vertices, faces, config.chamfer.samples, config.chamfer.seed)
        embedding = uni3d.encode_points(sample_surface(vertices, faces, config.uni3d.points, config.uni3d.seed))
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
