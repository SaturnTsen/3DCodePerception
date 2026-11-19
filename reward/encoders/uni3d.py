"""Uni3D point cloud encoder. Its embeddings are aligned with the CLIP image encoder
in clip.py.

Follows 3DHarnessBench metrics/shape_uni3d.py: points are centered, scaled into the
unit sphere, and colored uniform grey 0.4. The benchmark feeds glTF coordinates
(+Y up), so Blender coordinates (+Z up) are converted before encoding.
"""
import threading
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
import timm
from timm.models.eva import Eva
import torch

from services.reward.chamfer import sample_surface
from services.reward.config import Uni3DConfig
from services.reward.encoders.point_encoder import PointcloudEncoder, Uni3D

GREY = 0.4
CLASSIFIER_HEAD = ('point_encoder.visual.head.weight', 'point_encoder.visual.head.bias')
# Blender (+Z up, -Y front) to glTF (+Y up, +Z front): x' = x, y' = z, z' = -y.
BLENDER_TO_GLTF = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def unit(embedding: torch.Tensor) -> NDArray[np.float32]:
    embedding = embedding.float()
    return (embedding / embedding.norm(dim=-1, keepdim=True)).squeeze(0).cpu().numpy()


class Uni3DEncoder:
    """Point cloud encoder. Thread safe: concurrent calls run one at a time on the GPU."""

    def __init__(self, config: Uni3DConfig, device: str) -> None:
        self.config = config
        self.device = device
        self.lock = threading.Lock()
        # Uni3D backbones are timm EVA models; create_model is annotated as returning nn.Module.
        backbone = cast(Eva, timm.create_model(config.backbone, drop_path_rate=0.0, num_classes=0))
        model = Uni3D(PointcloudEncoder(
            backbone, backbone_dim=config.backbone_dim, embed_dim=config.embed_dim,
            group_size=config.group_size, num_group=config.num_group, encoder_dim=config.encoder_dim,
        ))
        state = torch.load(config.checkpoint, map_location='cpu', weights_only=False)['module']
        # The backbone is built without a classifier (num_classes=0), so its head is unused.
        for name in CLASSIFIER_HEAD:
            del state[name]
        model.load_state_dict(state)
        self.point_encoder = model.point_encoder.to(device).eval()

    def encode_mesh(self, vertices: NDArray[Any], faces: NDArray[Any]) -> NDArray[np.float32]:
        """Sample config.points surface points with config.seed and embed them.

        Raises ValueError when the mesh has no finite surface to sample.
        """
        return self.encode_points(sample_surface(vertices, faces, self.config.points, self.config.seed))

    @torch.no_grad()
    def encode_points(self, points: NDArray[np.float64]) -> NDArray[np.float32]:
        """Embed Blender-space surface points (Nx3) as a unit vector."""
        centered = (points - points.mean(axis=0)) @ BLENDER_TO_GLTF.T
        centered = centered / np.linalg.norm(centered, axis=1).max()
        with self.lock:
            xyz = torch.from_numpy(centered).to(self.device, torch.float32).unsqueeze(0)
            return unit(self.point_encoder(xyz, torch.full_like(xyz, GREY)))

