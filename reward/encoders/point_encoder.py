"""Uni3D point cloud encoder.

Adapted from models/point_encoder.py of https://github.com/baaivision/Uni3D
(commit 64e03c3, MIT License, Copyright (c) 2023 BAAI-Vision). Module and parameter
names are unchanged so the released checkpoints load directly. The only functional
change is farthest point sampling, which is written in PyTorch here instead of
calling the compiled pointnet2_ops extension. Training-only patch dropout is removed.
"""
from timm.models.eva import Eva
import torch
from torch import nn


def farthest_point_sample(points: torch.Tensor, count: int) -> torch.Tensor:
    """Return count sampled points per cloud, starting from the first point.

    points: (B, N, 3). Output: (B, count, 3).
    """
    batch, total, _ = points.shape
    indices = torch.zeros(batch, count, dtype=torch.long, device=points.device)
    distance = torch.full((batch, total), float('inf'), device=points.device, dtype=points.dtype)
    farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    for step in range(count):
        indices[:, step] = farthest
        center = points[rows, farthest].unsqueeze(1)
        distance = torch.minimum(distance, ((points - center) ** 2).sum(-1))
        farthest = distance.argmax(-1)
    return torch.gather(points, 1, indices.unsqueeze(-1).expand(-1, -1, 3))


def nearest_neighbors(count: int, points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """Indices of the count nearest points for each center. Output: (B, G, count)."""
    return torch.cdist(centers, points).topk(count, dim=-1, largest=False, sorted=False).indices


class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int) -> None:
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor, color: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a cloud into local patches. Returns centers (B, G, 3) and features (B, G, M, 6)."""
        center = farthest_point_sample(xyz, self.num_group)
        index = nearest_neighbors(self.group_size, xyz, center)
        batch = torch.arange(xyz.shape[0], device=xyz.device).view(-1, 1, 1)
        neighborhood = xyz[batch, index] - center.unsqueeze(2)
        return center, torch.cat((neighborhood, color[batch, index]), dim=-1)


class Encoder(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(6, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """point_groups: (B, G, M, 6). Output: (B, G, C)."""
        batch, groups, size, _ = point_groups.shape
        feature = self.first_conv(point_groups.reshape(batch * groups, size, 6).transpose(2, 1))
        pooled = feature.max(dim=2, keepdim=True)[0]
        feature = self.second_conv(torch.cat((pooled.expand(-1, -1, size), feature), dim=1))
        return feature.max(dim=2)[0].reshape(batch, groups, self.encoder_channel)


class PointcloudEncoder(nn.Module):
    def __init__(
        self, point_transformer: Eva, *, backbone_dim: int, embed_dim: int,
        group_size: int, num_group: int, encoder_dim: int,
    ) -> None:
        super().__init__()
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.encoder = Encoder(encoder_channel=encoder_dim)
        self.encoder2trans = nn.Linear(encoder_dim, backbone_dim)
        self.trans2embed = nn.Linear(backbone_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, backbone_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, backbone_dim))
        self.pos_embed = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, backbone_dim))
        self.visual = point_transformer

    def forward(self, points: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        center, features = self.group_divider(points, colors)
        tokens = self.encoder2trans(self.encoder(features))
        batch = tokens.shape[0]
        x = torch.cat((self.cls_token.expand(batch, -1, -1), tokens), dim=1)
        x = x + torch.cat((self.cls_pos.expand(batch, -1, -1), self.pos_embed(center)), dim=1)
        x = self.visual.pos_drop(x)
        for block in self.visual.blocks:
            x = block(x)
        x = self.visual.fc_norm(self.visual.norm(x[:, 0, :]))
        return self.trans2embed(x)


class Uni3D(nn.Module):
    """Container matching the checkpoint layout: logit_scale plus point_encoder."""

    def __init__(self, point_encoder: PointcloudEncoder) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.point_encoder = point_encoder
