"""Reward service configuration. Every field must be written in the YAML file."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import yaml


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class RenderConfig(StrictModel):
    engine: Literal['CYCLES', 'BLENDER_EEVEE']
    device: Literal['GPU', 'CPU']
    samples: int = Field(gt=0)
    resolution: int = Field(gt=0)
    # CUDA_VISIBLE_DEVICES for Blender, e.g. "1"; keeps renders off the encoders' GPU.
    gpus: str


class AlertConfig(StrictModel):
    """Bark push sent once per service process when rendering fails."""
    # https://api.day.app/<device key>
    url: str
    title: str
    level: Literal['active', 'timeSensitive', 'passive', 'critical']
    sound: str
    group: str
    ttl: int = Field(gt=0)


class ChamferConfig(StrictModel):
    samples: int = Field(gt=0)
    seed: int


class Dinov3Config(StrictModel):
    model: str


class Uni3DConfig(StrictModel):
    checkpoint: Path = Field(strict=False)
    backbone: str
    backbone_dim: int = Field(gt=0)
    embed_dim: int = Field(gt=0)
    group_size: int = Field(gt=0)
    num_group: int = Field(gt=0)
    encoder_dim: int = Field(gt=0)
    points: int = Field(gt=0)
    seed: int
    clip_model: str
    clip_pretrained: str


class RewardServiceConfig(StrictModel):
    host: str
    port: int = Field(gt=0, lt=65536)
    # Data preparation config: its Parquet outputs and target_dir define the targets.
    data_config: Path = Field(strict=False)
    features_dir: Path = Field(strict=False)
    work_dir: Path = Field(strict=False)
    blender: Path = Field(strict=False)
    blender_threads: int = Field(gt=0)
    max_parallel: int = Field(gt=0)
    script_timeout: float = Field(gt=0)
    render_timeout: float = Field(gt=0)
    render: RenderConfig
    alert: AlertConfig
    # Torch device of the encoders.
    device: str
    chamfer: ChamferConfig
    dinov3: Dinov3Config
    uni3d: Uni3DConfig


def load_config(path: Path) -> RewardServiceConfig:
    return RewardServiceConfig.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
