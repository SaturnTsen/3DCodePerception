"""Target feats, computed by the reward service at start for targets that lack them.

For every record of the prepared train and val Parquet files (outputs of the data
preparation config), the target mesh target_dir/<object_id>.npz and the reference
views stored in the record give features_dir/<object_id>.npz:
    chamfer_points  (S, 3)    surface samples
    dinov3          (V, D)    DINOv3 embeddings of the reference views
    uni3d           (1024,)   Uni3D embedding of the target mesh
    clip_image      (1024,)   CLIP embedding of the first reference view
The DINOv3 and Uni3D encoders are the instances that measure predictions, and the
Chamfer samples use the same settings, so both sides are encoded identically.
Existing files are kept, so delete features_dir after changing these settings.
"""
import gc
from io import BytesIO
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
import torch
import yaml

from services.reward.chamfer import sample_surface
from services.reward.config import RewardServiceConfig
from services.reward.encoders.clip import ClipImageEncoder
from services.reward.encoders.dinov3 import Dinov3Encoder
from services.reward.encoders.uni3d import Uni3DEncoder

logger = logging.getLogger(__name__)


class PreparedOutputs(BaseModel):
    model_config = ConfigDict(strict=True)
    train: Path = Field(strict=False)
    val: Path = Field(strict=False)


class PreparedData(BaseModel):
    """The fields of the data preparation config that locate its outputs."""
    model_config = ConfigDict(extra='ignore', strict=True)
    outputs: PreparedOutputs
    target_dir: Path = Field(strict=False)


def prepare_target_feats(dinov3: Dinov3Encoder, uni3d: Uni3DEncoder, config: RewardServiceConfig) -> None:
    """Write the feats of every prepared target that has no feats file yet.

    The CLIP image encoder is loaded only when some target is missing and is
    released afterwards, so it does not stay in GPU memory while serving.
    """
    data = PreparedData.model_validate(yaml.safe_load(config.data_config.read_text(encoding='utf-8')))
    missing: list[tuple[str, list[bytes]]] = []
    for parquet in (data.outputs.train, data.outputs.val):
        frame = pd.read_parquet(parquet, columns=['extra_info', 'images'])
        for extra_info, views in zip(frame['extra_info'], frame['images']):
            object_id = extra_info['object_id']
            if not (config.features_dir / f'{object_id}.npz').exists():
                missing.append((object_id, [view['bytes'] for view in views]))
    logger.info('target feats: %d missing in %s', len(missing), config.features_dir)
    if not missing:
        return
    config.features_dir.mkdir(parents=True, exist_ok=True)
    clip = ClipImageEncoder(config.uni3d, config.device)
    for object_id, views in missing:
        images: list[Image.Image] = [Image.open(BytesIO(view)) for view in views]
        with np.load(data.target_dir / f'{object_id}.npz', allow_pickle=False) as mesh:
            vertices, faces = mesh['vertices'], mesh['faces']
            feats: dict[str, NDArray[Any]] = {
                'chamfer_points': sample_surface(vertices, faces, config.chamfer.samples, config.chamfer.seed),
                'dinov3': dinov3.encode(images),
                'uni3d': uni3d.encode_mesh(vertices, faces),
                'clip_image': clip.encode_image(images[0]),
            }
        # Write under a temporary name so an interrupted start leaves no partial file.
        partial = config.features_dir / f'{object_id}.partial.npz'
        np.savez(partial, allow_pickle=False, **feats)
        partial.replace(config.features_dir / f'{object_id}.npz')
    del clip
    gc.collect()
    torch.cuda.empty_cache()
    logger.info('target feats: wrote %d files', len(missing))
