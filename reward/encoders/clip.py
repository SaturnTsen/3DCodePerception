"""CLIP image encoder that Uni3D point cloud embeddings are aligned with.

Only needed for the target side: it embeds the first reference view of each target.
"""
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray
import open_clip
from PIL import Image
import torch

from services.reward.config import Uni3DConfig
from services.reward.encoders.uni3d import unit


class ClipImageEncoder:
    def __init__(self, config: Uni3DConfig, device: str) -> None:
        self.device = device
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.clip_model, pretrained=config.clip_pretrained)
        # open_clip has no return annotations; EVA02-E-14-plus builds a CustomTextCLIP.
        self.preprocess = cast(Callable[[Image.Image], torch.Tensor], preprocess)
        self.model = cast(open_clip.CustomTextCLIP, model).to(device, torch.float16).eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> NDArray[np.float32]:
        pixels = self.preprocess(image.convert('RGB')).unsqueeze(0)
        return unit(self.model.encode_image(pixels.to(self.device, torch.float16)))
