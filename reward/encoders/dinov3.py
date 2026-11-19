"""DINOv3 image encoder.

Follows 3DHarnessBench metrics/image_similarity.py: images are converted to RGB and
embedded with the pooled output, normalized to unit length.
"""
import threading

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


class Dinov3Encoder:
    """Image encoder. Thread safe: concurrent calls run one at a time on the GPU."""

    def __init__(self, model: str, device: str) -> None:
        self.device = device
        self.lock = threading.Lock()
        self.processor = AutoImageProcessor.from_pretrained(model, local_files_only=True)
        self.model = AutoModel.from_pretrained(model, local_files_only=True).to(device).eval()

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> NDArray[np.float32]:
        """Return unit embeddings with shape (len(images), D)."""
        inputs = self.processor(images=[image.convert('RGB') for image in images], return_tensors='pt')
        with self.lock:
            features = self.model(**inputs.to(self.device)).pooler_output.float()
        return (features / features.norm(dim=-1, keepdim=True)).cpu().numpy()

