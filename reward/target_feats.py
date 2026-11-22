"""Target feats, computed by the reward service at start for targets that lack them.

Every directory under source_root is one target, named by its object_id. Blender
imports <object_id>.glb and exports its world-space triangle mesh (blender/import_glb.py,
blender/export_mesh.py, the same exporter as for responses), and the reference images
of the configured views are read from the directory. They give features_dir/<object_id>.npz:
    chamfer_points  (S, 3)    surface samples
    dinov3          (V, D)    DINOv3 embeddings of the reference images, in view order
    uni3d           (1024,)   Uni3D embedding of the target mesh
    clip_image      (1024,)   CLIP embedding of the first reference image
The DINOv3 and Uni3D encoders are the instances that measure responses, and the
Chamfer samples use the same settings, so both sides are encoded identically.
Existing files are kept, so delete features_dir after changing these settings.
"""
import gc
import logging
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import torch

import reward.blender
from reward.config import RewardServiceConfig
from reward.encoders.clip import ClipImageEncoder
from reward.encoders.dinov3 import Dinov3Encoder
from reward.encoders.uni3d import Uni3DEncoder
from reward.utils import sample_surface

logger = logging.getLogger(__name__)
BLENDER_SCRIPTS = Path(reward.blender.__file__).parent


def export_target(glb: Path, mesh: Path, config: RewardServiceConfig) -> None:
    """Convert a target GLB to a mesh NPZ (vertices, faces) with Blender."""
    log = mesh.with_suffix('.log')
    with log.open('w') as output:
        result = subprocess.run(
            [str(config.blender), '--background', '--factory-startup', '--python-exit-code', '1',
             '--python', str(BLENDER_SCRIPTS / 'import_glb.py'),
             '--python', str(BLENDER_SCRIPTS / 'export_mesh.py'), '--', str(mesh), str(glb)],
            stdout=output, stderr=subprocess.STDOUT, timeout=config.glb_timeout,
        )
    if result.returncode != 0:
        raise RuntimeError(f'failed to convert {glb}:\n{log.read_text(errors="replace")}')


def prepare_target_feats(dinov3: Dinov3Encoder, uni3d: Uni3DEncoder, config: RewardServiceConfig) -> None:
    """Write the feats of every target under source_root that has no feats file yet.

    The CLIP image encoder is loaded only when some target is missing and is
    released afterwards, so it does not stay in GPU memory while serving.
    """
    missing = sorted(
        directory for directory in config.source_root.iterdir()
        if directory.is_dir() and not (config.features_dir / f'{directory.name}.npz').exists())
    logger.info('target feats: %d missing in %s', len(missing), config.features_dir)
    if not missing:
        return
    config.features_dir.mkdir(parents=True, exist_ok=True)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    clip = ClipImageEncoder(config.uni3d, config.device)
    for directory in missing:
        object_id = directory.name
        images: list[Image.Image] = [Image.open(directory / view.reference) for view in config.views]
        with tempfile.TemporaryDirectory(dir=config.work_dir) as work:
            mesh_file = Path(work) / 'mesh.npz'
            export_target(directory.resolve() / f'{object_id}.glb', mesh_file, config)
            with np.load(mesh_file, allow_pickle=False) as mesh:
                vertices, faces = mesh['vertices'], mesh['faces']
                feats: dict[str, NDArray[np.float32] | NDArray[np.float64]] = {
                    'chamfer_points': sample_surface(vertices, faces, config.chamfer.samples, config.chamfer.seed),
                    'dinov3': dinov3.encode(images),
                    'uni3d': uni3d.encode_points(
                        sample_surface(vertices, faces, config.uni3d.points, config.uni3d.seed)),
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
