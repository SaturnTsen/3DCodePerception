"""Data shared by the reward service and its client rl/reward.py.

rl/ may import this module only; it depends on nothing else in reward/.
"""
from typing import Literal

from pydantic import BaseModel

# Outcome of running the generated code, not the HTTP status of the request.
# ok: ran and measured; execution_error: the script raised or exited before the mesh
# export; timeout: the script or rendering exceeded its limit; invalid_geometry: the
# exported mesh has no finite surface to sample.
RunStatus = Literal['ok', 'execution_error', 'timeout', 'invalid_geometry']
ChamferNormalization = Literal['rms_radius', 'max_radius']


class Dinov3Metrics(BaseModel):
    """Mean cosine similarity of the rendered views with the reference images."""

    view_paired: float
    best_assignment: float


class Uni3DMetrics(BaseModel):
    """Cosine similarity of the Uni3D embedding of the generated point cloud."""

    cos_3d_3d: float
    cos_image_3d: float


class MeasureResult(BaseModel):
    """Raw metric values of one run measured against its target."""

    chamfer: dict[ChamferNormalization, float]
    dinov3: Dinov3Metrics
    uni3d: Uni3DMetrics


class RunRequest(BaseModel):
    """Body of POST /run."""

    object_id: str
    code: str


class RunResponse(BaseModel):
    """Answer of POST /run; metrics is None unless run_status is ok."""

    run_status: RunStatus
    metrics: MeasureResult | None
    seconds: dict[str, float]
    log: str
