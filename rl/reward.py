"""verl custom reward entry point.

The response code runs on gbc1: the reward service (reward) executes it in
Blender, renders it, and measures it against the target. This module extracts the
code, sends it, and combines the raw metrics into one reward. Settings come from
configs/train-3d_code_data_factory-grpo.yaml: reward.custom_reward_function.reward_kwargs.
"""
import math
import re
from typing import Any

import httpx

CODE_BLOCK = re.compile(r'```python\s*\n(.*?)```', re.DOTALL)
COMPONENTS = ('chamfer', 'dinov3', 'uni3d', 'uni3d_image')


def extract_code(response: str) -> str:
    """The single ```python block of the response, or the whole response otherwise."""
    blocks = CODE_BLOCK.findall(response)
    return blocks[0] if len(blocks) == 1 else response


def combine_scores(
    status: str,
    metrics: dict[str, Any] | None,
    *,
    chamfer_normalization: str,
    chamfer_temperature: float,
    weights: dict[str, float],
) -> dict[str, float]:
    """Turn one execution result into the reward and its logged parts.

    Components: chamfer = exp(-d / temperature), dinov3 = best-assignment view
    similarity, uni3d = point cloud vs target point cloud, uni3d_image = point cloud vs
    reference image. score is their weighted sum, and 0 when the code did not run.
    chamfer_distance is -1 when the code did not run.
    """
    if status != 'ok' or metrics is None:
        return {'score': 0.0, 'success': 0.0, 'chamfer_distance': -1.0} | dict.fromkeys(COMPONENTS, 0.0)
    distance = metrics['chamfer'][chamfer_normalization]
    components = {
        'chamfer': math.exp(-distance / chamfer_temperature),
        'dinov3': metrics['dinov3']['best_assignment'],
        'uni3d': metrics['uni3d']['cos_3d_3d'],
        'uni3d_image': metrics['uni3d']['cos_image_3d'],
    }
    score = sum(weights[name] * value for name, value in components.items())
    return {'score': score, 'success': 1.0, 'chamfer_distance': distance} | components


async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None,
    *,
    service_url: str,
    request_timeout: float,
    chamfer_normalization: str,
    chamfer_temperature: float,
    weight_chamfer: float,
    weight_dinov3: float,
    weight_uni3d: float,
    weight_uni3d_image: float,
) -> dict[str, float]:
    """Score one response against the object named by ground_truth.

    ground_truth: object_id; the reward service holds its target features.
    HTTP errors are raised so that training stops instead of learning from false zeros.
    """
    async with httpx.AsyncClient(timeout=request_timeout) as client:
        response = await client.post(
            f'{service_url}/run', json={'object_id': ground_truth, 'code': extract_code(solution_str)})
    response.raise_for_status()
    result = response.json()
    return combine_scores(
        result['status'], result['metrics'],
        chamfer_normalization=chamfer_normalization, chamfer_temperature=chamfer_temperature,
        weights={
            'chamfer': weight_chamfer, 'dinov3': weight_dinov3,
            'uni3d': weight_uni3d, 'uni3d_image': weight_uni3d_image,
        },
    )
