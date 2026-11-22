"""Reward service: run generated Blender code and measure it against the target.

Run: pixi run reward-service --config configs/service-reward.yaml

At start, the DINOv3 and Uni3D encoders are loaded, and target feats missing from
features_dir are computed with them (reward/target_feats.py).

POST /run    JSON {"object_id": "...", "code": "..."}
             -> {"run_status": "ok" | "execution_error" | "timeout" | "invalid_geometry",
                 "metrics": raw metric values, or null,
                 "seconds": {"queue": ..., "blender": ..., "score": ...},
                 "log": tail of the Blender log, empty when run_status is ok}
             The files of the request stay in runs_dir/<run_id>/<time>_<object_id>_<random>/:
             response.py, mesh.npz, renders/, blender.log, and result.json (the response).
GET /health  {"status": "ok"}

One request runs one Blender process (reward/script_runner.py) that executes the code,
exports the mesh, and renders four views; reward/measure.py then compares them
with the target feats in this process.
An unknown object_id is answered with HTTP 404. Failures of the generated code are
reported in run_status; failures of this service are HTTP errors, so the caller can stop.
A rendering failure of a non-empty mesh is answered with HTTP 500 and, the first time
in this process, a Bark push at the configured level (critical).
"""
import argparse
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import httpx
import uvicorn

from reward.config import AlertConfig, RewardServiceConfig, load_config
from reward.encoders.dinov3 import Dinov3Encoder
from reward.encoders.uni3d import Uni3DEncoder
from reward.measure import measure
from reward.script_runner import RenderError, log_tail, mesh_path, render_paths, run_script
from reward.target_feats import prepare_target_feats
from reward.types import RunRequest, RunResponse

logger = logging.getLogger(__name__)


async def send_alert(alert: AlertConfig) -> None:
    """Send a Bark push to the configured URL with the configured level and sound."""
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(
            f'{alert.url}/{alert.title}/奖励服务渲染失败，训练已停止，请查看服务日志',
            params={'level': alert.level, 'sound': alert.sound, 'group': alert.group, 'ttl': alert.ttl})
        response.raise_for_status()


def create_app(config: RewardServiceConfig, dinov3: Dinov3Encoder, uni3d: Uni3DEncoder) -> FastAPI:
    app = FastAPI(title='Reward service')
    slots = asyncio.Semaphore(config.max_parallel)
    alerted = asyncio.Event()
    @app.exception_handler(RenderError)
    async def render_failed(request: Request, error: RenderError) -> JSONResponse:
        logger.error('rendering failed, Blender log tail:\n%s', error)
        if not alerted.is_set():
            alerted.set()
            await send_alert(config.alert)
        return JSONResponse({'detail': 'rendering failed', 'log': str(error)}, status_code=500)

    @app.get('/health')
    def health() -> dict[str, str]:
        return {'status': 'ok'}

    @app.post('/run')
    async def run(request: RunRequest) -> RunResponse:
        features = config.features_dir / f'{request.object_id}.npz'
        if Path(request.object_id).name != request.object_id or not features.is_file():
            raise HTTPException(404, f'unknown object_id {request.object_id!r}')
        instance = f'{datetime.now():%Y%m%d-%H%M%S-%f}_{request.object_id}_{uuid4().hex[:8]}'
        directory = config.runs_dir / request.run_id / instance
        directory.mkdir(parents=True)
        arrived = time.monotonic()
        async with slots:
            started = time.monotonic()
            # Step 1: run the code in Blender, export the mesh, and render the views.
            run_status = await run_script(request.code, directory, config)
            executed = time.monotonic()
            metrics = None
            if run_status == 'ok':
                # Step 2: measure the rendered views against the target feats.
                try:
                    metrics = await run_in_threadpool(
                        measure, dinov3, uni3d, config, mesh_path(directory),
                        render_paths(directory, len(config.views)), features)
                except ValueError:
                    # The mesh has faces but no finite surface area to sample.
                    run_status = 'invalid_geometry'
            log = '' if run_status == 'ok' else log_tail(directory)
        response = RunResponse(
            run_status=run_status, metrics=metrics, log=log,
            seconds={
                'queue': started - arrived, 'blender': executed - started,
                'score': time.monotonic() - executed,
            },
        )
        (directory / 'result.json').write_text(response.model_dump_json(indent=2))
        return response

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the reward service.')
    parser.add_argument('--config', type=Path, required=True)
    config = load_config(parser.parse_args().config)
    logging.basicConfig(level=logging.INFO)
    dinov3 = Dinov3Encoder(config.dinov3.model, config.device)
    uni3d = Uni3DEncoder(config.uni3d, config.device)
    prepare_target_feats(dinov3, uni3d, config)
    uvicorn.run(create_app(config, dinov3, uni3d), host=config.host, port=config.port)


if __name__ == '__main__':
    main()
