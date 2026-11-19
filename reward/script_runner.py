"""Run one generated Blender script: execute, export the mesh, render four views.

All three steps run in one Blender process. The script time limit covers the
generated script only: once mesh.npz exists, rendering gets its own time limit.
"""
import asyncio
import os
from pathlib import Path
import signal
from typing import Literal

import numpy as np

import reward.blender
from reward.config import RewardServiceConfig

Status = Literal['ok', 'execution_error', 'timeout', 'invalid_geometry']

BLENDER_SCRIPTS = Path(reward.blender.__file__).parent
CLEAR_SCENE = 'import bpy; bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)'
LOG_TAIL_CHARACTERS = 4000


def mesh_path(directory: Path) -> Path:
    return directory / 'mesh.npz'


def render_paths(directory: Path, count: int) -> list[Path]:
    return [directory / 'renders' / f'view_{index}.png' for index in range(count)]


def log_tail(directory: Path) -> str:
    return (directory / 'blender.log').read_text(errors='replace')[-LOG_TAIL_CHARACTERS:]


class RenderError(Exception):
    """A non-empty mesh was exported but the four views were not rendered.

    The model produced geometry, so the failure is attributed to the renderer
    (out of GPU memory, missing device, crash) and must stop training instead
    of becoming a zero reward. The message is the tail of the Blender log.
    """


def has_faces(directory: Path) -> bool:
    with np.load(mesh_path(directory), allow_pickle=False) as mesh:
        return len(mesh['faces']) > 0


async def _kill(process: asyncio.subprocess.Process) -> None:
    os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def run_script(code: str, directory: Path, config: RewardServiceConfig) -> Status:
    """Run code in a fresh Blender process inside directory and classify the outcome.

    execution_error: the script raised or exited before the mesh was exported.
    timeout: the script or the rendering exceeded its time limit.
    invalid_geometry: the exported mesh has no faces.
    Raises RenderError when a mesh with faces was exported but rendering failed.
    """
    script = directory / 'response.py'
    script.write_text(code)
    render = config.render
    command = [
        str(config.blender), '--background', '--factory-startup', '--threads', str(config.blender_threads),
        '--python-exit-code', '1', '--python-expr', CLEAR_SCENE,
        '--python', str(script),
        '--python', str(BLENDER_SCRIPTS / 'export_mesh.py'),
        '--python', str(BLENDER_SCRIPTS / 'render.py'),
        '--', str(mesh_path(directory)), '--render-dir', str(directory / 'renders'),
        '--engine', render.engine, '--device', render.device,
        '--samples', str(render.samples), '--resolution', str(render.resolution),
        '--azimuths', *(str(view.azimuth) for view in config.views),
    ]
    with (directory / 'blender.log').open('w') as log:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=log, stderr=asyncio.subprocess.STDOUT, cwd=directory, start_new_session=True,
            env=os.environ | {'CUDA_VISIBLE_DEVICES': render.gpus})
        try:
            await asyncio.wait_for(process.wait(), config.script_timeout)
        except TimeoutError:
            if not mesh_path(directory).exists():
                await _kill(process)
                return 'timeout'
            try:
                await asyncio.wait_for(process.wait(), config.render_timeout)
            except TimeoutError:
                await _kill(process)
                return 'timeout'
    if not mesh_path(directory).exists():
        return 'execution_error'
    if not has_faces(directory):
        return 'invalid_geometry'
    rendered = all(path.exists() for path in render_paths(directory, len(config.views)))
    if process.returncode != 0 or not rendered:
        raise RenderError(log_tail(directory))
    return 'ok'
