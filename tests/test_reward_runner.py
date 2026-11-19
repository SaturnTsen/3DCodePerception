"""Outcome classification of services/execute/runner.py with a real Blender process."""
import asyncio
from pathlib import Path
import tempfile
import unittest

import numpy as np

from services.execute.config import ExecuteServiceConfig, load_config
from services.execute.runner import RenderError, mesh_path, render_paths, run_script

SERVICE_CONFIG = Path(__file__).parents[1] / 'configs' / 'service-execute.yaml'
CUBE = 'import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)\n'


def test_config() -> ExecuteServiceConfig:
    """The repository configuration with a short time limit and a tiny CPU render."""
    config = load_config(SERVICE_CONFIG)
    render = config.render.model_copy(update={'engine': 'CYCLES', 'device': 'CPU', 'samples': 1, 'resolution': 64})
    return config.model_copy(update={'script_timeout': 10.0, 'render': render})


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def run_code(self, code: str, config: ExecuteServiceConfig | None = None) -> str:
        return asyncio.run(run_script(code, self.directory, config or test_config()))

    def test_cube_is_exported_and_rendered(self):
        self.assertEqual(self.run_code(CUBE), 'ok')
        with np.load(mesh_path(self.directory)) as mesh:
            self.assertEqual(mesh['faces'].shape, (12, 3))
            np.testing.assert_allclose(np.abs(mesh['vertices']), 1.0, atol=1e-6)
        self.assertTrue(all(path.exists() for path in render_paths(self.directory)))

    def test_script_error(self):
        self.assertEqual(self.run_code(CUBE + 'raise RuntimeError("broken")\n'), 'execution_error')

    def test_script_timeout(self):
        self.assertEqual(self.run_code('import time\ntime.sleep(60)\n'), 'timeout')

    def test_empty_scene(self):
        self.assertEqual(self.run_code('pass\n'), 'invalid_geometry')

    def test_render_failure_of_non_empty_mesh_raises(self):
        config = test_config()
        # A GPU render with no visible CUDA device fails after the mesh is exported.
        render = config.render.model_copy(update={'device': 'GPU', 'gpus': ''})
        with self.assertRaisesRegex(RenderError, 'no OptiX or CUDA device'):
            self.run_code(CUBE, config.model_copy(update={'render': render}))
        self.assertTrue(mesh_path(self.directory).exists())


if __name__ == '__main__':
    unittest.main()
