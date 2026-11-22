"""The reward service end to end: real encoders on GPU, real Blender processes, and target
feats prepared from a source_root with one cube object (GLB and reference images)."""
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import unquote

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from reward.app import create_app
from reward.config import RewardServiceConfig, load_config
from reward.encoders.dinov3 import Dinov3Encoder
from reward.encoders.uni3d import Uni3DEncoder
from reward.script_runner import CLEAR_SCENE
from reward.target_feats import prepare_target_feats

SERVICE_CONFIG = Path(__file__).parents[1] / 'configs' / 'service-reward.yaml'
# GPU 0 serves the reward service; tests use GPU 1.
DEVICE = 'cuda:1'
CUBE = 'import bpy\nbpy.ops.mesh.primitive_cube_add(size=2)\n'
EXPORT_GLB = CUBE + 'import sys\nbpy.ops.export_scene.gltf(filepath=sys.argv[-1], export_format="GLB")\n'
FEATS = {'chamfer_points', 'dinov3', 'uni3d', 'clip_image'}


class Bark(BaseHTTPRequestHandler):
    """Fake Bark server that records request paths."""
    received: list[str] = []

    def do_GET(self):
        Bark.received.append(unquote(self.path))
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, format, *args):
        pass


class RewardServiceTest(unittest.TestCase):
    root: Path
    bark: HTTPServer
    config: RewardServiceConfig
    dinov3: Dinov3Encoder
    uni3d: Uni3DEncoder

    @classmethod
    def setUpClass(cls):
        cls.root = Path(cls.enterClassContext(tempfile.TemporaryDirectory(dir='temp'))).resolve()
        config = load_config(SERVICE_CONFIG)
        cube = cls.root / 'source' / 'Cube'
        for view in config.views:
            (cube / view.reference).parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (64, 64), (128, 128, 128)).save(cube / view.reference)
        subprocess.run(
            [str(config.blender), '--background', '--factory-startup', '--python-exit-code', '1',
             '--python-expr', CLEAR_SCENE + '\n' + EXPORT_GLB, '--', str(cube / 'Cube.glb')],
            check=True, capture_output=True, timeout=config.glb_timeout)
        cls.bark = HTTPServer(('127.0.0.1', 0), Bark)
        threading.Thread(target=cls.bark.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.bark.server_close)
        cls.addClassCleanup(cls.bark.shutdown)
        # The repository configuration with a tiny CPU render and a fake Bark server.
        cls.config = config.model_copy(update={
            'source_root': cls.root / 'source', 'features_dir': cls.root / 'features', 'work_dir': cls.root / 'work',
            'device': DEVICE, 'script_timeout': 10.0,
            'render': config.render.model_copy(update={'device': 'CPU', 'samples': 1, 'resolution': 64}),
            'alert': config.alert.model_copy(update={'url': f'http://127.0.0.1:{cls.bark.server_port}/KEY'}),
        })
        cls.dinov3 = Dinov3Encoder(cls.config.dinov3.model, DEVICE)
        cls.uni3d = Uni3DEncoder(cls.config.uni3d, DEVICE)
        prepare_target_feats(cls.dinov3, cls.uni3d, cls.config)

    def setUp(self):
        Bark.received.clear()

    def client(self, config: RewardServiceConfig) -> TestClient:
        return TestClient(create_app(config, self.dinov3, self.uni3d))

    def run_code(self, client: TestClient, code: str, object_id: str = 'Cube'):
        return client.post('/run', json={'object_id': object_id, 'code': code})

    def test_target_feats_are_written_once(self):
        path = self.config.features_dir / 'Cube.npz'
        with np.load(path) as feats:
            self.assertEqual(set(feats.files), FEATS)
            self.assertEqual(feats['dinov3'].shape[0], len(self.config.views))
        written = path.stat().st_mtime_ns
        prepare_target_feats(self.dinov3, self.uni3d, self.config)
        self.assertEqual(path.stat().st_mtime_ns, written)

    def test_run_outcomes(self):
        client = self.client(self.config)
        response = self.run_code(client, CUBE)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result['run_status'], 'ok')
        metrics = result['metrics']
        self.assertLess(metrics['chamfer']['rms_radius'], 0.05)
        self.assertGreater(metrics['uni3d']['cos_3d_3d'], 0.95)
        self.assertTrue(-1 <= metrics['uni3d']['cos_image_3d'] <= 1)
        self.assertTrue(-1 <= metrics['dinov3']['best_assignment'] <= 1)
        self.assertEqual(self.run_code(client, 'pass\n').json()['run_status'], 'invalid_geometry')
        self.assertEqual(self.run_code(client, 'raise RuntimeError\n').json()['run_status'], 'execution_error')
        self.assertEqual(self.run_code(client, CUBE, 'Missing').status_code, 404)
        self.assertEqual(self.run_code(client, CUBE, '../Cube').status_code, 404)

    def test_render_failure_returns_500_and_alerts_once(self):
        # A GPU render with no visible CUDA device fails after the mesh is exported.
        render = self.config.render.model_copy(update={'device': 'GPU', 'gpus': ''})
        client = self.client(self.config.model_copy(update={'render': render}))
        for _ in range(2):
            response = self.run_code(client, CUBE)
            self.assertEqual(response.status_code, 500)
            self.assertIn('no OptiX or CUDA device', response.json()['log'])
        self.assertEqual(len(Bark.received), 1)
        self.assertTrue(Bark.received[0].startswith('/KEY/3DCodePerception/'))
        self.assertIn('level=critical', Bark.received[0])


if __name__ == '__main__':
    unittest.main()
