"""HTTP interface of services/score with the Chamfer metric only (no GPU models load)."""
from io import BytesIO
import unittest

from fastapi.testclient import TestClient
import numpy as np

from services.score.app import create_app
from services.score.config import ScoreServiceConfig
from services.score.scorer import Scorer
from services.score.targets import target_features

CUBE_VERTICES = np.array(
    [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=np.float64)
CUBE_FACES = np.array([
    [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
])
CONFIG = ScoreServiceConfig.model_validate({
    'host': '127.0.0.1', 'port': 8710, 'device': 'cpu', 'metrics': ['chamfer'],
    'chamfer': {'samples': 1024, 'seed': 42},
    'dinov3': {'model': 'unused'},
    'uni3d': {
        'checkpoint': 'unused', 'backbone': 'unused', 'backbone_dim': 1, 'embed_dim': 1,
        'group_size': 1, 'num_group': 1, 'encoder_dim': 1, 'points': 1, 'seed': 0,
        'clip_model': 'unused', 'clip_pretrained': 'unused',
    },
})


def npz(**arrays: np.ndarray) -> bytes:
    buffer = BytesIO()
    np.savez(buffer, allow_pickle=False, **arrays)
    return buffer.getvalue()


class ScoreServiceTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(create_app(CONFIG))

    def features(self) -> bytes:
        mesh = {'vertices': CUBE_VERTICES, 'faces': CUBE_FACES}
        return npz(**target_features(Scorer(CONFIG), None, mesh, []))

    def test_health(self):
        self.assertEqual(self.client.get('/health').json(), {'status': 'ok', 'metrics': ['chamfer']})

    def test_target_features_contain_enabled_metrics_only(self):
        mesh = {'vertices': CUBE_VERTICES, 'faces': CUBE_FACES}
        self.assertEqual(set(target_features(Scorer(CONFIG), None, mesh, [])), {'chamfer_points'})

    def test_score_moved_and_scaled_cube(self):
        target = np.frombuffer(self.features(), dtype=np.uint8)
        moved = CUBE_VERTICES * 3 + np.array([5.0, -2.0, 1.0])
        response = self.client.post('/score', content=npz(vertices=moved, faces=CUBE_FACES, target=target))
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(set(result), {'chamfer'})
        self.assertLess(result['chamfer']['rms_radius'], 0.05)

    def test_invalid_mesh_rejected_with_422(self):
        target = np.frombuffer(self.features(), dtype=np.uint8)
        response = self.client.post(
            '/score', content=npz(vertices=CUBE_VERTICES, faces=np.zeros((0, 3), dtype=np.int64), target=target))
        self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main()
