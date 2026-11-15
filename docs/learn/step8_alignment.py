"""第 8 步：对齐与归一化。比较几种对齐方式下，各种破坏的扣分。

pixi run python docs/learn/step8_alignment.py
"""

import itertools
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.geometry import sample_surface  # noqa: E402
from step7_chamfer import Mesh, box, merge  # noqa: E402

COUNT, SEED = 4096, 42
Points = NDArray[np.float64]


def rotation(axis: int, degrees: float) -> NDArray[np.float64]:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    i, j = [k for k in range(3) if k != axis]
    matrix = np.eye(3)
    matrix[i, i], matrix[i, j], matrix[j, i], matrix[j, j] = c, -s, s, c
    return matrix


YAW_4 = [rotation(2, angle) for angle in (0, 90, 180, 270)]
AXIS_24 = [
    matrix for permutation in itertools.permutations(range(3))
    for signs in itertools.product((1, -1), repeat=3)
    if np.linalg.det(matrix := np.eye(3)[list(permutation)] * np.array(signs)) > 0
]


def by_bbox(points: Points) -> Points:
    """包围盒中心移到原点，再除以包围盒对角线长度。"""
    low, high = points.min(axis=0), points.max(axis=0)
    return (points - (low + high) / 2) / np.linalg.norm(high - low)


def by_sphere(points: Points) -> Points:
    """benchmark 的做法：减去质心，再除以离质心最远的点的距离。"""
    centered = points - points.mean(axis=0)
    return centered / np.linalg.norm(centered, axis=1).max()


def by_rms(points: Points) -> Points:
    """减去质心，再除以所有点到质心的均方根距离（RMS 半径）。"""
    centered = points - points.mean(axis=0)
    return centered / np.sqrt((centered ** 2).sum(axis=1).mean())


def chamfer(a: Points, b: Points) -> float:
    return float(np.asarray(cKDTree(b).query(a)[0]).mean()
                 + np.asarray(cKDTree(a).query(b)[0]).mean()) / 2


def score(
    target: Points, prediction: Points,
    normalize: Callable[[Points], Points], rotations: list[NDArray[np.float64]],
) -> float:
    t, p = normalize(target), normalize(prediction)
    return min(chamfer(t, p @ r.T) for r in rotations)


def transform(mesh: Mesh, matrix: NDArray[np.float64]) -> Mesh:
    return mesh[0] @ matrix.T, mesh[1]


def main() -> None:
    with np.load('docs/learn/SimpleDeskFactory.npz') as data:
        desk: Mesh = data['vertices'].astype(np.float64), data['faces'].astype(np.int64)
    vertices, faces = desk
    centers = vertices[faces].mean(axis=1)
    leg = (centers[:, 2] < 0.33) & (centers[:, 0] > 0) & (centers[:, 1] > 0)

    cases: dict[str, Mesh] = {
        '完全相同（噪声底）': desk,
        '整体平移 10cm': (vertices + np.array([0.1, 0.0, 0.0]), faces),
        '整体缩小一半': (vertices * 0.5, faces),
        '水平转 90°': transform(desk, rotation(2, 90)),
        '水平转 30°': transform(desk, rotation(2, 30)),
        '删掉一条桌腿': (vertices, faces[~leg]),
        '桌面上多一个 20cm 方块': merge(desk, box((0.0, 0.0, 0.465), 0.2)),
        '远处多一个 2cm 小方块': merge(desk, box((0.0, 0.0, 1.5), 0.02)),
        '桌面长边拉长 30%': (vertices * np.array([1.0, 1.3, 1.0]), faces),
        '上下颠倒': transform(desk, rotation(0, 180)),
        '侧放（绕 x 轴转 90°）': transform(desk, rotation(0, 90)),
    }
    methods = {
        '不对齐（现状）': None,
        '包围盒+4 个水平角': (by_bbox, YAW_4),
        '包围盒+24 个旋转': (by_bbox, AXIS_24),
        '单位球+4 个水平角': (by_sphere, YAW_4),
        'RMS+4 个水平角': (by_rms, YAW_4),
    }
    target = sample_surface(*desk, COUNT, SEED)
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    print(f'{"破坏方式":<22}' + ''.join(f'{name:>16}' for name in methods))
    for label, mesh in cases.items():
        prediction = sample_surface(*mesh, COUNT, SEED + 1)
        row = []
        for method in methods.values():
            if method is None:
                row.append(chamfer(target, prediction) / diagonal)
            else:
                row.append(score(target, prediction, *method))
        print(f'{label:<22}' + ''.join(f'{value:>16.4f}' for value in row))
    print('\n注：各列的归一化尺度不同（对角线、最远点半径、RMS 半径），'
          '数值只能和本列第一行的噪声底比较。')


if __name__ == '__main__':
    main()
