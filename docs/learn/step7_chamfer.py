"""第 7 步：Chamfer 距离的细节。对书桌做几种破坏，分别看两个方向的距离。

pixi run python docs/learn/step7_chamfer.py
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.geometry import sample_surface  # noqa: E402

COUNT, SEED = 4096, 42
Mesh = tuple[NDArray[np.float64], NDArray[np.int64]]


def box(center: tuple[float, float, float], size: float) -> Mesh:
    """边长 size 的立方体，12 个三角形。"""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])
    return corners * size / 2 + np.array(center), faces


def merge(a: Mesh, b: Mesh) -> Mesh:
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1] + len(a[0])])


def rotate_z(mesh: Mesh, degrees: float) -> Mesh:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return mesh[0] @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T, mesh[1]


def distances(target: Mesh, prediction: Mesh) -> dict[str, float]:
    """两个方向分开算，并给出普通距离和平方距离两种平均方式（都按对角线归一化）。"""
    t = sample_surface(*target, COUNT, SEED)
    p = sample_surface(*prediction, COUNT, SEED + 1)
    diagonal = float(np.linalg.norm(np.ptp(target[0], axis=0)))
    t_to_p = np.asarray(cKDTree(p).query(t)[0]) / diagonal
    p_to_t = np.asarray(cKDTree(t).query(p)[0]) / diagonal
    return {
        '目标→生成': float(t_to_p.mean()),
        '生成→目标': float(p_to_t.mean()),
        'd（奖励）': float(t_to_p.mean() + p_to_t.mean()) / 2,
        '平方距离版': float((t_to_p ** 2).mean() + (p_to_t ** 2).mean()) / 2,
    }


def main() -> None:
    with np.load('docs/learn/SimpleDeskFactory.npz') as data:
        desk: Mesh = data['vertices'].astype(np.float64), data['faces'].astype(np.int64)
    vertices, faces = desk
    centers = vertices[faces].mean(axis=1)
    leg = (centers[:, 2] < 0.33) & (centers[:, 0] > 0) & (centers[:, 1] > 0)
    areas = np.linalg.norm(np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]],
    ), axis=1) / 2
    print(f'书桌包围盒对角线 {np.linalg.norm(np.ptp(vertices, axis=0)):.3f} m，'
          f'一条桌腿占总表面积 {areas[leg].sum() / areas.sum():.1%}\n')

    cases: dict[str, Mesh] = {
        '完全相同（噪声底）': desk,
        '删掉一条桌腿': (vertices, faces[~leg]),
        '桌面上多一个 20cm 方块': merge(desk, box((0.0, 0.0, 0.465), 0.2)),
        '远处多一个 2cm 小方块': merge(desk, box((0.0, 0.0, 1.5), 0.02)),
        '整体平移 2cm': (vertices + np.array([0.02, 0.0, 0.0]), faces),
        '整体平移 10cm': (vertices + np.array([0.10, 0.0, 0.0]), faces),
        '整体放大 10%': (vertices * 1.1, faces),
        '绕竖直轴转 90°': rotate_z(desk, 90),
        '绕竖直轴转 180°': rotate_z(desk, 180),
    }
    print(f'{"破坏方式":<22}{"目标→生成":>10}{"生成→目标":>10}{"d（奖励）":>10}{"平方距离版":>12}')
    for label, mesh in cases.items():
        row = distances(desk, mesh)
        print(f'{label:<22}' + ''.join(f'{value:>11.4f}' for value in row.values()))


if __name__ == '__main__':
    main()
