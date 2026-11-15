"""第 6 步：表面采样，从网格到点云。测量奖励的"噪声底"。

pixi run python docs/learn/step6_sampling.py

目标网格读自第一次审查时导出到 temp/review/ 的 NPZ。
"""

import sys
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.geometry import compare, sample_surface  # noqa: E402

NAMES = ('SimpleDeskFactory', 'CupFactory', 'LiteDoorFactory', 'SpiralStaircaseFactory',
         'MushroomFactory')
COUNTS = (1024, 4096, 8192, 32768)
SEED = 42


def load(name: str) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    with np.load(f'temp/review/{name}.npz') as mesh:
        return mesh['vertices'].astype(np.float64), mesh['faces'].astype(np.int64)


def subdivide(vertices: NDArray[np.float64], faces: NDArray[np.int64]):
    """每个三角形按三条边的中点切成 4 个：形状完全不变，三角形划分不同。"""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    n, m = len(vertices), len(faces)
    midpoints = np.concatenate([
        (vertices[a] + vertices[b]) / 2, (vertices[b] + vertices[c]) / 2,
        (vertices[c] + vertices[a]) / 2,
    ])
    ab, bc, ca = n + np.arange(m), n + m + np.arange(m), n + 2 * m + np.arange(m)
    new_faces = np.concatenate([
        np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
        np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1),
    ])
    return np.concatenate([vertices, midpoints]), new_faces


def chamfer(vertices_a, faces_a, vertices_b, faces_b, count: int, seed_a: int, seed_b: int) -> float:
    """和 rl/reward.py 相同：按目标包围盒对角线归一化的双向平均最近距离。"""
    target = sample_surface(vertices_a, faces_a, count, seed_a)
    prediction = sample_surface(vertices_b, faces_b, count, seed_b)
    scale = float(np.linalg.norm(np.ptp(vertices_a, axis=0)))
    return compare(prediction, target, scale)['chamfer']


def main() -> None:
    vertices, faces = load('SimpleDeskFactory')
    points = sample_surface(vertices, faces, 4096, SEED)
    areas = np.linalg.norm(np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    ), axis=1) / 2
    top = points[:, 2] > 0.36
    print('1. 书桌按面积采样 4096 个点')
    print(f'   桌面上表面占总面积 {areas[np.abs(vertices[faces].mean(1)[:, 2] - 0.3646) < 1e-3].sum() / areas.sum():.1%}，'
          f'落在上表面附近的点占 {top.mean():.1%}')

    print('\n2. 同一个形状、同一个种子，只改三角形划分（细分一次）')
    fine_vertices, fine_faces = subdivide(vertices, faces)
    print(f'   三角形数 {len(faces)} → {len(fine_faces)}')
    for count in COUNTS:
        same = chamfer(vertices, faces, vertices, faces, count, SEED, SEED)
        divided = chamfer(vertices, faces, fine_vertices, fine_faces, count, SEED, SEED)
        print(f'   点数 {count:>5}：原网格对自己 d={same:.4f}   原网格对细分网格 d={divided:.4f}')

    print('\n3. 噪声底：同一个形状，两次采样用不同种子（模拟三角形划分不同）')
    print(f'   {"物体":<24}' + ''.join(f'{f"N={c}":>10}' for c in COUNTS))
    for name in NAMES:
        mesh = load(name)
        floors = [np.mean([chamfer(*mesh, *mesh, count, SEED, SEED + k) for k in range(1, 4)])
                  for count in COUNTS]
        print(f'   {name:<24}' + ''.join(f'{d:>10.4f}' for d in floors))

    print('\n4. 噪声底的均值和波动：书桌，20 个不同种子')
    for count in COUNTS[:3]:
        floors = [chamfer(vertices, faces, vertices, faces, count, SEED, SEED + k)
                  for k in range(1, 21)]
        print(f'   点数 {count:>5}：均值 {np.mean(floors):.5f}   标准差 {np.std(floors):.5f}')

    print('\n5. 作为对照：真实的形状差异有多大（N=4096）')
    shift = vertices + np.array([0.0, 0.0, 0.01 * np.linalg.norm(np.ptp(vertices, axis=0))])
    scaled = vertices * 1.05
    for label, moved in (('整体上移 1% 对角线', shift), ('整体放大 5%', scaled)):
        exact = chamfer(vertices, faces, moved, faces, 4096, SEED, SEED)
        noisy = chamfer(vertices, faces, moved, faces, 4096, SEED, SEED + 1)
        print(f'   {label:<16} 同一组采样点 d={exact:.4f}   不同种子 d={noisy:.4f}'
              f'  奖励 exp(-d)={np.exp(-noisy):.4f}')


if __name__ == '__main__':
    main()
