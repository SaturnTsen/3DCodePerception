"""第 1 步：网格是什么。

运行：pixi run python docs/learn/step1_mesh.py docs/learn/SimpleDeskFactory.npz
"""

import sys

import numpy as np

# 1. 真实数据：书桌网格
while len(sys.argv) < 2:
    sys.argv.append('')
sys.argv[1] = 'docs/learn/SimpleDeskFactory.npz'


with np.load(sys.argv[1]) as mesh:
    vertices, faces = mesh['vertices'], mesh['faces']
    print(mesh['vertices'].dtype, mesh['faces'].dtype)
print('vertices:', vertices.shape, vertices.dtype)
print('faces:   ', faces.shape, faces.dtype)
print('前 3 个顶点（x, y, z，单位米）:\n', np.round(vertices[:3], 4))
print('前 3 个三角形（每行是 3 个顶点下标）:\n', faces[:3])
print('第 0 个三角形的 3 个顶点坐标:\n', np.round(vertices[faces[0]], 4))
print('包围盒 min:', np.round(vertices.min(0), 3), 'max:', np.round(vertices.max(0), 3))

# 2. 同一个 1x1 正方形，两种三角划分
coarse_v = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
coarse_f = np.array([[0, 1, 2], [0, 2, 3]])
fine_v = np.array([[x, y, 0] for y in (0, .5, 1) for x in (0, .5, 1)], dtype=float)
fine_f = np.array([[r*3+c, r*3+c+1, r*3+c+4] for r in range(2) for c in range(2)]
                  + [[r*3+c, r*3+c+4, r*3+c+3] for r in range(2) for c in range(2)])


def area(v: np.ndarray, f: np.ndarray) -> float:
    t = v[f]
    return float(np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1).sum() / 2)


print('\n同一个正方形:')
print(f'  粗划分: {len(coarse_v)} 个顶点, {len(coarse_f)} 个三角形, 面积 {area(coarse_v, coarse_f)}')
print(f'  细划分: {len(fine_v)} 个顶点, {len(fine_f)} 个三角形, 面积 {area(fine_v, fine_f)}')
