"""第 4 步：坐标系约定。比较同一张书桌在 glTF 文件里和在 Blender 导出结果里的坐标。

pixi run python docs/learn/step4_axes.py
"""

from pathlib import Path

import numpy as np
from step3_glb import ROOT, read_accessor, read_glb

AXES = 'xyz'
# glTF（y 轴向上）→ Blender（z 轴向上）：x 不变，新 y = −旧 z，新 z = 旧 y
GLTF_TO_BLENDER = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])


def describe(name: str, vertices: np.ndarray) -> None:
    extent = np.ptp(vertices, axis=0)
    print(f'{name:<28} 尺寸 x={extent[0]:.3f} y={extent[1]:.3f} z={extent[2]:.3f}'
          f'   最长轴 {AXES[extent.argmax()]}，最短轴 {AXES[extent.argmin()]}')


def rotation_z(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main() -> None:
    gltf, binary = read_glb(ROOT / 'SimpleDeskFactory.glb')
    primitives = gltf['meshes'][0]['primitives']
    local = np.concatenate([
        read_accessor(gltf, binary, p['attributes']['POSITION']) for p in primitives
    ])
    gltf_world = local + np.array(gltf['nodes'][0]['translation'])
    blender = np.load('docs/learn/SimpleDeskFactory.npz')['vertices']

    print('1. 同一张书桌，两种坐标系下的尺寸（桌高约 0.73，桌面长 1.18、宽 0.75）')
    describe('glTF 文件（y 向上）', gltf_world)
    describe('Blender 导出 NPZ（z 向上）', blender)

    print('\n2. 坐标轴转换矩阵')
    print(GLTF_TO_BLENDER)
    print('行列式:', round(np.linalg.det(GLTF_TO_BLENDER)), '（+1 表示是旋转，不是镜像）')
    converted = gltf_world @ GLTF_TO_BLENDER.T
    print('转换后与 NPZ 的最大误差:', np.abs(converted - blender).max())
    print('不转换直接比较的最大误差:', np.abs(gltf_world - blender).max().round(3))

    print('\n3. 绕 z 轴转 90°，作用在两种坐标系下的书桌上')
    describe('Blender 坐标（z 向上）', blender @ rotation_z(90).T)
    describe('glTF 坐标（y 向上）', gltf_world @ rotation_z(90).T)


if __name__ == '__main__':
    main()
