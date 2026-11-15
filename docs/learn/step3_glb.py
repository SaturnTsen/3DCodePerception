"""第 3 步：GLB 文件里装了什么。不依赖 3D 库，直接按 glTF 规范解析二进制文件。

pixi run python docs/learn/step3_glb.py
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path('data/raw/3DCodeDataFactory-Cleaned/SimpleDeskFactory')
COMPONENT_DTYPES = {5123: np.uint16, 5125: np.uint32, 5126: np.float32}
TYPE_WIDTHS = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3}


def read_chunks(path: Path) -> tuple[tuple[bytes, int, int], dict[bytes, bytes]]:
    """GLB = 12 字节文件头 + 若干个块；每个块 = 8 字节块头（长度、类型）+ 内容。"""
    data = path.read_bytes()
    header = struct.unpack('<4sII', data[:12])
    chunks: dict[bytes, bytes] = {}
    offset = 12
    while offset < len(data):
        size, kind = struct.unpack('<I4s', data[offset:offset + 8])
        chunks[kind] = data[offset + 8:offset + 8 + size]
        offset += 8 + size
    return header, chunks


def read_glb(path: Path) -> tuple[dict, bytes]:
    """返回 JSON 块（结构描述）和 BIN 块（原始数组字节）。"""
    _, chunks = read_chunks(path)
    return json.loads(chunks[b'JSON']), chunks[b'BIN\x00']


def read_accessor(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    """accessor 说明数组的类型和形状，bufferView 说明它在 BIN 块里的位置。"""
    accessor = gltf['accessors'][index]
    view = gltf['bufferViews'][accessor['bufferView']]
    start = view.get('byteOffset', 0) + accessor.get('byteOffset', 0)
    dtype = COMPONENT_DTYPES[accessor['componentType']]
    width = TYPE_WIDTHS[accessor['type']]
    array = np.frombuffer(binary, dtype, accessor['count'] * width, start)
    return array.reshape(-1, width)


def main() -> None:
    meshes = {}
    for name in ('SimpleDeskFactory.glb', 'SimpleDeskFactory_grey.glb'):
        print(f'\n===== {name} =====')
        (magic, version, length), chunks = read_chunks(ROOT / name)
        print(f'文件头: magic={magic!r} version={version} 总长度={length} 字节')
        for kind, content in chunks.items():
            print(f'  块 {kind!r}: {len(content)} 字节')
        gltf, binary = read_glb(ROOT / name)
        print('JSON 顶层字段:', list(gltf))
        print('场景的根节点:', gltf['scenes'][0]['nodes'])
        for node in gltf['nodes']:
            print('节点:', node)
        print('材质:', [m['name'] for m in gltf['materials']])
        print('贴图图片:', [i['name'] for i in gltf.get('images', [])])

        positions, triangles = [], []
        for k, primitive in enumerate(gltf['meshes'][0]['primitives']):
            position = read_accessor(gltf, binary, primitive['attributes']['POSITION'])
            indices = read_accessor(gltf, binary, primitive['indices']).reshape(-1, 3)
            print(f'  图元 {k}: 材质 {primitive["material"]}，'
                  f'属性 {list(primitive["attributes"])}，'
                  f'顶点 {len(position)}，三角形 {len(indices)}')
            print(f'    局部坐标范围 min {position.min(0).round(3)} max {position.max(0).round(3)}')
            triangles.append(indices + sum(len(p) for p in positions))
            positions.append(position)
        meshes[name] = (np.concatenate(positions), np.concatenate(triangles))
        vertices, faces = meshes[name]
        print(f'合计: 顶点 {len(vertices)}，三角形 {len(faces)}')

    color, grey = meshes.values()
    print('\n彩色版和灰色版的顶点完全相同:', np.array_equal(color[0], grey[0]))
    print('彩色版和灰色版的三角形完全相同:', np.array_equal(color[1], grey[1]))


if __name__ == '__main__':
    main()
