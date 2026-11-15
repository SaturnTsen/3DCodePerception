"""第 5 步：参考图是怎么来的，丢掉了哪些信息。

pixi run python docs/learn/step5_renders.py

真实尺寸读自第一次审查时导出到 temp/review/ 的目标 NPZ（Blender 坐标，z 向上）。
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('data/raw/3DCodeDataFactory-Cleaned')
VIEWS = ('Image_005.png', 'Image_015.png', 'Image_025.png', 'Image_035.png')
NAMES = ('CupFactory', 'MushroomFactory', 'SimpleDeskFactory', 'LiteDoorFactory',
         'PillarFactory', 'SpiralStaircaseFactory')
GRID_NAMES = ('SimpleDeskFactory', 'CupFactory', 'PillarFactory', 'SpiralStaircaseFactory')
CELL = 256


def pixel_extent(path: Path) -> tuple[int, int]:
    """不透明像素的包围框：物体在画面里占多宽、多高（像素）。"""
    alpha = np.asarray(Image.open(path))[:, :, 3] > 0
    rows, columns = np.nonzero(alpha)
    return int(np.ptp(columns)) + 1, int(np.ptp(rows)) + 1


def on_white(path: Path) -> Image.Image:
    image = Image.open(path).convert('RGBA').resize((CELL, CELL))
    background = Image.new('RGBA', image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    return background.convert('RGB')


def main() -> None:
    print(f'{"物体":<24}{"真实高度(m)":>12}{"图中高度(px)":>14}{"每米像素":>10}   四张图的宽度(px)')
    for name in NAMES:
        height = np.ptp(np.load(f'temp/review/{name}.npz')['vertices'][:, 2])
        extents = [pixel_extent(ROOT / name / 'color_renders' / view) for view in VIEWS]
        pixels = max(h for _, h in extents)
        widths = [w for w, _ in extents]
        print(f'{name:<24}{height:>12.3f}{pixels:>14}{pixels / height:>10.0f}   {widths}')

    grid = Image.new('RGB', (CELL * len(VIEWS), CELL * len(GRID_NAMES)), 'white')
    for row, name in enumerate(GRID_NAMES):
        for column, view in enumerate(VIEWS):
            grid.paste(on_white(ROOT / name / 'color_renders' / view), (column * CELL, row * CELL))
    grid.save('docs/learn/step5_views.png')
    print('\n四个物体的参考图已拼成 docs/learn/step5_views.png')


if __name__ == '__main__':
    main()
