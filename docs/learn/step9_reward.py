"""第 9 步：从距离到奖励。模拟 GRPO 组内标准化后的优势值。

pixi run python docs/learn/step9_reward.py

距离 d 取自第 8 步"质心 + RMS 半径 + 4 个水平角"的实测结果。
"""

from collections.abc import Callable

import numpy as np

FAIL = None  # 脚本执行失败
DISTANCES = {
    '完美（噪声底）': 0.025,
    '删掉一条桌腿': 0.0475,
    '长边拉长 30%': 0.0534,
    '上下颠倒': 0.2327,
    '完全不同的物体（杯子）': 0.4273,
    '完全不同的物体（石柱）': 0.7556,
}
GROUPS: dict[str, list[float | None]] = {
    'A 全部成功，质量参差': [0.025, 0.0475, 0.2327, 0.4273],
    'B 一个失败，三个形状都不错': [FAIL, 0.025, 0.0475, 0.0534],
    'C 全部成功，只差采样噪声': [0.02500, 0.02506, 0.02494, 0.02503],
}
MAPPINGS: dict[str, Callable[[float], float]] = {
    'exp(-d)（现状）': lambda d: float(np.exp(-d)),
    '1 - d（线性）': lambda d: 1 - d,
    'exp(-d/0.1)': lambda d: float(np.exp(-d / 0.1)),
}


def advantages(rewards: list[float]) -> np.ndarray:
    """和 verl 的 GRPO 一致：减去组内均值，再除以组内（无偏）标准差。"""
    scores = np.array(rewards)
    return (scores - scores.mean()) / (scores.std(ddof=1) + 1e-6)


def main() -> None:
    print('1. 不同质量的回答对应的距离和奖励（失败的回答奖励为 0）')
    print(f'   {"回答":<22}{"d":>8}' + ''.join(f'{name:>16}' for name in MAPPINGS))
    for label, d in DISTANCES.items():
        print(f'   {label:<22}{d:>8.4f}' + ''.join(f'{f(d):>16.4f}' for f in MAPPINGS.values()))

    print('\n2. 一组 4 个回答（rollout.n = 4）经过组内标准化后的优势值')
    for group, distances in GROUPS.items():
        print(f'\n   {group}：d = {["失败" if d is None else d for d in distances]}')
        for name, mapping in MAPPINGS.items():
            rewards = [0.0 if d is None else mapping(d) for d in distances]
            print(f'   {name:<16} 奖励 {np.round(rewards, 4)}  优势 {np.round(advantages(rewards), 2)}')


if __name__ == '__main__':
    main()
