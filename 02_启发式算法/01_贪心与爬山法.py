"""
启发式算法 01：贪心 (Greedy) + 爬山法 (Hill Climbing)

思路：
- 贪心：每一步选当前看起来最好的选择（快，但易局部最优）
- 爬山：在邻域里只接受更优解，直到无法改进
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _problems import (
    make_tsp_cities,
    random_tour,
    tour_length,
    tsp_distance_matrix,
    two_opt_swap,
)


def greedy_nearest_neighbor(dist: np.ndarray, start: int = 0) -> np.ndarray:
    """TSP 最近邻贪心：从 start 出发，每次去最近未访问城市。"""
    n = dist.shape[0]
    unvisited = set(range(n))
    tour = [start]
    unvisited.remove(start)
    while unvisited:
        last = tour[-1]
        nxt = min(unvisited, key=lambda j: dist[last, j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return np.array(tour, dtype=int)


def hill_climbing_tsp(
    dist: np.ndarray,
    init_tour: np.ndarray | None = None,
    max_iters: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """
    爬山法（2-opt 邻域）：只接受缩短路径的交换。
    返回：最优路径、长度、收敛曲线。
    """
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    tour = init_tour.copy() if init_tour is not None else random_tour(n, rng)
    best_len = tour_length(tour, dist)
    history = [best_len]

    no_improve = 0
    for _ in range(max_iters):
        i, k = sorted(rng.choice(n, size=2, replace=False))
        if k <= i + 1:
            continue
        cand = two_opt_swap(tour, i, k)
        cand_len = tour_length(cand, dist)
        if cand_len + 1e-12 < best_len:
            tour, best_len = cand, cand_len
            no_improve = 0
        else:
            no_improve += 1
        history.append(best_len)
        if no_improve > n * n:  # 长时间无改进则停
            break
    return tour, best_len, history


def hill_climbing_continuous(
    f,
    x0: np.ndarray,
    step: float = 0.1,
    bounds: tuple[float, float] = (-5.12, 5.12),
    max_iters: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """连续空间爬山：随机扰动，只接受更优。"""
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    fx = float(f(x))
    history = [fx]
    lo, hi = bounds

    for _ in range(max_iters):
        cand = x + rng.normal(0, step, size=x.shape)
        cand = np.clip(cand, lo, hi)
        fc = float(f(cand))
        if fc < fx:
            x, fx = cand, fc
        history.append(fx)
    return x, fx, history


if __name__ == "__main__":
    from _problems import rastrigin

    cities = make_tsp_cities(25, seed=1)
    dist = tsp_distance_matrix(cities)

    greedy = greedy_nearest_neighbor(dist, start=0)
    g_len = tour_length(greedy, dist)
    print(f"[贪心最近邻] TSP 路径长 = {g_len:.4f}")

    improved, hc_len, _ = hill_climbing_tsp(dist, init_tour=greedy, seed=1)
    print(f"[爬山 2-opt] 改进后路径长 = {hc_len:.4f}")

    x_best, f_best, _ = hill_climbing_continuous(
        rastrigin, x0=np.array([3.0, -2.5, 1.0]), step=0.3, seed=0
    )
    print(f"[连续爬山] Rastrigin 最优约 = {f_best:.6f}, x={np.round(x_best, 4)}")
