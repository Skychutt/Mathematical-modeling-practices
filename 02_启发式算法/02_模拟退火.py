"""
启发式算法 02：模拟退火 (Simulated Annealing, SA)

核心：以概率 exp(-(新-旧)/T) 接受更差解，温度 T 逐渐降低。
优点：实现简单，组合/连续都能用；适合跳出局部最优。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _problems import (
    make_tsp_cities,
    random_tour,
    rastrigin,
    tour_length,
    tsp_distance_matrix,
    two_opt_swap,
)


def simulated_annealing_tsp(
    dist: np.ndarray,
    t0: float = 100.0,
    t_min: float = 1e-3,
    alpha: float = 0.995,
    max_iters: int = 20000,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """TSP 模拟退火，邻域用 2-opt。"""
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    cur = random_tour(n, rng)
    cur_len = tour_length(cur, dist)
    best, best_len = cur.copy(), cur_len
    history = [best_len]
    t = t0

    for _ in range(max_iters):
        i, k = sorted(rng.choice(n, size=2, replace=False))
        if k <= i + 1:
            t = max(t * alpha, t_min)
            history.append(best_len)
            continue
        cand = two_opt_swap(cur, i, k)
        cand_len = tour_length(cand, dist)
        delta = cand_len - cur_len
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            cur, cur_len = cand, cand_len
            if cur_len < best_len:
                best, best_len = cur.copy(), cur_len
        t = max(t * alpha, t_min)
        history.append(best_len)
        if t <= t_min:
            break
    return best, best_len, history


def simulated_annealing_continuous(
    f,
    dim: int = 2,
    bounds: tuple[float, float] = (-5.12, 5.12),
    t0: float = 10.0,
    t_min: float = 1e-4,
    alpha: float = 0.99,
    step: float = 0.5,
    max_iters: int = 10000,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """连续变量模拟退火。"""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    cur = rng.uniform(lo, hi, size=dim)
    cur_f = float(f(cur))
    best, best_f = cur.copy(), cur_f
    history = [best_f]
    t = t0

    for _ in range(max_iters):
        cand = cur + rng.normal(0, step, size=dim)
        cand = np.clip(cand, lo, hi)
        cand_f = float(f(cand))
        delta = cand_f - cur_f
        if delta < 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
            cur, cur_f = cand, cand_f
            if cur_f < best_f:
                best, best_f = cur.copy(), cur_f
        t = max(t * alpha, t_min)
        history.append(best_f)
        if t <= t_min:
            break
    return best, best_f, history


if __name__ == "__main__":
    cities = make_tsp_cities(30, seed=2)
    dist = tsp_distance_matrix(cities)
    tour, length, hist = simulated_annealing_tsp(dist, seed=2)
    print(f"[SA-TSP] 最优路径长 ≈ {length:.4f} (迭代记录 {len(hist)} 步)")

    x, fx, _ = simulated_annealing_continuous(rastrigin, dim=3, seed=0)
    print(f"[SA-连续] Rastrigin ≈ {fx:.6f}, x={np.round(x, 4)}")
