"""
启发式算法 06：蚁群优化 (Ant Colony Optimization, ACO) — TSP

信息素启发：蚂蚁按概率选下一城市，走完路径后按质量释放信息素。
适合路径类组合优化。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _problems import make_tsp_cities, tour_length, tsp_distance_matrix


def aco_tsp(
    dist: np.ndarray,
    n_ants: int = 30,
    iters: int = 100,
    alpha: float = 1.0,
    beta: float = 5.0,
    rho: float = 0.5,
    q: float = 100.0,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """
    蚁群算法解 TSP。
    alpha: 信息素重要程度；beta: 启发信息(1/距离)重要程度；rho: 挥发系数。
    """
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    # 启发矩阵：距离越小越好；对角置很小避免除零
    eta = 1.0 / (dist + np.eye(n) * 1e9)
    tau = np.ones((n, n))
    best_tour = None
    best_len = float("inf")
    history: list[float] = []

    for _ in range(iters):
        tours = []
        lengths = []
        for _a in range(n_ants):
            start = int(rng.integers(0, n))
            visited = [start]
            unvisited = set(range(n)) - {start}
            while unvisited:
                i = visited[-1]
                candidates = list(unvisited)
                weights = []
                for j in candidates:
                    weights.append((tau[i, j] ** alpha) * (eta[i, j] ** beta))
                weights = np.asarray(weights, dtype=float)
                if weights.sum() <= 0:
                    probs = np.ones(len(candidates)) / len(candidates)
                else:
                    probs = weights / weights.sum()
                j = int(rng.choice(candidates, p=probs))
                visited.append(j)
                unvisited.remove(j)
            tour = np.array(visited, dtype=int)
            length = tour_length(tour, dist)
            tours.append(tour)
            lengths.append(length)
            if length < best_len:
                best_len = length
                best_tour = tour.copy()

        # 挥发 + 释放
        tau *= 1.0 - rho
        for tour, length in zip(tours, lengths):
            deposit = q / length
            for a, b in zip(tour, np.roll(tour, -1)):
                tau[a, b] += deposit
                tau[b, a] += deposit
        history.append(best_len)

    assert best_tour is not None
    return best_tour, best_len, history


if __name__ == "__main__":
    cities = make_tsp_cities(25, seed=5)
    dist = tsp_distance_matrix(cities)
    tour, length, hist = aco_tsp(dist, iters=80, seed=5)
    print(f"[蚁群-TSP] 路径长 ≈ {length:.4f}")
    print(f"收敛: {hist[0]:.2f} -> {hist[-1]:.2f}")
