"""
启发式算法 05：禁忌搜索 (Tabu Search)

在局部搜索基础上，把最近访问过的移动记入禁忌表，避免来回震荡。
这里用 TSP + 2-opt，禁忌对象为被反转的边对 (i,k)。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collections import deque

import numpy as np

from _problems import (
    make_tsp_cities,
    random_tour,
    tour_length,
    tsp_distance_matrix,
    two_opt_swap,
)


def tabu_search_tsp(
    dist: np.ndarray,
    max_iters: int = 800,
    tabu_tenure: int = 20,
    candidate_sample: int = 80,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """
    禁忌搜索解 TSP。
    candidate_sample: 每轮随机抽样的邻域规模（全枚举 n^2 太大时用抽样）。
    """
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    cur = random_tour(n, rng)
    cur_len = tour_length(cur, dist)
    best, best_len = cur.copy(), cur_len
    history = [best_len]

    tabu: deque[tuple[int, int]] = deque(maxlen=tabu_tenure)

    for _ in range(max_iters):
        best_cand = None
        best_cand_len = float("inf")
        best_move = None

        for _s in range(candidate_sample):
            i, k = sorted(rng.choice(n, size=2, replace=False))
            if k <= i + 1:
                continue
            move = (int(i), int(k))
            cand = two_opt_swap(cur, i, k)
            cand_len = tour_length(cand, dist)

            # 特赦准则：比历史最优还好，即使禁忌也接受
            aspirate = cand_len + 1e-12 < best_len
            if move in tabu and not aspirate:
                continue
            if cand_len < best_cand_len:
                best_cand, best_cand_len, best_move = cand, cand_len, move

        if best_cand is None:
            history.append(best_len)
            continue

        cur, cur_len = best_cand, best_cand_len
        if best_move is not None:
            tabu.append(best_move)
        if cur_len < best_len:
            best, best_len = cur.copy(), cur_len
        history.append(best_len)

    return best, best_len, history


if __name__ == "__main__":
    cities = make_tsp_cities(30, seed=4)
    dist = tsp_distance_matrix(cities)
    tour, length, hist = tabu_search_tsp(dist, seed=4)
    print(f"[禁忌搜索-TSP] 路径长 ≈ {length:.4f}")
    print(f"收敛: {hist[0]:.2f} -> {hist[-1]:.2f}")
