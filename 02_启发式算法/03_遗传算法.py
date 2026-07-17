"""
启发式算法 03：遗传算法 (Genetic Algorithm, GA)

流程：初始化种群 -> 选择 -> 交叉 -> 变异 -> 精英保留，循环迭代。
本文件给两套：连续编码 GA、TSP 排列编码 GA。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _problems import (
    make_tsp_cities,
    rastrigin,
    tour_length,
    tsp_distance_matrix,
)


# ---------- 连续编码 GA ----------

def ga_continuous(
    f,
    dim: int = 2,
    bounds: tuple[float, float] = (-5.12, 5.12),
    pop_size: int = 60,
    generations: int = 120,
    pc: float = 0.8,
    pm: float = 0.1,
    eta: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """实数编码遗传算法，最小化 f。"""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    pop = rng.uniform(lo, hi, size=(pop_size, dim))
    fitness = np.array([f(ind) for ind in pop])
    history: list[float] = []

    def tournament(k: int = 3) -> np.ndarray:
        idx = rng.choice(pop_size, size=k, replace=False)
        return pop[idx[np.argmin(fitness[idx])]].copy()

    for _ in range(generations):
        new_pop = []
        # 精英：保留当前最优
        elite = pop[np.argmin(fitness)].copy()
        new_pop.append(elite)

        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            c1, c2 = p1.copy(), p2.copy()
            if rng.random() < pc:
                alpha = rng.random(dim)
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = alpha * p2 + (1 - alpha) * p1
            for child in (c1, c2):
                mask = rng.random(dim) < pm
                child[mask] += rng.normal(0, eta * (hi - lo), size=mask.sum())
                child[:] = np.clip(child, lo, hi)
                new_pop.append(child)
                if len(new_pop) >= pop_size:
                    break

        pop = np.array(new_pop[:pop_size])
        fitness = np.array([f(ind) for ind in pop])
        history.append(float(fitness.min()))

    best_i = int(np.argmin(fitness))
    return pop[best_i], float(fitness[best_i]), history


# ---------- TSP 排列编码 GA ----------

def _ox_crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """顺序交叉 OX：保留片段相对次序，适合排列编码。"""
    n = len(p1)
    a, b = sorted(rng.choice(n, size=2, replace=False))
    child = np.full(n, -1, dtype=int)
    child[a : b + 1] = p1[a : b + 1]
    fill = [x for x in p2 if x not in child]
    j = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[j]
            j += 1
    return child


def _swap_mutation(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    t = tour.copy()
    i, j = rng.choice(len(t), size=2, replace=False)
    t[i], t[j] = t[j], t[i]
    return t


def ga_tsp(
    dist: np.ndarray,
    pop_size: int = 80,
    generations: int = 200,
    pc: float = 0.9,
    pm: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """排列编码遗传算法解 TSP。"""
    rng = np.random.default_rng(seed)
    n = dist.shape[0]
    pop = np.array([rng.permutation(n) for _ in range(pop_size)])
    fitness = np.array([tour_length(ind, dist) for ind in pop])
    history: list[float] = []

    def tournament(k: int = 3) -> np.ndarray:
        idx = rng.choice(pop_size, size=k, replace=False)
        return pop[idx[np.argmin(fitness[idx])]].copy()

    for _ in range(generations):
        new_pop = [pop[np.argmin(fitness)].copy()]
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            if rng.random() < pc:
                c1 = _ox_crossover(p1, p2, rng)
                c2 = _ox_crossover(p2, p1, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if rng.random() < pm:
                c1 = _swap_mutation(c1, rng)
            if rng.random() < pm:
                c2 = _swap_mutation(c2, rng)
            new_pop.extend([c1, c2])
        pop = np.array(new_pop[:pop_size])
        fitness = np.array([tour_length(ind, dist) for ind in pop])
        history.append(float(fitness.min()))

    best_i = int(np.argmin(fitness))
    return pop[best_i], float(fitness[best_i]), history


if __name__ == "__main__":
    x, fx, hist = ga_continuous(rastrigin, dim=3, generations=150, seed=0)
    print(f"[GA-连续] Rastrigin ≈ {fx:.6f}, x={np.round(x, 4)}")

    cities = make_tsp_cities(25, seed=3)
    dist = tsp_distance_matrix(cities)
    tour, length, _ = ga_tsp(dist, generations=150, seed=3)
    print(f"[GA-TSP] 路径长 ≈ {length:.4f}")
