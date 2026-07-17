"""
启发式算法共用测试问题：连续函数 + 旅行商问题(TSP)。
"""

from __future__ import annotations

import numpy as np


# ---------- 连续优化测试函数 ----------

def sphere(x: np.ndarray) -> float:
    """Sphere: 全局最优 x*=0, f*=0。"""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin: 多峰，最优 x*=0, f*=0。"""
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


# ---------- TSP ----------

def make_tsp_cities(n: int = 20, seed: int = 0) -> np.ndarray:
    """随机生成平面城市坐标，形状 (n, 2)。"""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * 100.0


def tsp_distance_matrix(cities: np.ndarray) -> np.ndarray:
    diff = cities[:, None, :] - cities[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def tour_length(tour: np.ndarray, dist: np.ndarray) -> float:
    """闭环路径总长。tour 为城市下标排列。"""
    tour = np.asarray(tour, dtype=int)
    return float(dist[tour, np.roll(tour, -1)].sum())


def random_tour(n: int, rng: np.random.Generator) -> np.ndarray:
    tour = np.arange(n)
    rng.shuffle(tour)
    return tour


def two_opt_swap(tour: np.ndarray, i: int, k: int) -> np.ndarray:
    """2-opt：反转 tour[i:k+1]。"""
    new_tour = tour.copy()
    new_tour[i : k + 1] = new_tour[i : k + 1][::-1]
    return new_tour
