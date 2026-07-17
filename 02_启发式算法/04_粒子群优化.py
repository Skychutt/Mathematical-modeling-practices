"""
启发式算法 04：粒子群优化 (Particle Swarm Optimization, PSO)

每个粒子有位置与速度，向「个体最优 + 全局最优」学习。
适合连续优化；离散问题需改编码。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _problems import rastrigin, sphere


def pso(
    f,
    dim: int = 2,
    bounds: tuple[float, float] = (-5.12, 5.12),
    n_particles: int = 40,
    iters: int = 150,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, float, list[float]]:
    """
    标准 PSO（最小化）。
    w: 惯性权重；c1: 认知系数；c2: 社会系数。
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    span = hi - lo

    pos = rng.uniform(lo, hi, size=(n_particles, dim))
    vel = rng.uniform(-0.1 * span, 0.1 * span, size=(n_particles, dim))
    fit = np.array([f(p) for p in pos])

    pbest = pos.copy()
    pbest_f = fit.copy()
    g_idx = int(np.argmin(pbest_f))
    gbest = pbest[g_idx].copy()
    gbest_f = float(pbest_f[g_idx])
    history = [gbest_f]

    for _ in range(iters):
        r1 = rng.random((n_particles, dim))
        r2 = rng.random((n_particles, dim))
        vel = (
            w * vel
            + c1 * r1 * (pbest - pos)
            + c2 * r2 * (gbest - pos)
        )
        # 限制速度，避免飞出太远
        vmax = 0.2 * span
        vel = np.clip(vel, -vmax, vmax)
        pos = np.clip(pos + vel, lo, hi)

        fit = np.array([f(p) for p in pos])
        improved = fit < pbest_f
        pbest[improved] = pos[improved]
        pbest_f[improved] = fit[improved]

        g_idx = int(np.argmin(pbest_f))
        if pbest_f[g_idx] < gbest_f:
            gbest = pbest[g_idx].copy()
            gbest_f = float(pbest_f[g_idx])
        history.append(gbest_f)

    return gbest, gbest_f, history


if __name__ == "__main__":
    x1, f1, _ = pso(sphere, dim=5, iters=100, seed=0)
    print(f"[PSO] Sphere ≈ {f1:.6e}, x={np.round(x1, 4)}")

    x2, f2, hist = pso(rastrigin, dim=3, iters=200, seed=1)
    print(f"[PSO] Rastrigin ≈ {f2:.6f}, x={np.round(x2, 4)}")
    print(f"      收敛: 起始 {hist[0]:.3f} -> 结束 {hist[-1]:.3f}")
