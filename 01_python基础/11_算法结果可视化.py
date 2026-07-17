"""
数学建模 · 可视化 11：启发式算法结果图
收敛曲线、TSP 路径、优化轨迹、Pareto 前沿 —— 论文/答辩可直接改用。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ALG_DIR = Path(__file__).resolve().parents[1] / "02_启发式算法"
sys.path.insert(0, str(ALG_DIR))

from _problems import make_tsp_cities, rastrigin, tsp_distance_matrix  # noqa: E402

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "_demo_data"
OUT.mkdir(exist_ok=True)


def _load_sa():
    path = ALG_DIR / "02_模拟退火.py"
    spec = importlib.util.spec_from_file_location("sa_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def plot_convergence_curves():
    """多算法收敛曲线对比。"""
    sa = _load_sa()
    _, _, hist_sa = sa.simulated_annealing_continuous(
        rastrigin, dim=2, max_iters=3000, seed=0
    )
    rng = np.random.default_rng(1)
    t = np.arange(len(hist_sa))
    hist_ga = hist_sa[0] * np.exp(-t / 800) + 0.5 + rng.normal(0, 0.05, len(t))
    hist_ga = np.minimum.accumulate(np.maximum(hist_ga, 0))
    hist_pso = hist_sa[0] * np.exp(-t / 600) + 0.2 + rng.normal(0, 0.04, len(t))
    hist_pso = np.minimum.accumulate(np.maximum(hist_pso, 0))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist_sa, label="SA", linewidth=1.5)
    ax.plot(hist_ga, label="GA(示意)", linewidth=1.5)
    ax.plot(hist_pso, label="PSO(示意)", linewidth=1.5)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("历史最优值")
    ax.set_title("收敛曲线对比")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png", dpi=140)
    plt.close(fig)


def plot_tsp_tour():
    """画出 TSP 城市与最优路径。"""
    sa = _load_sa()
    cities = make_tsp_cities(25, seed=7)
    dist = tsp_distance_matrix(cities)
    tour, length, hist = sa.simulated_annealing_tsp(dist, max_iters=8000, seed=7)
    path = cities[np.append(tour, tour[0])]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(cities[:, 0], cities[:, 1], c="#4C72B0", s=40, zorder=3)
    for i, (x, y) in enumerate(cities):
        axes[0].annotate(str(i), (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
    axes[0].plot(path[:, 0], path[:, 1], "-o", color="#C44E52", markersize=4, linewidth=1.2)
    axes[0].set_title(f"TSP 路径 (长={length:.2f})")
    axes[0].set_aspect("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist, color="#8172B2")
    axes[1].set_xlabel("迭代")
    axes[1].set_ylabel("最优路径长")
    axes[1].set_title("SA 求解 TSP 收敛过程")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "tsp_tour.png", dpi=140)
    plt.close(fig)


def plot_opt_trajectory():
    """在等高线上画优化轨迹。"""
    rng = np.random.default_rng(3)
    lo, hi = -5.12, 5.12
    cur = rng.uniform(lo, hi, size=2)
    cur_f = float(rastrigin(cur))
    best, best_f = cur.copy(), cur_f
    traj = [best.copy()]
    t, t_min, alpha, step = 5.0, 1e-3, 0.97, 0.6
    for _ in range(400):
        cand = np.clip(cur + rng.normal(0, step, size=2), lo, hi)
        cand_f = float(rastrigin(cand))
        delta = cand_f - cur_f
        if delta < 0 or rng.random() < np.exp(-delta / max(t, 1e-12)):
            cur, cur_f = cand, cand_f
            if cur_f < best_f:
                best, best_f = cur.copy(), cur_f
                traj.append(best.copy())
        t = max(t * alpha, t_min)
    traj = np.array(traj)

    xs = np.linspace(lo, hi, 250)
    ys = np.linspace(lo, hi, 250)
    X, Y = np.meshgrid(xs, ys)
    Z = 20 + X**2 + Y**2 - 10 * np.cos(2 * np.pi * X) - 10 * np.cos(2 * np.pi * Y)

    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(X, Y, Z, levels=30, cmap="viridis", alpha=0.9)
    fig.colorbar(cs, ax=ax, label="Rastrigin")
    ax.plot(traj[:, 0], traj[:, 1], "w.-", linewidth=1.2, markersize=4, label="历史最优轨迹")
    ax.scatter([0], [0], c="red", marker="*", s=160, zorder=5, label="理论最优")
    ax.scatter(traj[0, 0], traj[0, 1], c="cyan", s=50, zorder=5, label="起点附近")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"SA 搜索轨迹 (best≈{best_f:.3f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "opt_trajectory.png", dpi=140)
    plt.close(fig)


def plot_pareto_front():
    """双目标 Pareto 前沿示意。"""
    rng = np.random.default_rng(0)
    f1 = np.linspace(0.2, 5, 40)
    f2 = 6 / f1 + rng.normal(0, 0.08, size=f1.shape)
    dominated_f1 = rng.uniform(1, 5, 60)
    dominated_f2 = rng.uniform(1, 6, 60)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(dominated_f1, dominated_f2, c="lightgray", s=25, label="被支配解")
    ax.plot(f1, f2, "o-", color="#C44E52", markersize=4, label="Pareto 前沿")
    ax.set_xlabel("目标 f1（成本）")
    ax.set_ylabel("目标 f2（时间）")
    ax.set_title("双目标 Pareto 前沿")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "pareto.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    plot_convergence_curves()
    plot_tsp_tour()
    plot_opt_trajectory()
    plot_pareto_front()
    print(f"已生成算法可视化图 -> {OUT}")
