"""
数学建模 · 可视化 10：热力图 / 相关矩阵 / 三维曲面
适合：距离矩阵、相关系数、目标函数景观。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 注册 3D

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "_demo_data"
OUT.mkdir(exist_ok=True)


def plot_heatmap_matrix():
    """热力图：距离/相似度矩阵。"""
    rng = np.random.default_rng(1)
    n = 8
    pts = rng.random((n, 2))
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist, cmap="YlOrRd")
    ax.set_xticks(range(n), [f"P{i}" for i in range(n)])
    ax.set_yticks(range(n), [f"P{i}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{dist[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="距离")
    ax.set_title("点间距离矩阵热力图")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_dist.png", dpi=140)
    plt.close(fig)


def plot_corr_heatmap():
    """相关系数热力图。"""
    rng = np.random.default_rng(0)
    # 构造有相关结构的数据
    x1 = rng.normal(size=200)
    x2 = 0.8 * x1 + 0.2 * rng.normal(size=200)
    x3 = -0.5 * x1 + 0.5 * rng.normal(size=200)
    x4 = rng.normal(size=200)
    data = np.column_stack([x1, x2, x3, x4])
    names = ["x1", "x2", "x3", "x4"]
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4), names)
    ax.set_yticks(range(4), names)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="相关系数")
    ax.set_title("变量相关矩阵")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_corr.png", dpi=140)
    plt.close(fig)


def plot_surface_3d():
    """三维曲面：目标函数景观。"""
    xs = np.linspace(-2, 2, 80)
    ys = np.linspace(-1, 3, 80)
    X, Y = np.meshgrid(xs, ys)
    # 简化 Rosenbrock 景观
    Z = (1 - X) ** 2 + 100 * (Y - X**2) ** 2

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, np.log1p(Z), cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("log(1+f)")
    ax.set_title("Rosenbrock 函数景观（对数高度）")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="log(1+f)")
    fig.tight_layout()
    fig.savefig(OUT / "surface_3d.png", dpi=140)
    plt.close(fig)


def plot_scatter3d():
    """三维散点：多指标样本分布。"""
    rng = np.random.default_rng(2)
    n = 80
    cost = rng.uniform(20, 80, n)
    time_ = rng.uniform(5, 30, n)
    quality = 100 - 0.4 * cost - 0.8 * time_ + rng.normal(0, 3, n)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(cost, time_, quality, c=quality, cmap="plasma", s=35, alpha=0.85)
    ax.set_xlabel("成本")
    ax.set_ylabel("时间")
    ax.set_zlabel("质量")
    ax.set_title("方案三维散点")
    fig.colorbar(p, ax=ax, shrink=0.6, label="质量")
    fig.tight_layout()
    fig.savefig(OUT / "scatter3d.png", dpi=140)
    plt.close(fig)


def plot_quiver():
    """向量场/梯度示意（力学、流场、梯度下降方向）。"""
    x = np.linspace(-2, 2, 20)
    y = np.linspace(-2, 2, 20)
    X, Y = np.meshgrid(x, y)
    # f = x^2 + y^2 的负梯度方向（下降方向）
    U = -2 * X
    V = -2 * Y

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.quiver(X, Y, U, V, color="#4C72B0", alpha=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("负梯度方向场（指向极小）")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "quiver.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    plot_heatmap_matrix()
    plot_corr_heatmap()
    plot_surface_3d()
    plot_scatter3d()
    plot_quiver()
    print(f"已生成 5 张图 -> {OUT}")
