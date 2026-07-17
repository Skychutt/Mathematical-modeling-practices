"""
数学建模 · Python 基础 07：Matplotlib 绘图
论文/答辩图：折线、散点、等高线、多子图。
安装：pip install matplotlib
若弹窗不方便，可改用：plt.savefig("fig.png")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Windows 下尽量显示中文（若系统无该字体则忽略）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "_demo_data"
OUT.mkdir(exist_ok=True)


def plot_curve():
    x = np.linspace(0, 2 * np.pi, 200)
    y = np.sin(x)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label="sin(x)")
    plt.plot(x, np.cos(x), "--", label="cos(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("三角函数")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "curve.png", dpi=120)
    plt.close()


def plot_scatter_and_hist():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(100, 2))
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].scatter(pts[:, 0], pts[:, 1], s=20, alpha=0.7)
    axes[0].set_title("散点")
    axes[1].hist(pts[:, 0], bins=15, edgecolor="white")
    axes[1].set_title("直方图")
    fig.tight_layout()
    fig.savefig(OUT / "scatter_hist.png", dpi=120)
    plt.close(fig)


def plot_contour():
    """目标函数等高线：观察优化问题景观。"""
    xs = np.linspace(-3, 3, 200)
    ys = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = (X - 1) ** 2 + (Y + 2) ** 2

    plt.figure(figsize=(5, 4))
    cs = plt.contourf(X, Y, Z, levels=20, cmap="viridis")
    plt.colorbar(cs, label="f(x,y)")
    plt.scatter([1], [-2], c="red", marker="*", s=120, label="最优")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("目标函数等高线")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "contour.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    plot_curve()
    plot_scatter_and_hist()
    plot_contour()
    print(f"图片已保存到: {OUT}")
