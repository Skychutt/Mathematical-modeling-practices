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


def plot_subplots_style():
    """论文常用：2×2 子图排版。"""
    x = np.linspace(0, 4, 100)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes[0, 0].plot(x, np.exp(-x), color="#4C72B0")
    axes[0, 0].set_title("衰减曲线")
    axes[0, 1].fill_between(x, np.sin(x), alpha=0.4, color="#55A868")
    axes[0, 1].plot(x, np.sin(x), color="#55A868")
    axes[0, 1].set_title("填充折线")
    axes[1, 0].step(x[::5], np.cos(x[::5]), where="mid", color="#C44E52")
    axes[1, 0].set_title("阶梯图")
    axes[1, 1].plot(x, x**2, label="x^2")
    axes[1, 1].plot(x, x**1.5, label="x^1.5")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("多曲线")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.suptitle("子图排版示例", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "subplots.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    plot_curve()
    plot_scatter_and_hist()
    plot_contour()
    plot_subplots_style()
    print(f"图片已保存到: {OUT}")
    print("更多图表见: 09_常用统计图表.py / 10_热力图与三维.py / 11_算法结果可视化.py")
