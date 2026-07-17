"""
数学建模 · 13：非线性规划 (Nonlinear Programming, NLP)

一般形式：
    min  f(x)
    s.t. g_i(x) <= 0     （不等式约束，可为非线性）
         h_j(x) == 0     （等式约束，可为非线性）
         bounds

与线性规划的区别：目标或约束里出现了非线性项（平方、乘积、指数等）。
可行域不一定是凸多边形；可能有多个局部最优 —— 数值法对初值敏感。

常用方法（SciPy）：
  - SLSQP：序列二次规划，适合带约束的光滑问题
  - trust-constr：信赖域约束法
  - 无约束时可用 BFGS / L-BFGS-B
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "_demo_data"
OUT.mkdir(exist_ok=True)


def demo_unconstrained():
    """无约束 NLP：min (x-1)^2 + (y-2)^2 + 0.1*sin(3x)*sin(3y) 的光滑近似。"""

    def f(v):
        x, y = v
        return (x - 1) ** 2 + (y - 2) ** 2

    res = minimize(f, x0=np.array([-2.0, 5.0]), method="BFGS")
    print("=" * 40, "无约束 NLP")
    print("成功:", res.success, "解:", np.round(res.x, 6), "f*=", res.fun)


def demo_with_inequality():
    """
    带不等式约束：
      min  (x-2)^2 + (y-1)^2
      s.t. x^2 + y^2 <= 1      （落在单位圆内）
           x + y >= 0           （写成 -x-y <= 0）
    直觉：无约束最优在 (2,1) 圆外，约束最优应在圆边界上靠近 (2,1) 的点。
    """

    def f(v):
        x, y = v
        return (x - 2) ** 2 + (y - 1) ** 2

    cons = [
        {"type": "ineq", "fun": lambda v: 1 - v[0] ** 2 - v[1] ** 2},  # >=0
        {"type": "ineq", "fun": lambda v: v[0] + v[1]},                 # >=0
    ]
    res = minimize(f, x0=np.array([0.1, 0.1]), method="SLSQP", constraints=cons)
    print("=" * 40, "不等式约束 NLP")
    print("成功:", res.success, res.message)
    print("解:", np.round(res.x, 6), "f*=", round(res.fun, 6))
    print("圆约束剩余:", round(1 - res.x[0] ** 2 - res.x[1] ** 2, 6))
    return res


def demo_with_equality():
    """
    等式约束：在圆 x^2+y^2=1 上最小化 (x-1)^2 + (y-1)^2
    （几何：到点(1,1)最近的单位圆周上的点）
    """

    def f(v):
        x, y = v
        return (x - 1) ** 2 + (y - 1) ** 2

    cons = [{"type": "eq", "fun": lambda v: v[0] ** 2 + v[1] ** 2 - 1}]
    res = minimize(f, x0=np.array([0.7, 0.2]), method="SLSQP", constraints=cons)
    print("=" * 40, "等式约束 NLP")
    print("成功:", res.success)
    print("解:", np.round(res.x, 6), "f*=", round(res.fun, 6))
    print("是否在圆上:", round(res.x[0] ** 2 + res.x[1] ** 2, 6))
    return res


def demo_bounded_nlp():
    """
    工程里常见：变量有上下界 + 非线性目标
      min  x1^2 + x2^2 + x1*x2
      s.t. -1 <= xi <= 2
    """

    def f(v):
        return float(v[0] ** 2 + v[1] ** 2 + v[0] * v[1])

    res = minimize(f, x0=np.array([1.5, 1.5]), method="L-BFGS-B", bounds=[(-1, 2), (-1, 2)])
    print("=" * 40, "有界 NLP")
    print("解:", np.round(res.x, 6), "f*=", round(res.fun, 6))


def plot_nlp_circle_constraint():
    """可视化：圆约束下的最优点。"""
    res = demo_with_inequality()

    theta = np.linspace(0, 2 * np.pi, 300)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.fill(np.cos(theta), np.sin(theta), alpha=0.25, color="#4C72B0", label="可行域 x^2+y^2<=1")
    ax.plot(np.cos(theta), np.sin(theta), color="#4C72B0")
    # 目标等值线
    xs = np.linspace(-1.5, 2.5, 200)
    ys = np.linspace(-1.5, 2.5, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = (X - 2) ** 2 + (Y - 1) ** 2
    ax.contour(X, Y, Z, levels=12, colors="gray", alpha=0.5, linewidths=0.8)
    ax.scatter([2], [1], c="green", marker="x", s=80, label="无约束最优点(2,1)")
    ax.scatter([res.x[0]], [res.x[1]], c="red", s=100, marker="*", label="约束最优")
    ax.axline((0, 0), slope=-1, color="#C44E52", linestyle="--", alpha=0.7, label="x+y=0")
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 2.6)
    ax.set_ylim(-1.6, 2.6)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("非线性规划：圆约束")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "nlp_circle.png", dpi=140)
    plt.close(fig)
    print(f"NLP 图已保存: {OUT / 'nlp_circle.png'}")


if __name__ == "__main__":
    demo_unconstrained()
    demo_with_equality()
    demo_bounded_nlp()
    plot_nlp_circle_constraint()
