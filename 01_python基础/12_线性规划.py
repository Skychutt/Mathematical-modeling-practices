"""
数学建模 · 12：线性规划 (Linear Programming, LP)

标准形（最小化）：
    min  c^T x
    s.t. A_ub x <= b_ub
         A_eq x == b_eq
         bounds 下界 <= x <= 上界

特点：目标函数、约束都是「一次」的（线性）。可行域是凸多面体，
最优解一定在顶点上 —— 单纯形法/内点法可高效求解。

本文件用 scipy.optimize.linprog（HiGHS 求解器）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "_demo_data"
OUT.mkdir(exist_ok=True)


def demo_production():
    """
    经典生产计划：
      产品 A 利润 3，产品 B 利润 5
      原料约束：  x + 2y <= 8
      工时约束：  3x + 2y <= 12
      产量非负：  x,y >= 0
      目标：max 3x + 5y  （linprog 只做最小化，所以对目标取负）
    """
    # min -3x - 5y  <=>  max 3x + 5y
    c = [-3, -5]
    A_ub = [
        [1, 2],  # x + 2y <= 8
        [3, 2],  # 3x + 2y <= 12
    ]
    b_ub = [8, 12]
    bounds = [(0, None), (0, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    print("=" * 40, "生产计划 LP")
    print("成功:", res.success, res.message)
    print(f"最优产量: A={res.x[0]:.4f}, B={res.x[1]:.4f}")
    print(f"最大利润: {-res.fun:.4f}")
    return res


def demo_transport():
    """
    运输问题（简化）：2 个仓库 -> 2 个门店
      供应：仓1=20, 仓2=25
      需求：店1=15, 店2=30
      运费矩阵（单位）：
            店1  店2
        仓1  2    3
        仓2  4    1
      变量 x11,x12,x21,x22 >= 0
    """
    # min 2*x11 + 3*x12 + 4*x21 + 1*x22
    c = [2, 3, 4, 1]
    # 供应等式：x11+x12=20, x21+x22=25
    # 需求等式：x11+x21=15, x12+x22=30
    A_eq = [
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    b_eq = [20, 25, 15, 30]
    bounds = [(0, None)] * 4

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    x11, x12, x21, x22 = res.x
    print("=" * 40, "运输问题 LP")
    print("成功:", res.success)
    print(f"运量矩阵:\n  [[{x11:.1f}, {x12:.1f}],\n   [{x21:.1f}, {x22:.1f}]]")
    print(f"最小总运费: {res.fun:.2f}")
    return res


def plot_lp_feasible_region():
    """画出二维 LP 可行域与最优顶点（对应生产计划例子）。"""
    # 约束：y <= (8-x)/2 , y <= (12-3x)/2 , x>=0, y>=0
    xs = np.linspace(0, 5, 400)
    y1 = (8 - xs) / 2
    y2 = (12 - 3 * xs) / 2
    y_feas = np.minimum(y1, y2)
    y_feas = np.clip(y_feas, 0, None)

    # 最优在顶点；枚举顶点验证
    vertices = np.array([
        [0, 0],
        [0, 4],      # x=0 与 x+2y=8
        [2, 3],      # x+2y=8 与 3x+2y=12
        [4, 0],      # y=0 与 3x+2y=12
    ], dtype=float)
    profits = 3 * vertices[:, 0] + 5 * vertices[:, 1]
    best = vertices[np.argmax(profits)]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.fill_between(xs, 0, y_feas, where=(y_feas > 0), alpha=0.35, color="#4C72B0", label="可行域")
    ax.plot(xs, y1, "--", label="x+2y=8")
    ax.plot(xs, y2, "--", label="3x+2y=12")
    ax.scatter(vertices[:, 0], vertices[:, 1], c="k", s=40, zorder=3)
    ax.scatter([best[0]], [best[1]], c="red", s=120, marker="*", zorder=4, label=f"最优 {best}")
    # 等利润线示意
    for p in [10, 20, 30]:
        # 3x+5y=p => y=(p-3x)/5
        ax.plot(xs, (p - 3 * xs) / 5, ":", alpha=0.5)
    ax.set_xlim(-0.2, 5)
    ax.set_ylim(-0.2, 5)
    ax.set_xlabel("x (产品A)")
    ax.set_ylabel("y (产品B)")
    ax.set_title("线性规划：可行域与最优顶点")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "lp_feasible.png", dpi=140)
    plt.close(fig)
    print(f"可行域图已保存: {OUT / 'lp_feasible.png'}")


if __name__ == "__main__":
    demo_production()
    demo_transport()
    plot_lp_feasible_region()
