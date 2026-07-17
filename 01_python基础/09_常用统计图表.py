"""
数学建模 · 可视化 09：常用统计图表
柱状图、饼图、箱线图、误差棒、双轴、堆叠面积 —— 论文结果对比常用。
运行后图片保存在同目录 _demo_data/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "_demo_data"
OUT.mkdir(exist_ok=True)


def plot_bar_compare():
    """多算法性能对比柱状图。"""
    methods = ["贪心", "爬山", "SA", "GA", "PSO"]
    scores = [420, 390, 355, 348, 360]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(methods, scores, color=colors, edgecolor="white", width=0.65)
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set_ylabel("路径长度（越小越好）")
    ax.set_title("TSP 不同算法结果对比")
    ax.set_ylim(0, max(scores) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bar_compare.png", dpi=140)
    plt.close(fig)


def plot_grouped_bar():
    """分组柱状：多指标对比。"""
    labels = ["算例1", "算例2", "算例3", "算例4"]
    ga = [12.1, 15.3, 9.8, 14.0]
    pso = [11.8, 14.9, 10.2, 13.5]
    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, ga, w, label="GA", color="#4C72B0")
    ax.bar(x + w / 2, pso, w, label="PSO", color="#55A868")
    ax.set_xticks(x, labels)
    ax.set_ylabel("目标函数值")
    ax.set_title("多算例分组对比")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "bar_grouped.png", dpi=140)
    plt.close(fig)


def plot_pie():
    """饼图：结构占比。"""
    labels = ["运输", "仓储", "人工", "能源", "其他"]
    sizes = [35, 25, 20, 12, 8]
    explode = (0.04, 0, 0, 0, 0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
    )
    ax.set_title("成本结构占比")
    fig.tight_layout()
    fig.savefig(OUT / "pie.png", dpi=140)
    plt.close(fig)


def plot_boxplot():
    """箱线图：多次实验分布（稳定性）。"""
    rng = np.random.default_rng(0)
    data = [
        rng.normal(350, 12, 40),
        rng.normal(340, 18, 40),
        rng.normal(330, 10, 40),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot(data, tick_labels=["SA", "GA", "PSO"], patch_artist=True)
    colors = ["#C44E52", "#8172B2", "#55A868"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_ylabel("目标值")
    ax.set_title("30 次独立实验分布")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "boxplot.png", dpi=140)
    plt.close(fig)


def plot_errorbar():
    """误差棒：均值 ± 标准差。"""
    x = np.arange(1, 6)
    mean = np.array([10.2, 9.1, 8.4, 7.9, 7.6])
    std = np.array([0.8, 0.6, 0.7, 0.5, 0.4])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, mean, yerr=std, fmt="-o", capsize=4, color="#4C72B0", label="均值±标准差")
    ax.fill_between(x, mean - std, mean + std, alpha=0.15, color="#4C72B0")
    ax.set_xlabel("迭代代数（×50）")
    ax.set_ylabel("最优值")
    ax.set_title("收敛过程（带误差带）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "errorbar.png", dpi=140)
    plt.close(fig)


def plot_twin_axis():
    """双纵轴：两个量纲不同的指标。"""
    t = np.arange(1, 11)
    cost = 100 * np.exp(-0.15 * t) + 20
    service = 60 + 35 * (1 - np.exp(-0.25 * t))

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()
    l1 = ax1.plot(t, cost, "o-", color="#C44E52", label="成本")
    l2 = ax2.plot(t, service, "s--", color="#4C72B0", label="服务水平(%)")
    ax1.set_xlabel("方案编号")
    ax1.set_ylabel("成本", color="#C44E52")
    ax2.set_ylabel("服务水平 (%)", color="#4C72B0")
    ax1.set_title("成本-服务双目标趋势")
    lines = l1 + l2
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="center right")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "twin_axis.png", dpi=140)
    plt.close(fig)


def plot_stacked_area():
    """堆叠面积：构成随时间变化。"""
    t = np.arange(0, 12)
    a = 20 + 2 * np.sin(t / 2)
    b = 15 + 0.8 * t
    c = 30 - 0.5 * t
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.stackplot(t, a, b, c, labels=["类别A", "类别B", "类别C"], alpha=0.85)
    ax.set_xlabel("月份")
    ax.set_ylabel("数量")
    ax.set_title("构成随时间变化")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "stacked_area.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    plot_bar_compare()
    plot_grouped_bar()
    plot_pie()
    plot_boxplot()
    plot_errorbar()
    plot_twin_axis()
    plot_stacked_area()
    print(f"已生成 7 张图 -> {OUT}")
