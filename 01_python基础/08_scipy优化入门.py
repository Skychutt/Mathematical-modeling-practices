"""
数学建模 · Python 基础 08：SciPy 优化入门
连续优化可先试解析/数值方法；难问题再上启发式。
安装：pip install scipy
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution


def sphere(x: np.ndarray) -> float:
    """经典测试函数：min 在原点，最优值 0。"""
    return float(np.sum(x ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock：香蕉谷，较难。"""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def demo_local_minimize():
    x0 = np.array([1.2, -0.8, 0.5])
    res = minimize(sphere, x0, method="BFGS")
    print("BFGS 求 sphere:")
    print("  成功:", res.success, "最优解:", np.round(res.x, 6), "值:", res.fun)


def demo_bounded():
    bounds = [(-2, 2), (-2, 2)]
    res = minimize(rosenbrock, x0=np.array([-1.0, 1.5]), bounds=bounds, method="L-BFGS-B")
    print("L-BFGS-B 求 Rosenbrock:")
    print("  解:", np.round(res.x, 6), "值:", res.fun)


def demo_global_de():
    # 差分进化：全局启发，适合低维连续问题
    bounds = [(-5, 5), (-5, 5)]
    res = differential_evolution(rosenbrock, bounds, seed=0, polish=True)
    print("差分进化求 Rosenbrock:")
    print("  解:", np.round(res.x, 6), "值:", res.fun)


if __name__ == "__main__":
    print("=" * 40, "局部优化")
    demo_local_minimize()
    print("=" * 40, "有界优化")
    demo_bounded()
    print("=" * 40, "全局启发(DE)")
    demo_global_de()
