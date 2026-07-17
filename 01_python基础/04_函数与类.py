"""
数学建模 · Python 基础 04：函数、lambda、简单类
把「目标函数 / 约束 / 邻域操作」封装成函数，是算法代码的基本习惯。
"""

from typing import Callable, List


def objective(x: float, y: float) -> float:
    """示例目标函数：越小越好。"""
    return (x - 1) ** 2 + (y + 2) ** 2


def clip(x: float, lo: float, hi: float) -> float:
    """把变量裁剪到边界内（约束处理常用）。"""
    return max(lo, min(hi, x))


def evaluate_population(pop: List[List[float]], fitness: Callable) -> List[float]:
    """对一群解统一求适应度。"""
    return [fitness(ind) for ind in pop]


def demo_function():
    print("f(1,-2) =", objective(1, -2))
    print("clip(12, 0, 10) =", clip(12, 0, 10))

    # 默认参数、关键字参数
    def anneal_step(t, alpha=0.95, t_min=1e-3):
        return max(t * alpha, t_min)

    print("降温:", anneal_step(100), anneal_step(100, alpha=0.9))


def demo_lambda_and_sort():
    # lambda：短小的匿名函数，适合当 key
    points = [(1, 5), (0, 2), (3, 1)]
    by_y = sorted(points, key=lambda p: p[1])
    print("按 y 排序:", by_y)

    fitness = lambda ind: sum(v ** 2 for v in ind)
    pop = [[1, 2], [0, 1], [3, 0]]
    print("适应度:", evaluate_population(pop, fitness))


class Particle:
    """粒子群里一个粒子的极简表示。"""

    def __init__(self, position, velocity):
        self.position = list(position)
        self.velocity = list(velocity)
        self.best_position = list(position)
        self.best_value = float("inf")

    def update_best(self, value: float):
        if value < self.best_value:
            self.best_value = value
            self.best_position = list(self.position)

    def __repr__(self):
        return f"Particle(pos={self.position}, best={self.best_value:.4f})"


def demo_class():
    p = Particle([0.5, -0.2], [0.1, 0.0])
    p.update_best(objective(*p.position))
    print(p)


if __name__ == "__main__":
    print("=" * 40, "函数")
    demo_function()
    print("=" * 40, "lambda")
    demo_lambda_and_sort()
    print("=" * 40, "类")
    demo_class()
