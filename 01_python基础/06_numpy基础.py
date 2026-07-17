"""
数学建模 · Python 基础 06：NumPy 核心
几乎所有数值建模都依赖 NumPy：向量、矩阵、向量化运算。
安装：pip install numpy
"""

import numpy as np


def demo_array_basics():
    a = np.array([1, 2, 3], dtype=float)
    b = np.zeros((2, 3))
    c = np.ones((3,))
    d = np.eye(3)                 # 单位矩阵
    e = np.linspace(0, 1, 5)      # 等间距
    f = np.arange(0, 10, 2)       # 类似 range
    g = np.random.rand(2, 2)      # [0,1) 均匀随机
    print("a:", a)
    print("zeros:\n", b)
    print("linspace:", e)
    print("arange:", f)
    print("rand:\n", g)
    print("shape/dtype:", a.shape, a.dtype)


def demo_ops():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    print("逐元素 +:", x + y)
    print("逐元素 *:", x * y)
    print("点积:", np.dot(x, y), "或", x @ y)

    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[0.0, 1.0], [1.0, 0.0]])
    print("矩阵乘:\n", A @ B)
    print("转置:\n", A.T)
    print("行列式:", np.linalg.det(A))
    print("逆矩阵:\n", np.linalg.inv(A))

    # 广播：标量/向量自动扩展维度
    print("x + 10:", x + 10)
    print("均值/方差:", x.mean(), x.var())
    print("按轴求和 axis=0:\n", A.sum(axis=0))


def demo_index_slice():
    M = np.arange(12).reshape(3, 4)
    print("M:\n", M)
    print("M[1, 2] =", M[1, 2])
    print("第2行:", M[1, :])
    print("前两列:\n", M[:, :2])
    # 布尔索引：筛选
    print("偶数:", M[M % 2 == 0])


def demo_distance_matrix():
    """建模常用：n 个点两两欧氏距离矩阵。"""
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    # (n,1,2) - (1,n,2) -> (n,n,2) 再对最后一维求范数
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    print("距离矩阵:\n", np.round(dist, 3))


if __name__ == "__main__":
    np.random.seed(0)
    print("=" * 40, "创建与生成")
    demo_array_basics()
    print("=" * 40, "运算")
    demo_ops()
    print("=" * 40, "索引")
    demo_index_slice()
    print("=" * 40, "距离矩阵")
    demo_distance_matrix()
