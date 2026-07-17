"""
数学建模 · Python 基础 02：列表 / 元组 / 字典 / 集合
建模中：列表存解向量，字典存参数，集合做去重。
"""


def demo_list():
    # 可变序列，下标从 0 开始
    xs = [3, 1, 4, 1, 5]
    print("原列表:", xs)
    print("长度:", len(xs), "求和:", sum(xs), "最大:", max(xs))
    print("切片 xs[1:4]:", xs[1:4])
    print("倒序:", xs[::-1])

    xs.append(9)       # 末尾追加
    xs.extend([2, 6])  # 拼接多个
    xs.insert(0, 0)    # 指定位置插入
    print("修改后:", xs)

    # 列表推导式：生成候选解、过滤数据时非常好用
    squares = [i ** 2 for i in range(6)]
    evens = [x for x in xs if x % 2 == 0]
    print("平方:", squares, "偶数:", evens)

    # 排序（不影响原列表用 sorted；原地排序用 .sort）
    print("排序副本:", sorted(xs))
    print("降序:", sorted(xs, reverse=True))


def demo_tuple():
    # 不可变序列：适合当字典键、返回多值
    point = (1.0, 2.5)
    x, y = point  # 解包
    print("点:", point, "x=", x, "y=", y)

    def min_max(arr):
        return min(arr), max(arr)

    lo, hi = min_max([3, 8, 1, 5])
    print("min_max:", lo, hi)


def demo_dict():
    # 键值对：参数表、邻接表、计数器
    params = {"pop_size": 50, "iters": 200, "pc": 0.8, "pm": 0.05}
    print("种群规模:", params["pop_size"])
    params["iters"] = 300
    params["elitism"] = True
    print("全部键:", list(params.keys()))
    print("遍历:")
    for k, v in params.items():
        print(f"  {k} = {v}")

    # get：键不存在时给默认值，避免 KeyError
    print("不存在的键:", params.get("alpha", 0.1))


def demo_set():
    # 去重、集合运算
    a = {1, 2, 3, 3, 2}
    b = {2, 3, 4}
    print("去重后 a:", a)
    print("并:", a | b, "交:", a & b, "差:", a - b)


if __name__ == "__main__":
    print("=" * 40, "列表")
    demo_list()
    print("=" * 40, "元组")
    demo_tuple()
    print("=" * 40, "字典")
    demo_dict()
    print("=" * 40, "集合")
    demo_set()
