"""
数学建模 · Python 基础 01：变量、类型与运算符
运行本文件即可看到示例输出。
"""


def demo_variables():
    # 动态类型：赋值即定义，无需声明类型
    x = 10
    pi = 3.14159
    name = "建模"
    ok = True
    nothing = None

    print("整数:", x, type(x))
    print("浮点:", pi, type(pi))
    print("字符串:", name, type(name))
    print("布尔:", ok, type(ok))
    print("空值:", nothing, type(nothing))

    # 类型转换（读入数据、处理表格时常用）
    print("str -> int:", int("42"))
    print("float -> int:", int(3.9))  # 截断，不是四舍五入
    print("int -> float:", float(7))
    print("任意 -> str:", str(3.14))


def demo_operators():
    a, b = 17, 5
    print(f"{a} + {b} =", a + b)
    print(f"{a} - {b} =", a - b)
    print(f"{a} * {b} =", a * b)
    print(f"{a} / {b} =", a / b)    # 真除法，结果为 float
    print(f"{a} // {b} =", a // b)  # 整除
    print(f"{a} % {b} =", a % b)    # 取余
    print(f"{a} ** {b} =", a ** b)  # 幂

    # 比较与逻辑（约束判断、条件分支）
    print("a > b:", a > b)
    print("and:", a > 10 and b < 10)
    print("or:", a < 0 or b == 5)
    print("not:", not False)


def demo_string():
    # f-string：建模写日志、打印结果最常用
    n, err = 100, 1.23e-4
    print(f"迭代 {n} 次, 误差={err:.6e}")

    s = "TSP,VRP,GA"
    print("分割:", s.split(","))
    print("拼接:", "-".join(["A", "B", "C"]))
    print("包含:", "GA" in s)


if __name__ == "__main__":
    print("=" * 40, "变量与类型")
    demo_variables()
    print("=" * 40, "运算符")
    demo_operators()
    print("=" * 40, "字符串")
    demo_string()
