"""
数学建模 · Python 基础 03：条件与循环
"""


def demo_if():
    score = 82
    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 60:
        grade = "C"
    else:
        grade = "D"
    print(f"分数 {score} -> 等级 {grade}")

    # 三元表达式
    flag = "可行" if score >= 60 else "不可行"
    print(flag)


def demo_for():
    # range(n): 0..n-1；range(a,b): a..b-1；range(a,b,step)
    print("range(5):", list(range(5)))
    print("range(1, 10, 2):", list(range(1, 10, 2)))

    total = 0
    for i in range(1, 6):
        total += i
    print("1+..+5 =", total)

    cities = ["北京", "上海", "广州"]
    for idx, city in enumerate(cities):
        print(f"  第{idx}个: {city}")

    # zip：并行遍历两条序列（配对坐标、两条曲线等）
    xs = [0, 1, 2]
    ys = [0, 1, 4]
    for x, y in zip(xs, ys):
        print(f"  ({x}, {y})")


def demo_while_and_break():
    # 迭代算法常用 while：未达标就继续
    x, step = 0.0, 0
    target = 10.0
    while x < target:
        x += 1.5
        step += 1
        if step > 100:  # 防止死循环
            break
    print(f"while 结果 x={x}, 步数={step}")

    # continue：跳过本轮
    odds = []
    for i in range(10):
        if i % 2 == 0:
            continue
        odds.append(i)
    print("奇数:", odds)


def demo_nested():
    # 双重循环：距离矩阵、两两比较
    pts = [(0, 0), (1, 0), (0, 1)]
    print("点对距离平方:")
    for i, (x1, y1) in enumerate(pts):
        for j, (x2, y2) in enumerate(pts):
            if i >= j:
                continue
            d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
            print(f"  P{i}-P{j}: {d2}")


if __name__ == "__main__":
    print("=" * 40, "条件")
    demo_if()
    print("=" * 40, "for")
    demo_for()
    print("=" * 40, "while")
    demo_while_and_break()
    print("=" * 40, "嵌套循环")
    demo_nested()
