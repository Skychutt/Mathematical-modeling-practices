class Person:
    # 构造方法 __init__
    def __init__(self, name):
        # self ≈ Java的this，**必须写在第一个参数，不能省略**
        self.name = name


    def say_hello(self):
        print(f"我是{self.name}")


# 实例化，不需要 new！
p = Person("张三")
print(p.name)
p.say_hello()

