import numpy as np

x = np.array([[1,2,3],[4,5,6]])
print(x)
print(x.shape)
print(x.dtype)
print("x:\n", x)
print("x:\n{}".format(x))



# 1. 查看变量是什么类型
type(x)       # numpy.ndarray

# 2. 查看文档（最重要！等价Java查看类/方法注释文档）
help(x)               # 查看 ndarray 整个类文档
help(x.shape)         # 查看属性
help(np.array)        # 查看np.array函数文档

# 3. 列出这个对象所有可用属性、方法（等价看类所有成员）
dir(x)