import mglearn
import matplotlib.pyplot as plt
#生成数据集
X, y = mglearn.datasets.make_forge() #make_forge()： 是分类数据集（用来做分类任务）
# 生成一个小型二分类数据集（经典监督学习演示数据集）
#数据集绘制
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0","Class 1"],loc = 4)  #legend：图例，loc=4 代表右下角；
#xlabel /ylabel横、纵轴文字说明。
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
print("X.shape: {}".format(X.shape))
#X.shape 输出：(26, 2)
#含义：一共 26 个样本，每个样本2 个特征


"""
X[0, :]     # 第0行，所有列 → 第0个样本全部特征
X[:, 1]     # 所有行，第1列 → 全部样本第1个特征
X[2:5, :]   # 第2、3、4行，所有列（左闭右开！不含5）
X[10:, 0]   # 第10行到最后一行，第0列
X[:8, 1]    # 前8行，第1列"""


"""mglearn.discrete_scatter(X[:, 0], X[:, 1], y)里
第一个参数 X[:, 0]：所有样本的第 0 号特征（第一个特征） → 作为绘图的横坐标
第二个参数 X[:, 1]：所有样本的第 1 号特征（第二个特征） → 作为绘图的纵坐标
第三个参数 y：每一条样本对应的类别标签，函数依靠这个标签区分样本，绘制不同颜色 / 形状的点。
第i个样本：
横坐标 = X[i,0]
纵坐标 = X[i,1]
类别 = y[i]
数组顺序不能乱！三组数据长度必须一模一样（这里都是 26）。"""