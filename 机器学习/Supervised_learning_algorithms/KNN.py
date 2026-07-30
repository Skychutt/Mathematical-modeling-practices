import mglearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors = 3)

clf.fit(X_train, y_train)

print("Test set predictions: {}".format(clf.predict(X_test)))

print("True result: {}".format(y_test))

print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1,3, figsize=(10,3))

for n_neighbors, ax in zip([1,3,9],axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf, X, fill = True, eps = 0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
    ax.set_title("{} neighbors(s)".format(n_neighbors))
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
axes[0].legend(loc = 3)
plt.show()

"""
# zip([1,3,9],axes) 把两个序列一一配对：
# 第1轮：n_neighbors=1，ax=axes[0]（第一张子图）
# 第2轮：n_neighbors=3，ax=axes[1]（第二张子图）
# 第3轮：n_neighbors=9，ax=axes[2]（第三张子图）
for n_neighbors, ax in zip([1, 3, 9], axes):
    # 创建KNN模型，设置当前轮的k值；使用全部数据集X,y训练模型
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)

    # 绘制KNN的决策边界
    # clf：训练好的模型 | X：数据集（用来确定绘图范围）
    # fill=True：给不同类别区域填充颜色
    # eps=0.5：向外扩大绘图边界，避免点贴到图片边缘
    # ax=ax：指定画在当前循环对应的子图上
    # alpha=.4：填充颜色透明度（0完全透明，1不透明）
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)

    # 在当前子图绘制原始样本散点，自动根据y区分圆点/三角
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)

    # 设置当前子图上方标题，显示当前K值
    ax.set_title("{} neighbors(s)".format(n_neighbors))
    # 设置当前子图横轴文字
    ax.set_xlabel("Feature 0")
    # 设置当前子图纵轴文字
    ax.set_ylabel("Feature 1")

# 只在第一个子图（axes[0]）添加图例，位置loc=3=左下角
axes[0].legend(loc=3)
"""