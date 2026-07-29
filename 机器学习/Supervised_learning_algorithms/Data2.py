import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_wave(n_samples=40)
#是回归数据集（用来演示监督学习里的回归 Regression）
plt.plot(X, y,"o")
plt.ylim(-3,3)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()






