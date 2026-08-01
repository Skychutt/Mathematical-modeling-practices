from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import mglearn
import numpy as np

X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
print("斜率{}".format(lr.coef_))
print("截距{}".format(lr.intercept_))

#这种情况是欠拟合 训练集和测试机评分接近且都比较低
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


#这种情况是过拟合，训练集准但是测试机不够准，因为506样本，105特征，特征数过多导致拟合程度太大
X,y  = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))


#R^2=1-预测残差平方和/真实值均值的总平方和，
#通俗来说：对比模型预测产生的误差和永远猜均值的误差，代表模型相比无脑猜均值能降低多少误差，最高为1，越小效果越差，负数说明模型还不如直接猜均值。

#用岭回归（ridge regression）来解决这个问题，可以控制复杂度的模型

"""一群人真实体重：50、60、70，平均值 = 60
笨办法总误差：(50-60)²+(60-60)²+(70-60)² = 200
你的模型预测：51、59、70
模型总误差：(50-51)²+(60-59)²+(70-70)² = 2
R² = 1 - (2 / 200) = 0.99
意思：相比直接猜平均体重，模型消除了 99% 的误差。"""
