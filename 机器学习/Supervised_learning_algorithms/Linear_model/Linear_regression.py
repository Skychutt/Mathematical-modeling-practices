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
X,y  = mglearn.datasets.make_wave(n_samples=60)





