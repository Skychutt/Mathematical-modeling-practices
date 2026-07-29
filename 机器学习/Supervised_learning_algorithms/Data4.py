from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures

# 自定义拓展数据集函数
def load_extended_diabetes():
    diabetes = load_diabetes()
    poly = PolynomialFeatures(degree=2)
    X_ext = poly.fit_transform(diabetes.data)
    return X_ext, diabetes.target

# 主程序（和你原来代码结构几乎一模一样）
diabetes = load_diabetes()
print("Data shape: {}".format(diabetes.data.shape))

X, y = load_extended_diabetes()
print("X.shape: {}".format(X.shape))