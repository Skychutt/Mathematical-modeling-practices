from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))

print("Shape of cancer data: {}".format(cancer.data.shape))

print("Sample counts per class:\n{}".format({n: v for n,v in zip(cancer.target_names,np.bincount(cancer.target)) }))

print("Feature names:\n{}".format(cancer.feature_names))

