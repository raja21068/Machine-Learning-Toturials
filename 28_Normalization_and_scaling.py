from sklearn import datasets
import numpy as np
from sklearn import preprocessing

iris=datasets.load_iris()
X=iris.data[:,[2,3]]
Y=iris.target

std_scale= preprocessing.MinMaxScaler().fit(X)
X_std=std_scale.transform(X)

minmax_scale=preprocessing.MinMaxScaler().fit(X)
X_minmax=minmax_scale.transform(X)
print("----Mean----")
print('Mean before standardization: petal length={:.1f}, petal width={:.1f}'.format(X[:,0].mean(), X[:,1].mean()))
print('Mean after standardization: petal length={:.1f}, petal width={:.1f}'.format(X_std[:,0].mean(), X_std[:,1].mean()))
print("----SD----")
print('SD before standardization: petal length={:.1f}, petal width={:.1f}'.format(X[:,0].std(), X[:,1].std()))
print('SD after standardization: petal length={:.1f}, petal width={:.1f}'.format(X_std[:,0].std(), X_std[:,1].std()))
print("----Min Value----")
print('Min value before min-max scaling: patel length={:.1f}, patelwidth={:.1f}'.format(X[:,0].min(), X[:,1].min()))
print('Min value after min-max scaling: patel length={:.1f}, patelwidth={:.1f}'.format(X_minmax[:,0].min(), X_minmax[:,1].min()))
print("----Max Value----")
print('Max value before min-max scaling: petal length={:.1f}, petalwidth={:.1f}'.format(X[:,0].max(), X[:,1].max()))
print('Max value after min-max scaling: petal length={:.1f}, petal width={:.1f}'.format(X_minmax[:,0].max(), X_minmax[:,1].max()))

