import pandas as pd
import numpy as np
import scipy.stats as stats

# importing linear regression function
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.model_selection._split import train_test_split
import matplotlib.pyplot as plt


def draw_plot():

    plt.figure(figsize=(6, 6))
    plt.scatter(np.extract(pos, x1),
    np.extract(pos, x2),
    c='b', marker='s', label='pos')
    plt.scatter(np.extract(neg, x1),np.extract(neg, x2),c='r', marker='o', label='neg')
    plt.xlabel('x1');
    plt.ylabel('x2');
    plt.axes().set_aspect('equal', 'datalim')
    plt.legend();



# create hihger order polynomial for independent variables
order_no = 6
# map the variable 1 & 2 to its higher order polynomial
def map_features(variable_1, variable_2, order=order_no):
    assert order >= 1
    def iter():
        for i in range(1, order + 1):
            for j in range(i + 1):
                yield np.power(variable_1, i - j) * np.power(variable_2, j)
    return np.vstack(iter())


def draw_boundary(classifier):
    dim = np.linspace(-0.8, 1.1, 100)
    dx, dy = np.meshgrid(dim, dim)
    v = map_features(dx.flatten(), dy.flatten(), order=order_no)
    z = (np.dot(classifier.coef_, v) + classifier.intercept_).reshape(100, 100)
    plt.contour(dx, dy, z, levels=[0], colors=['r'])


data = pd.read_csv('chapter3/LR_NonLinear.csv')
pos = data['class'] == 1
neg = data['class'] == 0
x1 = data['x1']
x2 = data['x2']
# function to draw scatter plot between two variables

out = map_features(data['x1'], data['x2'], order=order_no)
X = out.transpose()
y = data['class']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

# function to draw classifier line


# fit with c = 0.01
clf = LogisticRegression(C=0.01).fit(X_train, y_train)
print ('Train Accuracy for C=0.01: ', clf.score(X_train, y_train))
print ('Test Accuracy for C=0.01: ', clf.score(X_test, y_test))
draw_plot()
plt.title('Fitting with C=0.01')
draw_boundary(clf)
plt.legend();
plt.show(plt.legend)

# fit with c = 1
clf = LogisticRegression(C=1).fit(X_train, y_train)
print ('Train Accuracy for C=1: ', clf.score(X_train, y_train))
print ('Test Accuracy for C=1: ', clf.score(X_test, y_test))
draw_plot()

plt.title('Fitting with C=1')
draw_boundary(clf)
plt.legend();
plt.show(plt.legend)
# fit with c = 10000
clf = LogisticRegression(C=10000).fit(X_train, y_train)
print ('Train Accuracy for C=10000: ', clf.score(X_train, y_train))
print ('Test Accuracy for C=10000: ', clf.score(X_test, y_test))
draw_plot()
plt.title('Fitting with C=10000')
draw_boundary(clf)
plt.legend();
plt.show(plt.legend)