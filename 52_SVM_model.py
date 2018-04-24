from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import metrics


iris = datasets.load_iris()

X = iris.data[:, [2, 3]]
y = iris.target
#print('Class labels:', np.unique(y))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# split data into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)


from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1.0, random_state=0)
clf.fit(X_train, y_train)

# generate evaluation metrics
print ("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
print ("Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train)))
print ("Train - classification report :", metrics.classification_report(y_train, clf.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
print ("Test - Confusion matrix :",metrics.confusion_matrix(y_test, clf.predict(X_test)))
