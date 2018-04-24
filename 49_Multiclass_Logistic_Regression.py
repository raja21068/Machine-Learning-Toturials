#Logistic regression can also be used to predict the dependent or target variable with
#multiclass. Let’s learn multiclass prediction with iris dataset, one of the best-known
#databases to be found in the pattern recognition literature. The dataset contains 3 classes
#of 50 instances each, where each class refers to a type of iris plant. This comes as part
#of the scikit-learn datasets, where the third column represents the petal length, and the
#fourth column the petal width of the flower samples. The classes are already converted to
#integer labels where 0=Iris-Setosa, 1=Iris-Versicolor, 2=Iris-Virginica.

from sklearn import datasets
import numpy as np
import pandas as pd
iris = datasets.load_iris()
X = iris.data
y = iris.target
print('Class labels:', np.unique(y))

#Normalize Data
#The unit of measurement might differ so let’s normalize the data before building the model

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

#Split data into train and test. Whenever we are using random function it’s advised to use a
#seed to ensure the reproducibility of the results
# split data into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

#train logistic regression model traning and evloution
from sklearn.linear_model import LogisticRegression
# l1 regularization gives better results
lr = LogisticRegression(penalty='l1', C=10, random_state=0)
lr.fit(X_train, y_train)


from sklearn import metrics
# generate evaluation metrics
print ("Train - Accuracy :", metrics.accuracy_score(y_train, lr.predict(X_train)))
print ("Train - Confusion matrix :",metrics.confusion_matrix(y_train,lr.predict(X_train)))
print ("Train - classification report :", metrics.classification_report(y_train, lr.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, lr.predict(X_test)))
print ("Test - Confusion matrix :",metrics.confusion_matrix(y_test,lr.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, lr.predict(X_test)))
