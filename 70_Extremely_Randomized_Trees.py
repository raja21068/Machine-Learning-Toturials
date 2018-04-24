# Bagged Decision Trees for Classification
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import  cross_validation
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
# read the data in
df = pd.read_csv("chapter4/Diabetes.csv")

X = df.ix[:,:8].values # independent variables
y = df['class'].values # dependent variables

#Normalize
X = StandardScaler().fit_transform(X)
# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)

kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 100

# Decision Tree with 5 fold cross validation
clf_ET = ExtraTreesClassifier(n_estimators=num_trees).fit(X_train, y_train)
results = cross_validation.cross_val_score(clf_ET, X_train, y_train, cv=kfold)


print ("\nExtraTrree - Train : ", results.mean())
print ("ExtraTree - Test : ", metrics.accuracy_score(clf_ET.predict(X_test), y_test))

