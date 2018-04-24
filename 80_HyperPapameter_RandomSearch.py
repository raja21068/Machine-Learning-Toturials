import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
# Classifiers
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

seed = 2017


# read the data in
df = pd.read_csv("chapter4/Diabetes.csv")

X = df.ix[:,0:8] # independent variables
y = df['class'].values # dependent variables

#Normalize
X = StandardScaler().fit_transform(X)

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)
kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 100
clf_rf = RandomForestClassifier(random_state=seed).fit(X_train, y_train)

# specify parameters and distributions to sample from
param_dist = {'n_estimators':sp_randint(100,1000),
              'criterion': ['gini', 'entropy'],
                'max_features': [None, 'auto', 'sqrt', 'log2'],
                'max_depth': [None, 1, 3, 5, 7, 9]}

#run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf_rf, param_distributions=param_dist, cv=kfold,n_iter=n_iter_search, verbose=10, n_jobs=-1, random_state=seed)
random_search.fit(X_train, y_train)
# report(random_search.cv_results_)

print ('Best Parameters: ', random_search.best_params_)
results = cross_validation.cross_val_score(random_search.best_estimator_,X_train,y_train, cv=kfold)
print ("Accuracy - Train CV: ", results.mean())
print ("Accuracy - Train : ", metrics.accuracy_score(random_search.best_estimator_.predict(X_train), y_train))
print ("Accuracy - Test : ", metrics.accuracy_score(random_search.best_estimator_.predict(X_test), y_test))

