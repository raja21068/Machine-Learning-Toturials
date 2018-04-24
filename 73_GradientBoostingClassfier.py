# Bagged Decision Trees for Classification

import pandas as pd
from sklearn.cross_validation import train_test_split

from sklearn import  cross_validation
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier


# read the data in
df = pd.read_csv("chapter4/digit.csv")

# Let's use some week features to build the tree
X = df.ix[:,1:17].values
y = df['lettr'].values



# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=2017)

kfold = cross_validation.StratifiedKFold(y=y_train, n_folds=5, random_state=2017)
num_trees = 10

clf_GBT = GradientBoostingClassifier(n_estimators=num_trees, learning_rate=0.1, max_depth=3, random_state=2017).fit(X_train, y_train)
results = cross_validation.cross_val_score(clf_GBT, X_train, y_train,cv=kfold)

print ("Gradient Boosting - Test : ", metrics.accuracy_score(clf_GBT.predict(X_test), y_test))
