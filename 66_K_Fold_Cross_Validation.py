from sklearn.cross_validation import cross_val_score
import pandas as pd
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# read the data in
df = pd.read_csv("chapter4/Diabetes.csv")
X = df.ix[:,:8].values # independent variables
y = df['class'].values # dependent variables

# Normalize Data
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

# evaluate the model by splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=2017)

# build a decision tree classifier
clf = tree.DecisionTreeClassifier(random_state=2017)

# evaluate the model using 10-fold cross-validation
train_scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=5)
test_scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=5)

print ("Train Fold AUC Scores: ", train_scores)
print ("Train CV AUC Score: ", train_scores.mean())