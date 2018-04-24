import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn.datasets import load_breast_cancer

# Load data
cancer = load_breast_cancer()
cancer.keys()




# 569 data points with 30 features
cancer['data'].shape

X = cancer['data']
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Standardazie data, fit only to the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)

print ("Confusion Matrix: \n", confusion_matrix(y_test,predictions))
print ("Classification Repor: \n", classification_report(y_test,predictions))