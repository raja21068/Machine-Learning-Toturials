#Stochastic Gradient Descent
#Fitting the right slope that minimizes the error (also known as cost function) for a large
#dataset can be tricky. However this can be achieved through a stochastic gradient descent
#(steepest descent) optimization algorithm. In case of regression problems, the cost
#function J to learn the weights can be defined as the sum of squared errors (SSE) between
#actual vs. predicted value.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# importing linear regression function
import sklearn.linear_model as lm

# function to calculate r-squared, MAE, RMSE
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
1
# Load data
df = pd.read_csv('chapter3/Grade_Set_1_Classification.csv')
#print (df)

# manually add intercept
df['intercept'] = 1
independent_variables = ['Hours_Studied', 'intercept']

x = df[independent_variables] # independent variable
y = df['Result'] # dependent variable

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
model = model.fit(x, y)
111
from sklearn import metrics

# Determine the false positive and true positive rates
fpr, tpr, _ = metrics.roc_curve(y, model.predict_proba(x)[:,1])
# Calculate the AUC
roc_auc = metrics.auc(fpr, tpr)
print ('ROC AUC: %0.2f' % roc_auc)
# Plot of a ROC curve for a specific class
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()