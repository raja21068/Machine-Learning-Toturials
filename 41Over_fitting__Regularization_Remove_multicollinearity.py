# With an increase in number of variables, and increase in model complexity, the probability
# of over-fitting also increases. Regularization is a technique to avoid the over-fitting
# problem. Over-fitting occurs when the model fits the data too well, capturing all the noises.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection._split import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn import linear_model


# Load data
df = pd.read_csv('chapter3/Grade_Set_2.csv')
df.columns = ['x','y']

for i in range(2,50):              # power of 1 is already there
    colname = 'x_%d'%i              # new var will be x_power

    df[colname] = df['x']**i

independent_variables = list(df.columns)
independent_variables.remove('y')

X= df[independent_variables]       # independent variable
y= df.y                            # dependent variable

# split data into train and test
X_train, X_test, y_train, y_test=train_test_split(X,y,train_size=.80,random_state=1)

# Ridge regression
lr = linear_model.Ridge(alpha=0.001)
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print("------ Ridge Regression ------")
print ("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
print ("Train RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print ("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
print ("Test RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print ("Ridge Coef: ", lr.coef)
# LASSO regression
lr = linear_model.Lasso(alpha=0.001)
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

print("----- LASSO Regression -----")
print ("Train MAE: ", metrics.mean_absolute_error(y_train, y_train_pred))
print ("Train RMSE: ", np.sqrt(metrics.mean_squared_error(y_train, y_train_pred)))

print ("Test MAE: ", metrics.mean_absolute_error(y_test, y_test_pred))
print ("Test RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))
print ("LASSO Coef: ", lr.coef_)