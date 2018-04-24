from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm

# importing linear regression function
import sklearn.linear_model as lm

# function to calculate r-squared, MAE, RMSE
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error


# Load data
df = pd.read_csv('chapter3/Grade_Set_1.csv')


# add predict value to the data frame
x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade                      # dependent variable


# Create linear regression object
lr = lm.LinearRegression()

degree = 4
model = make_pipeline(PolynomialFeatures(degree), lr)

# Train the model using the training sets
model.fit(x, y)

plt.scatter(x, y, color='black')
plt.plot(x, model.predict(x), color='green')
print ("R Squared using built-in function: ", r2_score(y, model.predict(x)))
