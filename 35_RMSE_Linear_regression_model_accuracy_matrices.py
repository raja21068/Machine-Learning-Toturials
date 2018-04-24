
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


# Train the model using the training sets
lr.fit(x, y)




df['Test_Grade_Pred'] = lr.predict(x)
# Manually calculating R Squared
df['SST'] = np.square(df['Test_Grade'] - df['Test_Grade'].mean())
df['SSR'] = np.square(df['Test_Grade_Pred'] - df['Test_Grade'].mean())

print ("Sum of SSR:", df['SSR'].sum())
print ("Sum of SST:", df['SST'].sum())
print ("R Squared using manual calculation: ", df['SSR'].sum() / df['SST'].sum())
# Using built-in function

#R-squared value designates the total proportion of variance in the dependent
#variable explained by the independent variable. It is a value between 0 and 1; the value
#toward 1 indicates a better model fit
#R-squared =Total Sum of Square Residual (Σ SSR) / Sum of Square Total(Σ SST)
print ("R Squared using built-in function: ", r2_score(df.Test_Grade, y))

#This is the mean or average of absolute value of the errors, that is, the predicted - actual.
print ("Mean Absolute Error: ", mean_absolute_error(df.Test_Grade, df.Test_Grade_Pred))

# RMSE indicates how close the predicted values are to the actual values; hence a lower RMSE value signifies that the model performance is good
print ("Root Mean Squared Error: ", np.sqrt(mean_squared_error(df.Test_Grade,df.Test_Grade_Pred)))