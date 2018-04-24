# Fitting a Slope
# Let’s try to fit a slope line through all the points such that the error
# The error could be positive or negative based on its location from the slope
# Y = mX + c, where Y is the predicted value for a given x value.
# m is the change in y, divided by change in x, that is, m is the slope of the line for the x variable
#  and it indicates the steepness at which it increases with every unit increase in x variable value
# c is the intercept that indicates the location or point on the axis where it intersects,

# Together the slope and intercept define the linear relationship between the two
# variables and can be used to predict or estimate an average rate of change

#Say a student is planning to study an overall of 6 hours in preparation for the test.
# Simply drawing a connecting line from the x-axis and y-axis to the slope shows that there is a possibility of him scoring 80.
# importing linear regression function

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

x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade                      # dependent variable

# Create linear regression object
lr = lm.LinearRegression()

# Train the model using the training sets
lr.fit(x, y)

print ("Intercept: ", lr.intercept_)
print ("Coefficient: ", lr.coef_)

# predict using the built-in function
print ("Using predict function: ", lr.predict(1))



# plotting fitted line
plt.scatter(x, y,  color='black')
plt.plot(x, lr.predict(x), color='blue', linewidth=3)
plt.title('Hours Studied vs Result')
plt.ylabel('Result')
plt.xlabel('Hours_Studied')
plt.show()
# add predict value to the data frame

# Using built-in function
#print ("R Squared : ", r2_score(df.Result, df.Result_Pred))
#print ("Mean Absolute Error: ", mean_absolute_error(df.Result, df.Result_Pred))
#print ("Root Mean Squared Error: ", np.sqrt(mean_squared_error(df.Result, df.Result_Pred)))