
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
df = pd.read_csv('chapter3/Grade_Set_2.csv')

# Simple scatter plot
df.plot(kind='scatter', x='Hours_Studied', y='Test_Grade', title='Grade vsHours Studied')
print (df)
# check the correlation between variables
print("Correlation Matrix: ")
df.corr()

# Create linear regression object
lr = lm.LinearRegression()

# add predict value to the data frame
x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade                      # dependent variable



# Train the model using the training sets
lr.fit(x, y)





# plotting fitted line
plt.scatter(x, y, color='black')
plt.plot(x, lr.predict(x), color='blue', linewidth=3)
plt.title('Grade vs Hours Studied')
plt.ylabel('Test_Grade')
plt.xlabel('Hours_Studied')

print ("R Squared: ", r2_score(y, lr.predict(x)))