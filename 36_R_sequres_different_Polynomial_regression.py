
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


# Create linear regression object
lr = lm.LinearRegression()

# add predict value to the data frame
x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade                      # dependent variable


for deg in[1,2,3,4,5]:
    lr.fit(np.vander(x,deg+1),y)
    y_lr=lr.predict(np.vander((x,deg+1)))
    plt.plot(x,y_lr,label='degree'+str(deg))
    plt.legend(loc=2)
    print(r2_score(y,y_lr))
plt.plot(x,y,'ok')
