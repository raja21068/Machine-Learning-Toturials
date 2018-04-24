#Telecom Is a customer likely to leave the network? (churn prediction)
#Retail Is he a prospective customer?, that is, likelihood of purchase vs. non-purchase?
#Insurance To issue insurance, should a customer be sent for a medical checkup?
#Insurance Will the customer renew the insurance?
#Banking Will a customer default on the loan amount?
#Banking Should a customer be given a loan?
#Manufacturing Will the equipment fail?
#Health Care Is the patient infected with a disease?
#Health Care What type of disease does a patient have?
#Entertainment What is the genre of music?

#The answers to these questions are a discrete class. The number of level or class can va
# ry from a minimum of two (example: true or false, yes or no) to multiclass

#Let’s consider a use case where we have to predict students’ test outcomes, that is, pass
#(1) or fail (0) based on hours studied. In this case the outcome to be predicted is discrete.
#Let’s build a linear regression and try to use a threshold, that is, anything over some value

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# importing linear regression function
import sklearn.linear_model as lm

# function to calculate r-squared, MAE, RMSE
from sklearn.metrics import r2_score , mean_absolute_error, mean_squared_error


# Load data
df = pd.read_csv('chapter3/Grade_Set_1_Classification.csv')
#print (df)

x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Result # dependent variable

# Create linear regression object
lr = lm.LinearRegression()

# Train the model using the training sets
lr.fit(x, y)

# plotting fitted line
plt.scatter(x, y, color='black')
plt.plot(x, lr.predict(x), color='blue', linewidth=3)
plt.title('Hours Studied vs Result')
plt.ylabel('Result')
plt.xlabel('Hours_Studied')
plt.show()
# add predict value to the data frame
df['Result_Pred'] = lr.predict(x)


# Using built-in function
print("R Squared : ", r2_score(df.Result, df.Result_Pred))
print("Mean Absolute Error: ", mean_absolute_error(df.Result, df.Result_Pred))
print ("Root Mean Squared Error: ", np.sqrt(mean_squared_error(df.Result,df.Result_Pred)))

# plot sigmoid function
x = np.linspace(-10, 10, 100)
y = 1.0 / (1.0 + np.exp(-x))
plt.plot(x, y, 'r-', label='logit')
plt.legend(loc='lower right')
plt.show()

