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
from sklearn.linear_model import LogisticRegression


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

# check the accuracy on the training set
model.score(x, y)
print (model.predict(x))
print (model.predict_proba(x)[:,0])


# plotting fitted line
plt.scatter(df.Hours_Studied, y, color='black')
plt.yticks([0.0, 0.5, 1.0])
plt.plot(df.Hours_Studied, model.predict_proba(x)[:,1], color='blue',linewidth=3)
plt.title('Hours Studied vs Result')
plt.ylabel('Result')
plt.xlabel('Hours_Studied')
plt.show()