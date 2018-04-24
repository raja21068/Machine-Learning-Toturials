#GLM was an effort by John Nelder and Robert Wedderburn to unify commonly used
#various statistical models such as linear, logistic, and poisson, etc
#Binomial   Target variable is binary response.
#Poisson    Target variable is a count of occurrence.
#Gaussian   Target variable is a continuous number.
#Gamma      This distribution arises when the waiting times between Poisson
#            distribution events are relevant, that is, the number of events
#              occurred between two time periods.
#InverseGaussian The tails of the distribution decrease slower than normal distribution, t
#                  hat is, there is an inverse relationship between
#                  the time required to cover a unit distance and distance covered in unit time.
#NegativeBinomial  Target variable denotes number

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lm

df = pd.read_csv('Data/Grade_Set_1.csv')

print('####### Linear Regression Model ########')
# Create linear regression object
lr = lm.LinearRegression()

x= df.Hours_Studied[:, np.newaxis] # independent variable
y= df.Test_Grade.values # dependent variable

# Train the model using the training sets
lr.fit(x, y)

print ("Intercept: ", lr.intercept)
print ("Coefficient: ", lr.coef_)

print('\n####### Generalized Linear Model ########')
import statsmodels.api as sm

# To be able to run GLM, we'll have to add the intercept constant to x variable
x = sm.add_constant(x, prepend=False)

# Instantiate a gaussian family model with the default link function.
model = sm.GLM(y, x, family = sm.families.Gaussian())
model = model.fit()
print (model.summary())


#However GLM can be used for other distributions such as binomial, poisson, etc., by just changing the family parameter.

