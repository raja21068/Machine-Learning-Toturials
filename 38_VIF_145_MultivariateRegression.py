import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from sklearn.cross_validation import train_test_split


# Load data
df = pd.read_csv('chapter3/Housing_Modified.csv')
# Convert binary fields to numeric boolean fields
lb = preprocessing.LabelBinarizer()
df.driveway = lb.fit_transform(df.driveway)
df.recroom = lb.fit_transform(df.recroom)
df.fullbase = lb.fit_transform(df.fullbase)
df.gashw = lb.fit_transform(df.gashw)
df.airco = lb.fit_transform(df.airco)
df.prefarea = lb.fit_transform(df.prefarea)

print("----------------------------")
print(df.head())
# Create dummy variables for stories
df_stories = pd.get_dummies(df['stories'], prefix='stories', drop_first=True)

#print("----------------DF_Stories------------")

#print(df_stories)
# Join the dummy variables to the main dataframe
df = pd.concat([df, df_stories], axis=1)
del df['stories']


# lets plot correlation matrix using statmodels graphics packages's plot_corr
# create correlation matrix
corr = df.corr()
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
#plt.show(sm.graphics.plot_corr(corr, xnames=list(corr.columns)))

# Alternatively we can use seaborn package

import seaborn as sns

arr=sns.heatmap(df.corr(), vmax=.8, square=True,annot=True,fmt='.1f')
plt.show(arr)