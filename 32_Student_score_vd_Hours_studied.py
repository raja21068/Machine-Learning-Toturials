import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
# Load data
df = pd.read_csv('chapter3/Grade_Set_1.csv')
print("---Show Data Set---------------")
print (df)

# Simple scatter plot
df.plot(kind='scatter', x='Hours_Studied', y='Test_Grade', title='Grade vsHours Studied')

print("------check the correlation between variables------------")
# check the correlation between variables
print (df.corr())

