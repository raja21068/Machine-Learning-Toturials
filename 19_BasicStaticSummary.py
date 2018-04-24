import pandas as pd
import numpy as np

print("Basic statistics on dataframe")
df = pd.read_csv('chapter2/iris.csv')
#describe()- will returns the quick stats such as count, mean, std (standard
#deviation), min, first quartile, median, third quartile, max on each column
#of the dataframe
print("-------------df.Desccribe--------------------")
print(df.describe())

#cov() - Covariance indicates how two variables are related. A positive
#covariance means the variables are positively related, while a negative
#covariance means the variables are inversely related. Drawback of covariance
#is that it does not tell you the degree of positive or negative relation
print("-------------df.cov--------------------")
print(df.cov())
