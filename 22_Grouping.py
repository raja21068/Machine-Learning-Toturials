import numpy as np
import pandas as pd

df = pd.DataFrame({'Name' : ['jack', 'jane', 'jack', 'jane', 'jack', 'jane',
'jack', 'jane'],
'State' : ['SFO', 'SFO', 'NYK', 'CA', 'NYK', 'NYK',
'SFO', 'CA'],
'Grade':['A','A','B','A','C','B','C','A'],
'Age' : np.random.uniform(24, 50, size=8),
'Salary' : np.random.uniform(3000, 5000, size=8),})\


# Note that the columns are ordered automatically in their alphabetic order df
# for custom order please use below code
# df = pd.DataFrame(data, columns = ['Name', 'State', 'Age','Salary'])
# Find max age and salary by Name / State
# with groupby, we can use all aggregate functions such as min, max, mean, count, cumsum
print("--------Data Frame-------------")
print(df)
print("-------Find Max age and salary byu Name / State-------")
print(df.groupby(['Name','State']).max())


print("-------------Pivot Table-------------")
print("by state and name find mean age for each grade")
pd.pivot_table(df, values='Age', index=['State', 'Name'], columns=['Grade'])