import pandas as pd
import numpy as np


data = {
'emp_id': ['1', '2', '3', '4', '5'],
'first_name': ['Jason', 'Andy', 'Allen', 'Alice', 'Amy'],
'last_name': ['Larkin', 'Jacob', 'A', 'AA', 'Jackson']}
df_1 = pd.DataFrame(data, columns = ['emp_id', 'first_name', 'last_name'])


data = {
'emp_id': ['4', '5', '6', '7'],
'first_name': ['Brian', 'Shize', 'Kim', 'Jose'],
'last_name': ['Alexander', 'Suma', 'Mike', 'G']}
df_2 = pd.DataFrame(data, columns = ['emp_id', 'first_name', 'last_name'])


print("____ Usingconcat________")
df = pd.concat([df_1, df_2])
print(df)
# or
print("-------OR Using append-------------")# Using append
#print (df_1.append(df_2))

print("--- Join the two dataframes along columns--")
print(pd.concat([df_1, df_2], axis=1))

print("---Merge two dataframes based on the emp_id value---")
# in this case only the emp_id's present in both table will be joined
pd.merge(df_1, df_2, on='emp_id')