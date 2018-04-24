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


# Left join
print (pd.merge(df_1, df_2, on='emp_id', how='left'))
# Merge while adding a suffix to duplicate column names of both table
print (pd.merge(df_1, df_2, on='emp_id', how='left', suffixes=('_left', '_right')))

