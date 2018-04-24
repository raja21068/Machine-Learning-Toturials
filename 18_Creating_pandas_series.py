import pandas as pd
import numpy as np
# creating a series by passing a list of values, and a custom index label.
#Note that the labeled index reference for each row and it can have duplicate values

print("Creating a pandas series")
s = pd.Series([1,2,3,np.nan,5,6,7], index=['A','B','C','D','E','F','G'])
print (s)

print("Creating a pandas dataframe")
data = {'Gender': ['F', 'M', 'M'],'Emp_ID': ['E01', 'E02','E03'], 'Age': [25, 27, 25]}
# We want the order the columns, so lets specify in columns parameter
df = pd.DataFrame(data, columns=['Emp_ID','Gender', 'Age'])
print(df)

print("Reading and Writing Data")
# Reading
df=pd.read_csv('chapter2/mtcars.csv') # from csv
df=pd.read_csv('chapter2/mtcars.txt', sep='\t') # from text file
df=pd.read_excel('chapter2/mtcars.xlsx','Sheet2') # from Excel

print("reading from multiple sheets of same Excel into different dataframes")
xlsx = pd.ExcelFile('chapter2/file_name.xls')
sheet1_df = pd.read_excel(xlsx, 'Sheet1')
sheet2_df = pd.read_excel(xlsx, 'Sheet2')

print(" writing")
# index = False parameter will not write the index values, default is True
df.to_csv('chapter2/mtcars_new.csv', index=False)
df.to_csv('chapter2/mtcars_new.txt', sep='\t', index=False)
df.to_excel('chapter2/mtcars_new.xlsx',sheet_name='Sheet1', index = False)
