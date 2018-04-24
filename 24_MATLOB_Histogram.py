import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib
#matplotlib inline

# Hide all warning messages
# You can also create the histogram, line graph and the boxplot directly on a dataframe
df = pd.read_csv('chapter2/iris.csv')
print("Histogram")
df.hist()
plt.show()
df.plot() # Line Graph
plt.show()
df.boxplot() # Box plot
plt.show()