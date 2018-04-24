import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib
#matplotlib inline

# Hide all warning messages
import warnings
warnings.filterwarnings('ignore')

#plt.bar – creates a bar chart
#plt.scatter – makes a scatter plot
#plt.boxplot – makes a box and whisker plot
#plt.hist – makes a histogram
#plt.plot – creates a line plot


# simple bar and scatter plot
x = np.arange(5) # assume there are 5 students
y = (20, 35, 30, 35, 27) # their test scores
plt.bar(x,y) # Bar plot
# need to close the figure using show() or close(), if not closed any follow
#up plot commands will use same figure.
plt.show() # Try commenting this an run
plt.scatter(x,y) # scatter plot
plt.show()


