from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#The correlation function uses Pearson correlation coefficient, which results in a number
#between -1 to 1. A strong negative relationship is indicated by a coefficient closer to -1
#and a strong positive correlation is indicated by a coefficient toward 1.

iris = datasets.load_iris()
# Let's convert to dataframe
iris = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
columns= iris['feature_names'] + ['species'])

# create correlation matrix
corr = iris.corr()
print(corr)

import statsmodels.api as sm
sm.graphics.plot_corr(corr, xnames=list(corr.columns))
plt.show()


#You can understand the relationship attributes by looking at the distribution of the interactions of each pair of attributes.
from pandas.tools.plotting import scatter_matrix
scatter_matrix(iris, figsize=(10, 10))
# use suptitle to add title to all sublots
plt.suptitle("Pair Plot", fontsize=20)
