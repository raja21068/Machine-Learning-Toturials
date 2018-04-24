#Most of Machine Learning libraries are desigend to work well with number of vasribales.
# so categorial varibales in their original form of text description can not be directiely used for model bulding

import pandas as pd
from patsy import dmatrices

df = pd.DataFrame({'A': ['high', 'medium', 'low'],
                   'B': [10, 20, 30]},
                  index=[0, 1, 2])


print(df)

# using get_dummies function of pandas package
#to create a dummy variable for a given categorical variable
df_with_dummies= pd.get_dummies(df, prefix='A', columns=['A'])
print (df_with_dummies)