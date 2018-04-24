import pandas as pd


df = pd.DataFrame({'A': ['high', 'medium', 'low'],
                   'B': [10, 20, 30]},
                  index=[0, 1, 2])

# using pandas package's factorize function
df['A_pd_factorized'] = pd.factorize(df['A'])[0]

# Alternatively you can use sklearn package's LabelEncoder function
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df['A_LabelEncoded'] = le.fit_transform(df.A)
print (df)

