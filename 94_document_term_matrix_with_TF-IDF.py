import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib as plt
# Function to create a dictionary with key as file names and values as text for all files in a given folder
def CorpusFromDir(dir_path):
    result = dict(docs = [open(os.path.join(dir_path,f)).read() for f in
    os.listdir(dir_path)],
    ColNames = map(lambda x: x, os.listdir(dir_path)))
    return result


docs = CorpusFromDir('chapter5/Data/text_files')

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
doc_vec = vectorizer.fit_transform(docs.get('docs'))
#create dataFrame
df = pd.DataFrame(doc_vec.toarray().transpose(), index = vectorizer.get_feature_names())
# Change column headers to be file names
df.columns = docs.get('ColNames')
print (df)


words = df.index
freq = df.ix[:,0].sort(ascending=False, inplace=False)
pos = np.arange(len(words))
width=1.0
ax=plt.axes(frameon=True)
ax.set_xticks(pos)
ax.set_xticklabels(words, rotation='vertical', fontsize=9)
ax.set_title('Word Frequency Chart')
ax.set_xlabel('Words')
ax.set_ylabel('Frequency')
plt.bar(pos, freq, width, color='b')
plt.show()