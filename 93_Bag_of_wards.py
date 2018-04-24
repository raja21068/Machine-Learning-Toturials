import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# Function to create a dictionary with key as file names and values as text for all files in a given folder
def CorpusFromDir(dir_path):
    result = dict(docs = [open(os.path.join(dir_path,f)).read() for f in
    os.listdir(dir_path)],
    ColNames = map(lambda x: x, os.listdir(dir_path)))
    return result


docs = CorpusFromDir('chapter5/Data/text_files')

# Initialize
vectorizer = CountVectorizer()
doc_vec = vectorizer.fit_transform(docs.get('docs'))

#create dataFrame
df = pd.DataFrame(doc_vec.toarray().transpose(), index = vectorizer.get_feature_names())

# Change column headers to be file names
df.columns = docs.get('ColNames')
print (df)

