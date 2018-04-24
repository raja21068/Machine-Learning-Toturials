import statsmodels.api as sm
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import feature_extraction
import os

def CorpusFromDir(dir_path):
    result = dict(docs = [open(os.path.join(dir_path,f)).read() for f in
    os.listdir(dir_path)],
    ColNames = map(lambda x: x, os.listdir(dir_path)))
    return result


docs = CorpusFromDir('chapter5/Data/text_files')
# default unigram model
count_model = feature_extraction.text.CountVectorizer(ngram_range=(1,1))
docs_unigram = count_model.fit_transform(docs.get('docs'))

# co-occurrence matrix in sparse csr format
docs_unigram_matrix = (docs_unigram.T * docs_unigram)

# fill same word cooccurence to 0
docs_unigram_matrix.setdiag(0)

# co-occurrence matrix in sparse csr format
docs_unigram_matrix = (docs_unigram.T * docs_unigram)
docs_unigram_matrix_diags = sp.diags(1./docs_unigram_matrix.diagonal())

# normalized co-occurence matrix
docs_unigram_matrix_norm = docs_unigram_matrix_diags * docs_unigram_matrix

df = pd.DataFrame(docs_unigram_matrix_norm.todense(), index = count_model.get_feature_names())
df.columns = count_model.get_feature_names()

sm.graphics.plot_corr(df, title='Co-occurrence Matrix', xnames=list(df.index))
plt.show()