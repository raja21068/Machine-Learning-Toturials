#LSA is a mathematical method that tries to bring out latent relationships within
#a collection of documents. Rather than looking at each document isolated from
#the others, it looks at all the documents as a whole and the terms within them to
#identify relationships. Let’s perform LSA by running SVD on the data to reduce the
#dimensionality

#SVD of matrix A = U * Σ * VT
#r = rank of matrix X
#U = column orthonormal m * r matrix
#Σ = diagonal r * r matrix with singular value sorted in descending order
#V = column orthonormal r * n matrix

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

from IPython.display import Image
Image(filename='../Chapter 5 Figures/SVD.png', width=500)
from sklearn.decomposition import TruncatedSVD

# Lets reduce the dimensionality to 2000
svd = TruncatedSVD(2000)
lsa = make_pipeline(svd, Normalizer(copy=False))

X = lsa.fit_transform(X)

explained_variance = svd.explained_variance_ratio_.sum()
print("Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))