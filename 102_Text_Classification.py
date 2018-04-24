# The ability of representing text features as numbers opens up the opportunity to run
#classification machine learning algorithms. Let’s use a subset of 20 newsgroups data to
#build a classification model and assess its accuracy

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np

newsgroups_train = fetch_20newsgroups(subset='train')
print(list(newsgroups_train.target_names))

newsgroups_test = fetch_20newsgroups(subset='train')


categories = ['alt.atheism', 'comp.graphics', 'rec.motorcycles', 'sci.space', 'talk.politics.guns']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                      shuffle=True, random_state=2017, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                     shuffle=True, random_state=2017, remove=('headers', 'footers', 'quotes'))

y_train = newsgroups_train.target
y_test = newsgroups_test.target

vectorizer = TfidfVectorizer(sublinear_tf=True, smooth_idf = True, max_df=0.5,  ngram_range=(1, 2), stop_words='english')
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

print("Train Dataset")
print("%d documents" % len(newsgroups_train.data))
print("%d categories" % len(newsgroups_train.target_names))
print("n_samples: %d, n_features: %d" % X_train.shape)

print("Test Dataset")
print("%d documents" % len(newsgroups_test.data))
print("%d categories" % len(newsgroups_test.target_names))
print("n_samples: %d, n_features: %d" % X_test.shape)

