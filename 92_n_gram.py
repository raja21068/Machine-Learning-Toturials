from nltk.util import ngrams

import nltk

from collections import Counter
# Function to extract n-grams from text
def get_ngrams(text, n):
    n_grams = ngrams(nltk.word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]


# Function to remove punctuations
def remove_punctuations(text):
    words = nltk.word_tokenize(text)
    punt_removed = [w for w in words if w.lower() not in string.punctuation]
    return " ".join(punt_removed)


# remove punctuations
text = remove_punctuations(text)
# Extracting bigrams
result = get_ngrams(text,2)
# Counting bigrams
result_count = Counter(result)
# Converting to the result to a data frame
import pandas as pd
df = pd.DataFrame.from_dict(result_count, orient='index')
df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index and column name
print (df)


