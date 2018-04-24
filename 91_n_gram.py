from nltk.util import ngrams

import nltk

from collections import Counter
# Function to extract n-grams from text
def get_ngrams(text, n):
    n_grams = ngrams(nltk.word_tokenize(text), n)
    return [ ' '.join(grams) for grams in n_grams]

text = 'This is a sample English sentence'
print ("1-gram: ", get_ngrams(text, 1))
print ("2-gram: ", get_ngrams(text, 2))
print ("3-gram: ", get_ngrams(text, 3))
print ("4-gram: ", get_ngrams(text, 4))


text = 'Statistics skills, and programming skills are equally important for analytics. Statistics skills, and domain knowledge are important for analytics'

# remove punctuations
#text = remove_punctuations(text)

# Extracting bigrams
result = get_ngrams(text,2)

# Counting bigrams
result_count = Counter(result)
# Converting to the result to a data frame
import pandas as pd
df = pd.DataFrame.from_dict(result_count, orient='index')
df = df.rename(columns={'index':'words', 0:'frequency'}) # Renaming index and column name
print (df)