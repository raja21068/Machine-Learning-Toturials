import nltk
from nltk.corpus import stopwords
def remove_whitespace(text):
    return " ".join(text.split())

text = 'This is a       sample English sentence, \n with whitespace and numbers 1234!'
print ('Removed whitespace: ', remove_whitespace(text))