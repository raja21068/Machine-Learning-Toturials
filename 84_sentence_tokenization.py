import nltk
nltk.download()
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
text='Statistics skills, and programming skills are equally important for analytics. Statistics skills, and domain knowledge are important for analytics. I like reading books and travelling.'

# sent_tokenize uses an instance of PunktSentenceTokenizer from the nltk. tokenize.punkt module. This instance has already been trained on and works well for many European languages. So it knows what punctuation and characters mark the end of a sentence and the beginning of a new sentence.
sent_tokenize_list = sent_tokenize(text)
print (word_tokenize(text))
print(sent_tokenize_list)



# There are total 17 european languages that NLTK support for sentence tokenize
# Let's try loading a spanish model
import nltk.data
spanish_tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
spanish_tokenizer.tokenize('Hola. Esta es una frase espanola.')