import nltk
from nltk import chunk

tagged_sent = nltk.pos_tag(nltk.word_tokenize('This is a sample English sentence'))
print (tagged_sent)

from nltk.tag.perceptron import PerceptronTagger
PT = PerceptronTagger()
print (PT.tag('This is a sample English sentence'.split()))
nltk.help.upenn_tagset('VBZ')
