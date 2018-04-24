#Tomas Mikolov’s lead team at Google created Word2Vec (word to vector) model in 2013,
#which uses documents to train a neural network model to maximize the conditional
#probability of a context given the word.
#It utilizes two models: CBOW and skip-gram.
#1. Continuous bag-of-words (CBOW) model predicts the current
#word from a window of surrounding context words or given a
#set of context words predict the missing word that is likely to
#appear in that context. CBOW is faster than skip-gram to train
#and gives better accuracy for frequently appearing words.
#2. Continuous skip-gram model predicts the surrounding
#window of context words using the current word or given a
#single word, predict the probability of other words that are
#likely to appear near it in that context. Skip-gram is known to
#show good results for both frequent and rare words


#You can download Google’s pre-trained model (from given link below) for word2vec, which includes a vocabulary of 3 million words/phrases taken from 100 billion words from a Google News dataset
#URL: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.Word2Vec.load_word2vec_format('Data/GoogleNews-vectors-negative300.bin', binary=True)

model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)

model.most_similar(['girl', 'father'], ['boy'], topn=3)


model.doesnt_match("breakfast cereal dinner lunch".split())

#Training word2vec model on your own custom data Parameters

#size: The dimensionality of the vectors, note that bigger size values require more training data, but can lead to more accurate models
#window: The maximum distance between the current and predicted word within a sentence
#min_count: Ignore all words with total frequency lower than this
#sg = 0 for CBOW model and 1 for skip-gram model

sentences = [['cigarette','smoking','is','injurious', 'to', 'health'],['cigarette','smoking','causes','cancer'],['cigarette','are','not','to','be','sold','to','kids']]

# train word2vec on the two sentences
model = gensim.models.Word2Vec(sentences, min_count=1, sg=1, window = 3)

model.most_similar(positive=['cigarette', 'smoking'], negative=['kids'], topn=1)
