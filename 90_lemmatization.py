#It is the process of transforming to the dictionary base form. For this you can use
#WordNet, which is a large lexical database for English words that are linked together by
#their semantic relationships. It works as a thesaurus, that is, it groups words together
#based on their meanings

from nltk.corpus import wordnet


syns = wordnet.synsets("good")
print ("Definition: ", syns[0].definition())
print ("Example: ", syns[0].examples())
synonyms = []
antonyms = []
# Print synonums and antonyms (having opposite meaning words)
for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print ("synonyms: \n", set(synonyms))
print ("antonyms: \n", set(antonyms))