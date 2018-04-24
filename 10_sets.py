# Creating an empty set
languages = set()
print (type(languages), languages)
languages = {'Python', 'R', 'SAS', 'Julia'}
print (type(languages), languages)

# set of mixed datatypes
mixed_set = {"Python", (2.7, 3.4)}
print (type(mixed_set), languages)

print (list(languages)[0])
print (list(languages)[0:3])
# initialize a set
languages = {'Python', 'R'}
print(languages)
# add an element
languages.add('SAS')
print(languages)
# add multiple elements
languages.update(['Julia','SPSS'])
print(languages)
# add list and set
languages.update(['Java','C'], {'Machine Learning','Data Science','AI'})
print(languages)

# remove an element
languages.remove('AI')
print(languages)
# discard an element, although AI has already been removed discard will not
#throw an error
languages.discard('AI')
print(languages)
# Pop will remove a random item from set
print ("Removed:", (languages.pop()), "from", languages)