dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
print ("Sample dictionary: ", dict)

# Accessing items in dictionary
print ("Value of key Name, from sample dictionary:", dict['Name'])

# Deleting a dictionary
dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
print ("Sample dictionary: ", dict)
del dict['Name'] # Delete specific item
print ("Sample dictionary post deletion of item Name:", dict)
dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
dict.clear() # Clear all the contents of dictionary
print ("dict post dict.clear():", dict)
dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
del dict # Delete the dictionary


# Updating dictionary
dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
print ("Sample dictionary: ", dict)
dict['Age'] = 6.5
print ("Dictionary post age value update: ", dict)


dict = {'Name': 'Jivin', 'Age': 6, 'Class': 'First'}
print ("Length of dict: ", len(dict))
dict1 = {'Name': 'Jivin', 'Age': 6};
dict2 = {'Name': 'Pratham', 'Age': 7};
dict3 = {'Name': 'Pranuth', 'Age': 7};
dict4 = {'Name': 'Jivin', 'Age': 6};
#print ("Return Value: dict1 vs dict2", cmp (dict1, dict2))
#print ("Return Value: dict2 vs dict3", cmp (dict2, dict3))
#print ("Return Value: dict1 vs dict4", cmp (dict1, dict4))

# String representation of dictionary
dict = {'Name': 'Jivin', 'Age': 6}
print ("Equivalent String: ", str (dict))
# Copy the dict
dict1 = dict.copy()
print (dict1)
# Create new dictionary with keys from tuple and values to set value
seq = ('name', 'age', 'sex')
dict = dict.fromkeys(seq)
print ("New Dictionary: ", str(dict))
dict = dict.fromkeys(seq, 10)
print ("New Dictionary: ", str(dict))
# Retrieve value for a given key
dict = {'Name': 'Jivin', 'Age': 6};
print ("Value for Age: ", dict.get('Age'))
# Since the key Education does not exist, the second argument will be

print ("Value for Education: ", dict.get('Education', "First Grade"))
# Check if key in dictionary
print ("Age exists? ", dict.has_key('Age'))
print ("Sex exists? ", dict.has_key('Sex'))
# Return items of dictionary
print ("dict items: ", dict.items())
# Return items of keys
print ("dict keys: ", dict.keys())
# return values of dict
print ("Value of dict: ", dict.values())
# if key does not exists, then the arguments will be added to dict and
print ("Value for Age : ", dict.setdefault('Age', None))
print ("Value for Sex: ", dict.setdefault('Sex', None))
# Concatenate dicts
dict = {'Name': 'Jivin', 'Age': 6}
dict2 = {'Sex': 'male' }
dict.update(dict2)
print ("dict.update(dict2) = ", dict)