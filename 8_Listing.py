# Create lists
list_1 = ['Statistics', 'Programming', 2016, 2017, 2018];
list_2 = ['a', 'b', 1, 2, 3, 4, 5, 6, 7 ];
# Accessing values in lists
print ("list_1[0]: ", list_1[0])
print ("list2_[1:5]: ", list_2[1:5])

print ("list_1 values: ", list_1)
# Adding new value to list
list_1.append(2019)
print ("list_1 values post append: ", list_1)


print ("Values of list_1: ", list_1)
# Updating existing value of list
print ("Index 2 value : ", list_1[2])
list_1[2] = 2015;
print ("Index 2's new value : ", list_1[2])


print ("list_1 values: ", list_1)
# Deleting list element
del list_1[5];
print ("After deleting value at index 2 : ", list_1)


# Listing 1-24. Example code for basic operations on lists
print ("Length: ", len(list_1))
print ("Concatenation: ", [1,2,3] + [4, 5, 6])
print ("Repetition :", ['Hello'] * 4)
print ("Membership :", 3 in [1,2,3])
print ("Iteration :")
for x in [1,2,3]:
    print (x)
# Negative sign will count from the right
print ("slicing :", list_1[-2])
# If you dont specify the end explicitly, all elements from the specified
#@start index will be printed
print ("slicing range: ", list_1[1:])
# Comparing elements of lists
#print ("Compare two lists: ", cmp([1,2,3, 4], [1,2,3]))
print ("Max of list: ", max([1,2,3,4,5]))
print ("Min of list: ", min([1,2,3,4,5]))
print ("Count number of 1 in list: ", [1,1,2,3,4,5,].count(1))
list_1.extend(list_2)
print ("Extended :", list_1)
print ("Index for Programming : ", list_1.index( 'Programming'))
print (list_1)
print ("pop last item in list: ", list_1.pop())
print ("pop the item with index 2: ", list_1.pop(2))
list_1.remove('b')
print ("removed b from list: ", list_1)
list_1.reverse()
print ("Reverse: ", list_1)
list_1 = ['a', 'b','c',1,2,3]
list_1.sort()
print ("Sort ascending: ", list_1)
list_1.sort(reverse = True)
print ("Sort descending: ", list_1)


