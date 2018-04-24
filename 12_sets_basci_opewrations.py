
# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
print (A.isdisjoint(B)) # True, when two sets have a null intersection
print (A.issubset(B)) # True, when another set contains this set
print (A.issuperset(B)) # True, when this set contains another set
sorted(B) # Return a new sorted list
print (sum(A) )# Retrun the sum of all items
print (len(A)) # Return the length
print (min(A) )# Return the largestitem
print (max(A)) # Return the smallest item