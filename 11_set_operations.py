#Example code for set union operation
# initialize A and B
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
# use | operator
print ("Union of A | B", A|B)
# alternative we can use union()
A.union(B)


#Set Intersection
# use & operator
print ("Intersection of A & B", A & B)
# alternative we can use intersection()
print (A.intersection(B))

#Set Difference
# use - operator on A
print ("Difference of A - B", A - B)
# alternative we can use difference()
print (A.difference(B))

#Set Symmetric Difference
# use ^ operator
print ("Symmetric difference of A ^ B", A ^ B)
# alternative we can use symmetric_difference()
A.symmetric_difference(B)
