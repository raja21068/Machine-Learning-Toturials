import numpy as np
# create a rank 1 array
a=np.array([0, 1, 2])
print(type(a))

#This will be print dimension of the array0
print(a.shape)
print(a[0])
print(a[1])
print(a[2])

# Create a rank 2 array
b = np.array([[0,1,2],[3,4,5]])
print (b.shape)
print (b)
print (b[0, 0], b[0, 1], b[1, 0])

# Create a 3x3 array of all zeros
c = np.zeros((3,3))
print ("Create a 3x3 array of all zeros")
print (c)


# Create a 3x3 constant array
d = np.full((3,3), 7)
print("Create a 3x3 constant array")
print (d)


# Create a 3x3 array filled with random values
e= np.random.random((3,3))
print("Create a 3x3 array filled with random values")
print (e)

# convert list to array
f = np.array([2, 3, 1, 0])
print("convert list to array")
print (f)

print("note mix of tuple and lists")
h = np.array([[0, 1,2.0],[0,0,0],(1+1j,3.,2.)])
print (h)

print("create an array of range with float data type")
i = np.arange(1, 8, dtype=np.float)
print (i)

print(" linspace() will create arrays with a specified number of items which are")
print("spaced equally between the specified beginning and end values")
j = np.linspace(2., 4., 5)
print (j)

