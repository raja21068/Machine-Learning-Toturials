import numpy as np

print("1- Field Access")

x = np.zeros((3,3), dtype=[('a', np.int32), ('b', np.float64, (3,3))])
print( "x['a'].shape: ",x['a'].shape)
print( "x['a'].dtype: ", x['a'].dtype)
print( "x['b'].shape: ", x['b'].shape)
print( "x['b'].dtype: ", x['b'].dtype)

print("1- Basic Slicing")
x = np.array([5, 6, 7, 8, 9])
print(x[1:7:2])

print (x[-2:5])
print (x[-1:1:-1])


y = np.array([[[1],[2],[3]], [[4],[5],[6]]])
print ("Shape of y: ", x.shape)
y[1:3]


# Create a rank 2 array with shape (3, 4)
a = np.array([[5,6,7,8], [1,2,3,4], [9,10,11,12]])
print ("Array a:", a)

# Multiple array can be accessed in two ways
row_r1 = a[1,:]# Rank 1 view of the second row of a
row_r2 = a[1:2,:]# Rank 2 view of the second row of a
print (row_r1, row_r1.shape )# Prints "[5 6 7 8] (4,)"
print (row_r2, row_r2.shape )# Prints "[[5 6 7 8]] (1, 4)"

print("Advanced Indexing")

# An example of integer array indexing.
# The returned array will have shape (2,) and
print (a[[0, 1], [0, 1]])
# The above example of integer array indexing is equivalent to this:
print (np.array([a[0, 0], a[1, 1]]))

print("Array Math")
x=np.array([[1,2],[3,4],[5,6]])
y=np.array([7,8],[9,10],[11,12])

# Elementwise sum; both produce the array
print(x+y)
print (np.add(x, y))

