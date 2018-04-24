# First Example
print ("First Example")
for item in [1,2,3,4,5]:
    print ('item :', item)
# Second Example
print ("Second Example")
letters = ['A', 'B', 'C']
for letter in letters:
    print (' First loop letter :', letter)
# Third Example - Iterating by sequency index
print ("Third Example")
for index in range(len(letters)):
    print ('First loop letter :', letters[index])
# Fourth Example - Using else statement
print ("Fourth Example")
for item in [1,2,3,4,5]:
    print ('item :', item)
else:
    print ('looping over item complete!')