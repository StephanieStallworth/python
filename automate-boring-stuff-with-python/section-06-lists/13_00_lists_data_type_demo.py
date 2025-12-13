################# Assigning Lists #################
# This is a list value that contains four values
>>> ['cat','bat','rat','elephant']
# ['cat', 'bat', 'rat', 'elephant']

# Assign to a variable just like any other value
>>> spam = ['cat','bat','rat','elephant']

>>> spam
# ['cat', 'bat', 'rat', 'elephant']

################# Indexes #################
# In order to access an item in a list use an integer index for the item's position in the list
# Begins and ends with square brackets

# The first item is at index 0
>>> spam[0]
# 'cat'

>>> spam[2]
# 'rat'

################# List of Lists #################
>>> spam = [['cat','bat'],[10,20,30,40,50]]

# First list
>>> spam[0] 
# ['cat', 'bat']

# Use additional indexes to evaluate to an item inside of the list
>>> spam[0][1] 
# 'bat'

>>> spam[1][4]
# 50

################# Negative Indexes #################
# Indexes start at zero and go up
# Can also use negative integers, these count from the end going backwards

# -1 is the last index in the list
>>> spam[-1]
# [10, 20, 30, 40, 50]

>>> spam = ['cat','bat','rat','elephant']
>>> spam[-1]
# 'elephant'

# -2 is the second to last index and so on
>>> spam[-2]
# 'rat'

# Can use inside of expressions just like any other value
>>> 'The ' + spam[-1] + ' is afraid of the ' + spam[-3] + '.' 
# 'The elephant is afraid of the bat.'

################# Slices #################
# The index gets a single value from a list: Index = Single Value
# A slice can get several values from a list: Slice = List of values

# Typed between square brackets but also has 2 integers separated by a colon for the start and end indexes
>>> spam[1:3]
# ['bat', 'rat']

################# Changing A List's Items #################
# Use the index to assign a new value to an item in a list
>>> spam = [10,20,30]
>>> spam[1] = 'Hello'
>>> spam
# [10, 'Hello', 30]

# Same can be done for multiple values in a list using a slice
>>> spam[1:3] = ['CAT','DOG','MOUSE']
>>> spam
# [10, 'CAT', 'DOG', 'MOUSE']

################# Slice Shortcuts #################
# Can leave out one or both of the indexes on either side of the colon in a slice

# Leaving out the first index is like using index 0 or the beginning of the list
>>> spam = ['cat','bat','rat','elephant']
>>> spam
# ['cat', 'bat', 'rat', 'elephant']

>>> spam[:2]
# ['cat', 'bat']

# Leaving out the second index is the same as using the length of the list, which will slice to the end of the list
# Grab all the values up to the end of the list
>>> spam[1:]
# ['bat', 'rat', 'elephant']

################# del Statements #################
# del Statement = "Unassignment" statement

# Delete item a specific index
>>> spam = ['cat','bat','rat','elephant']
>>> del spam[2]
>>> spam
# ['cat', 'bat', 'elephant']

# All the items after it move up one, doesn't leave any gaps in the list
>>> del spam[2]
>>> spam
# ['cat', 'bat']

################# String and List Similarities #################
# Many of the things you can do with strings, Python also lets you do with lists
# You can think of a string value as a list of single character values

# The len() function on a list returns the number of characters in that string
>>> len('Hello')
# 5

# Can also pass len() a list to return the number of items in a list
>>> len([1,2,3])
# 3

# Just like string concatenation with "+" operator
'Hello ' + 'world'
# 'Hello world'

# Can also do list concatenation with the list the + operator 
>>> [1,2,3] + [4,5,6]
# [1, 2, 3, 4, 5, 6]

# String replication
>>> 'Hello' * 3
# 'HelloHelloHello'

# List replication
>>> [1,2,3] * 3
# [1, 2, 3, 1, 2, 3, 1, 2, 3]

################# the list() Function #################
# Returns a list form of the value that you pass it
# Same as if we need to convert a string to an integer with the int() function

# Converts string to an integer
>>> int('42')
# 42

# Convert integer to a string
>>> str(42) 
# '42'

# Convert string to a list 
>>> list('Hello')
# ['H', 'e', 'l', 'l', 'o']

################# The in and not in Operators #################
# Like other operators, in and not in are used in expressions and connect two values

## The in operator ##
>>> 'howdy' in ['hello','hi','howdy','heyas']
# True

>>> 42 in ['hello','hi','howdy','heyas']
# False

## The not in operator ##
>>> # The not in operator does the exact opposite
>>> 'howdy' not in ['hello','hi','howdy','heyas']
# False