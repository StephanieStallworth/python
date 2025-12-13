######## Range Objects and List-Like Values ###############
# Use `for` loops to execute a block of code a certain number of times
# Technically, a `for` loop repeats the code block once for each value in a list or list like value

>>> for i in range(4):
	print(i)

# 0
# 1
# 2
# 3

######## The range() function #######
# The range() function returns a value that is of the Range data type 
# Range objects are list-like values
# Use the term "list like" for date types that are technically named sequences in Python

# Python considers this range object 
>>> range(4)
# range(0, 4)

# To  be similar to the list [0,1,2,3]
# This do the exact same thing as previously

>>> for i in [0,1,2,3]:
	print(i)

# 0
# 1
# 2
# 3

###### The list() + range() Function #######
# Pass a range object the list() function to get the actual list values from that range object

# Returns an actual list for you
>>> list(range(4))
# [0, 1, 2, 3]

# Handy if you need to get a collection of integers into a list
# Instead of typing out the numbers manually into a list, just pass the range object
>>> list(range(0,100,2))
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]

# Can assign to a variable just like any other value
>>> spam = list(range(0,100,2))
>>> spam
# [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98]

######## for i in range(len(someList)): #############
# Can use i to refer to the integer index and also use index in brackets to get the value inside the list
>>> supplies = ['pens','staplers','flame-throwers','binders']
>>> for i in range(len(supplies)):
	print('Index' + str(i) + ' in supplies is: ' + supplies[i]) 

# Index0 in supplies is: pens
# Index1 in supplies is: staplers
# Index2 in supplies is: flame-throwers
# Index3 in supplies is: binders

# The list can be of any size and the same code will work
>>> supplies = ['pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens','pens']
>>> for i in range(len(supplies)):
	print('Index ' + str(i) + ' in supplies is: ' + supplies[i])

# Index 0 in supplies is: pens
# Index 1 in supplies is: pens
# Index 2 in supplies is: pens
# Index 3 in supplies is: pens
# Index 4 in supplies is: pens
# Index 5 in supplies is: pens
# Index 6 in supplies is: pens
# Index 7 in supplies is: pens
# Index 8 in supplies is: pens
# Index 9 in supplies is: pens
# Index 10 in supplies is: pens
# Index 11 in supplies is: pens
# Index 12 in supplies is: pens
# Index 13 in supplies is: pens
# Index 14 in supplies is: pens
# Index 15 in supplies is: pens
# Index 16 in supplies is: pens

############## Multiple Assignment ##############
# Instead of individually assigning items from a list to different variables
>>> cat = ['fat','orange','loud']
>>> size = cat[0]
>>> color = cat[1]
>>> disposition = cat[2]

# Can have multiple variables on the left side of the assignment operator seperated by commas
# Then the list value on the right side of the assignment operator 
# Will automatically do the same thing with one line of code
>>> size, color, disposition = cat

# 1st item of the list cat assigned to the first variable on the left left side of the assignment operator 
>>> size
# 'fat'

# 2nd value of the list to 2nd variable
>>> color
# 'orange'

# 3rd value of the list to the 3rd variable
>>> disposition
# 'loud'

# Can have multiple variables on the left side 
# AND multiple values on the right side, just separate those with commas
>>> size, color, disposition = 'skinny','black','quiet'

>>> size
# 'skinny'

>>> color
# 'black'

>>> disposition
# 'quiet'

############## Swapping Variables ##############
# Multiple assignment is often used for swapping variables
>>> a = 'AAA'
>>> b = 'BBB'
>>> a, b = b, a

>>> a
# 'BBB'

>>> b
# 'AAA'

############## Augmented Assignment Operators ##############
# When assigning a value to a variable, you'll frequently use the variable itself
# Use Augmented Assignment Operators as a shortcut so you don't have to retype the variable name
# There are augmented assignment operators for plus, minus, multiplication, division and the modulus operators

# Increment the value inside of a variable
>>> spam = 42
>>> spam = spam + 1
>>> spam += 1
>>> spam
# 44