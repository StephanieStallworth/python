# Strings and lists are similar if you consider a string to be a list of single character strings
# Pass string to list function to convert it to a list
>>> list('Hello')
# ['H', 'e', 'l', 'l', 'o']

# Many of the things you can do with lists, you can do with strings
>>> name = 'Zophie'
>>> name[0]
# 'Z'

>>> name[1:3]
# 'op'

>>> name[-2]
# 'i'

>>> 'Zo' in name
# True

>>> 'xxx' in name
# False

>>> for letter in name:
		print(letter)
# Z
# o
# p
# h
# i
# e

############ Mutable and Immutable Data Types ################
# A string value is immutable it cannot be changed
name = 'Zophie the cat'

# Can use indexing to access letter from that string
>>> name[7]
# 't'

# But can NOT reassign letters in that string
>>> name[7] = 'X'
# Traceback (most recent call last):
#   File "<pyshell#20>", line 1, in <module>
#     name[7] = 'X'
# TypeError: 'str' object does not support item assignment

############ Creating New Strings From Slices #################
>>> name = 'Zophie a cat'
>>> # Would need to create a new string and use slices to pick out parts of the old string I want and parts after that I do want
>>> newName = name[0:7] + 'the' + name[8:12]
>>> newName
# 'Zophie the cat'

############ References #################
# The difference between immutable and mutable comes up with "references" explained next

# Variables store string and integer values
# Whatever value this expression evaluates to is the value that gets copied into the `cheese` variable
>>> spam = 42
>>> cheese = spam

# Update the `spam` variable 
>>> spam = 100
>>> spam
# 100

# `cheese` variable is still 42
# Because `cheese` was assigned when the `spam` variable was 42
>>> cheese
# 42

# But lists don't work this way
# Python will create this list in its computer memory and also assigned a reference to this list
>>> spam = [0,1,2,3,4,5]

# When you assign a list to a variable, you're actually assigning a list REFERENCE to the variable
#  This expression evaluates to a REFERENCE that gets copied to `cheese`
>>> cheese = spam

# So when you update the `cheese` variable 
>>> cheese[1] = 'Hello!'
>>> cheese
# [0, 'Hello!', 2, 3, 4, 5]

# Only modified the `cheese` variable, but spam was also modified
# Even though we have two separate references, they're referencing the same list
# So when you modify list that's referenced to by `cheese` you're also modifying list that is referred to by `spam` (because they're the same list)
>>> spam
# [0, 'Hello!', 2, 3, 4, 5]

############ copy.deepcopy() #################
>>> import copy

# Original list 
>>> spam = ['A', 'B', 'C', 'D']

# Want to copy list (and make a brand new list), not just copy a list reference
# Call the `copy.deepcopy()` function and pass `spam` to it
# This creates a brand new list based on values of `spam` list passed to it
# And returns a reference to the new list
>>> cheese = copy.deepcopy(spam)

# So now we can make all the changes we want to `cheese` list
# Because it's actually a separate list to the one in `spam`
>>> cheese[1] = 42
>>> cheese
# ['A', 42, 'C', 'D']

>>> # Modifying cheese doesn't modify `spam`
>>> spam
# ['A', 'B', 'C', 'D']

############ Line Continuation ############
# Python considers this to be one line of code
spam = ['apples',
		'oranges',
		'bananas',
		'cats']

# Can do this even when you don't have lists by using the backslash `\` line continuation character 
print('Four score and seven ' + \ # Tells Python to ignore indentation on the next line, just continuing this previous line (not starting/ending a block)
	  'years ago') 

# Same output 
print('Four score and seven ' + 'years ago')