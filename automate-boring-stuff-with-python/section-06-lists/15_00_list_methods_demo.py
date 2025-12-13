############ The index() List Method ############
>>> spam = ['hello','hi','howdy','heyas']

# All list values have a method called index(), 
# Call that method on the list value in spam
 
# Returns the index where Python finds the value that you pass it
>>> spam.index('hello')
# 0

# Have to call it on a value, Python doesn't know what list you're referring to
# Can't just run the method by itself 
>>> index('hello')
# Traceback (most recent call last):
#   File "<pyshell#7>", line 1, in <module>
#     index('hello')
# NameError: name 'index' is not defined

# Returns the index if value exists in the list 
>>> spam.index('heyas')
# 3

# Raises an exception if value doesn't exist inside of list
>>> spam.index('a;sdlfjad;fj') 
# Traceback (most recent call last):
#   File "<pyshell#12>", line 1, in <module>
#     spam.index('a;sdlfjad;fj') # Doesn't exist inside of list
# ValueError: 'a;sdlfjad;fj' is not in list

# If value appears multiple times, it returns the index of the first time it sees it in the list
>>> spam = ['Zophie','Pooka','Fat-tail','Pooka']
>>> spam.index('Pooka')
# 1

############ The apppend() and insert List Methods ############
# The append() list method adds a value to the end of the a list
>>> spam = ['cat','dog','bat']
>>> spam.append('moose')
>>> spam
# ['cat', 'dog', 'bat', 'moose']

# The insert() method can insert value at any point inside the list
>>> spam = ['cat','dog','bat']
>>> spam.insert(1,'chicken')

# It will insert value at the specified in the list, everything else gets bumped up
>>> spam
# ['cat', 'chicken', 'dog', 'bat']

# Don't assign the return value of append() and insert() you just call the method itself
# These methods just return the None value
# So we type `spam.append('moose')`
# Don't type `spam = spam.append('moose')`
# This would assign the None value to `spam` and gets rid of that list entirely
# The list is modified "in place"
>>> spam.append('moose')
>>> spam.append('moose')
>>> spam.append('moose')
>>> spam.append('moose')
>>> spam
# ['cat', 'chicken', 'dog', 'bat', 'moose', 'moose', 'moose', 'moose']

# Methods belong to a single data type
# The append() and insert() methods are list methods and can only be called on list values
# Can't be called on strings and integers 
>>> eggs = 'hello'
>>> eggs.append('world')
# Traceback (most recent call last):
#   File "<pyshell#65>", line 1, in <module>
#     eggs.append('world')
# AttributeError: 'str' object has no attribute 'append'

############ The remove() List Method ############
# Pass a value you want removed from the list its called on
# Remove allows you to specify a value you want to remove, no matter WHERE it is in the list
>>> spam = ['cat','bat','rat','elephant']
>>> spam.remove('bat')
>>> spam
# ['cat', 'rat', 'elephant']

# Error if try to remove a value that doesn't exist in the list
>>> spam.remove('bat')
# Traceback (most recent call last):
#   File "<pyshell#75>", line 1, in <module>
#     spam.remove('bat')
# ValueError: list.remove(x): x not in list

# Different from the `del` statement 
# `del` will delete the value at an index, no matter what it is 
>>> del spam[0] 
>>> spam
# ['rat', 'elephant']

# Only the first instance of value will be removed
>>> spam = ['cat','bat','rat','cat','hat','cat']
>>> spam.remove('cat')

# Removes only the first cat variable it found
>>> spam
# ['bat', 'rat', 'cat', 'hat', 'cat']

############ The sort() List Method ############
# Lists with number of string values can be sorted with the sort() method

# List with numbers 
>>> spam = [2,5,3.14, 1, -7]
>>> spam.sort()
>>> spam
# [-7, 1, 2, 3.14, 5]

# List with strings
>>> spam = ['ants', 'cats', 'dogs','badgers','elephants']
>>> spam.sort()
>>> spam
# ['ants', 'badgers', 'cats', 'dogs', 'elephants']

# To sort in reverse, pass in the `reverse` keyword argument that takes a Boolean value
>>> spam.sort(reverse = True)
>>> spam
# ['elephants', 'dogs', 'cats', 'badgers', 'ants']

# Can't sort lists that have BOTH number and string values, Python doesn't know how to compare these values
>>> spam = [1,2,3,'Alice','Bob']
>>> spam.sort()
# Traceback (most recent call last):
#   File "<pyshell#115>", line 1, in <module>
#     spam.sort()
# TypeError: '<' not supported between instances of 'str' and 'int'

# Technically sort() doesn't use alphabetical order
# Sorts in ASCII-betical order
# Upper case characters come before lowercase characters
>>> spam = ['Alice','Bob','ants', 'badgers','Carol','cats']
>>> spam.sort()
>>> spam
# ['Alice', 'Bob', 'Carol', 'ants', 'badgers', 'cats']

>>> spam = ['a','z','A','Z']
>>> spam.sort()
>>> spam
# ['A', 'Z', 'a', 'z']

# To sort in true alphabetical order
# Pass the convert to lowercase string method
>>> spam.sort(key = str.lower)
>>> spam
# ['A', 'a', 'Z', 'z']

# Works with the uppercase string method also 
>>> spam.sort(key = str.upper)
>>> spam
# ['A', 'a', 'Z', 'z']