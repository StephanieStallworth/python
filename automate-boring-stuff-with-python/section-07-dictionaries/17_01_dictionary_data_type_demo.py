# Create a dictionary with curly brackets 
>>> myCat = {'size': 'fat','color':'gray','disposition':'loud'}

# Can access these values through their keys
>>> myCat['size']
# 'fat'

# Can use in an expression
>>> 'My cat has ' + myCat['color'] + ' fur.'
# 'My cat has gray fur.'

# Dictionaries can still use integer values as keys, just like lists use integers for indexes
# But they don't have to start at zero and they can be any number
>>> spam = {12345:'Luggage combination', 42:'The Answer'}

# While the order of items matters for determining whether two LISTS are the same
>>> [1, 2, 3] == [3, 2, 1]
# False

# For dictionaries, that is not the case
>>> eggs = {'name':'Zophie', 'species': 'cat', 'age': 8}
>>> ham = {'species':'cat','name':'Zophie','age':8}
>>> eggs == ham
# True

# Trying to access a key that doesn't exist in a dictionary will result in a `KeyError` error message
>>> eggs['color']
# Traceback (most recent call last):
#   File "<pyshell#18>", line 1, in <module>
#     eggs['color']
# KeyError: 'color'

# Can check if a key exists in a dictionary value with the `in` and `not in` operators
# `in` operator 
>>> 'name' in eggs
# True

>>> eggs
# {'name': 'Zophie', 'species': 'cat', 'age': 8}

# `not in` operator  
>>> 'name' not in eggs
# False 

########## The keys(), values(), and items() Dictionary Methods ##########
# keys() dictionary method 
>>> list(eggs.keys())
# ['name', 'species', 'age']

# values() dictionary method 
>>> list(eggs.values())
# ['Zophie', 'cat', 8]

# items() dictionary method
# Returns a list of two-item tuples (with key as the first item in the tuple, value as the second item)
>>> list(eggs.items())
# [('name', 'Zophie'), ('species', 'cat'), ('age', 8)]

# These methods return "list like" data types
# If you want an actual list values, have to pass it to the list() function like above
>>> eggs.keys()
# dict_keys(['name', 'species', 'age'])

# Can use these methods in `for` loops
>>> for k in eggs.keys():
	print(k)

# name
# species
# age

>>> for v in eggs.values():
	print(v)

# Zophie
# cat
# 8

# Can use the multiple assignment trick and have multiple variables in the `for` loop for the items
>>> for k, v in eggs.items():
	print(k,v)

# name Zophie
# species cat
# age 8

# Without multiple assignment, it would print the tuples themselves
>>> for i in eggs.items():
	print(i)

# ('name', 'Zophie')
# ('species', 'cat')
# ('age', 8)

# Can also use the `in` and `not in` operators to see whether a certain key or value exists in a dictionary
# `in` operator 
>>> eggs
# {'name': 'Zophie', 'species': 'cat', 'age': 8}

>>> 'cat' in eggs.values()
# True

########## The get() Dictionary Method ##########
# Want to avoid the `KeyError` error message that will crash your program
>>> eggs['color']
# Traceback (most recent call last):
#   File "<pyshell#79>", line 1, in <module>
#     eggs['color'] 
# KeyError: 'color'

# Would have to do an `if` statement, but would be tedious to do every time
>>> if 'color' in eggs:
	print(eggs['color'])

# Better to use the get() method instead
# If key exists returns the value
>>> eggs.get('age',0)
# 8

# If key doesn't exist, defaults to 2nd argument passed
>>> eggs.get('color','')
# ''

>>> # Handy if you have some dictionary that is keeping track of how many things you're bringing to a picnic
>>> picnicItems = {'apples':5, 'cups': 2}

# With the get() method, there is no error if key doesn't exist  
>>> print('I am bringing ' + str(picnicItems.get('napkins',0)) + ' to the picnic.') 
# I am bringing 0 to the picnic.

# Normal square bracket syntax, get an error if key doesn't exist
>>> print('I am bringing ' + str(picnicItems['napkins']) + ' to the picnic.')
# Traceback (most recent call last):
#   File "<pyshell#105>", line 1, in <module>
#     print('I am bringing ' + str(picnicItems['napkins']) + ' to the picnic.')
# KeyError: 'napkins'

########## The setdefault() Dictionary Method ##########
# Set values in a dictionary only if that key doesn't already have a value
>>> eggs
# {'name': 'Zophie', 'species': 'cat', 'age': 8}

# Set `color` key to value `black` if the color key doesn't already have a value in the dictionary
>>> if 'color' not in eggs:
	eggs['color'] = 'black' 
	
# The setdefault() is a way to do this in one line of code
>>> eggs = {'name':'Zophie', 'species': 'cat', 'age': 8}
>>> eggs.setdefault('color','black')
# 'black'

# Adds key value pair `color` with string `black`
>>> eggs
# {'name': 'Zophie', 'species': 'cat', 'age': 8, 'color': 'black'}

# If we try to set it to something different, doesn't change anything 
#  Because the `color` key already exists and has the setting `black`
>>> eggs.setdefault('color','orange')
# 'black'

>>> eggs
# {'name': 'Zophie', 'species': 'cat', 'age': 8, 'color': 'black'}