####### Dictionary Data Structure Example: Cats  #######
# Create a dictionary about a cat 
>>> cat = {'name':'Zophie','age':7,'color':'gray'}

# Combine with other dictionaries about multiple cats into a list
# This list of dictionaries is called a "Data Structure"
>>> allCats = []
>>> allCats.append({'name':'Zophie','age':7,'color':'gray'}) 
>>> allCats.append({'name':'Pooka','age':5,'color':'black'})
>>> allCats.append({'name':'Fat-tail','age':5,'color':'gray'})
>>> allCats.append({'name':'???','age':-1,'color':'orange'})

# This `allCats` variable can then be used in some cats-related program
>>> allCats
# [{'name': 'Zophie', 'age': 7, 'color': 'gray'}, {'name': 'Pooka', 'age': 5, 'color': 'black'}, {'name': 'Fat-tail', 'age': 5, 'color': 'gray'}, {'name': '???', 'age': -1, 'color': 'orange'}]

####### Dictionary Data Structure Example: Tic-Tac-Toe Board  #######
>>> theBoard = {'top-L':' ', 'top-M':' ', 'top-R':' ',
		'mid-L':' ', 'mid-M':'X', 'mid-R':' ',
		'low-L':' ', 'low-M':' ', 'top-R':' '}
		
# Print the board
>>> theBoard
# {'top-L': ' ', 'top-M': ' ', 'top-R': ' ', 'mid-L': ' ', 'mid-M': 'X', 'mid-R': ' ', 'low-L': ' ', 'low-M': ' '}

# Pretty version 		
>>> import pprint
>>> pprint.pprint(theBoard)
# {'low-L': ' ',
#  'low-M': ' ',
#  'mid-L': ' ',
#  'mid-M': 'X',
#  'mid-R': ' ',
#  'top-L': ' ',
#  'top-M': ' ',
#  'top-R': ' '}

# Empty Tic-Tac-Toe board
>>> theBoard['mid-M'] = ' '

>>> pprint.pprint(theBoard)
# {'low-L': ' ',
#  'low-M': ' ',
#  'mid-L': ' ',
#  'mid-M': ' ',
#  'mid-R': ' ',
#  'top-L': ' ',
#  'top-M': ' ',
#  'top-R': ' '}

# To Python, this is just a basic dictionary
# But we can have our program use this to represent a Tic-Tac-Toe board

# NOTE: The strings we use for keys (or even using dictionaries in the first place) are arbitrary; 
# As long as it's possible to represent all possible Tic-Tac-Toe boards with your data structure, you can use it in a Tic-Tac-Toe game program
# I use strings like "top-L" because they're short and easy to remember

>>> theBoard['mid-M'] = 'X'
>>> pprint.pprint(theBoard)
# {'low-L': ' ',
#  'low-M': ' ',
#  'mid-L': ' ',
#  'mid-M': 'X',
#  'mid-R': ' ',
#  'top-L': ' ',
#  'top-M': ' ',
#  'top-R': ' '}

# Can keep changing around the dictionary to represent the board in any way that we want
>>> # Top row all 'O'
>>> theBoard['top-L'] = '0'
>>> theBoard['top-M'] = '0'
>>> theBoard['top-R'] = '0'
>>> theBoard['mid-L'] = 'X'
>>> theBoard['low-R'] = 'X'

# This dictionary value is a data structure for Tic-Tac-Toe board in which Player O wins (3 O's in a row, on the top row)
>>> pprint.pprint(theBoard)
# {'low-L': ' ',
#  'low-M': ' ',
#  'low-R': 'X',
#  'mid-L': 'X',
#  'mid-M': 'X',
#  'mid-R': ' ',
#  'top-L': '0',
#  'top-M': '0',
#  'top-R': '0'}

# Write code that can recognize whenever there is 3 in a row inside this dictionary value
# Use print() function calls to print out line characters and X and Os
>>> def printBoard(board):
	print(board['top-L'] + '|' + board['top-M'] + '|' + board['top-R'])
	print('-----')
	print(board['mid-L'] + '|' + board['mid-M'] + '|' + board['mid-R'])
	print('-----')
	print(board['low-L'] + '|' + board['low-M'] + '|' + board['low-R'])

# Created a Data structure for Tic-Tac-Toe Board and wrote code (function) to interpret that data stucture
# So now you now have a program that models a Tic-Tac-Toe board
# Output can be as sophisticated as we want, fancy graphics with graphics library
>>> printBoard(theBoard)
# 0|0|0
# -----
# X|X| 
# -----
#  | |X

###### type() function ##############
# Can pass any value to the type() function and it will tell you the data type of that value
# Helpful in the interactive shell when you need to know what type of value you're dealing with

>>> type(42)
# <class 'int'>

>>> type('hello')
# <class 'str'>

>>> type(3.14)
# <class 'float'>

>>> type(theBoard)
# <class 'dict'>

# The value at this key is a string value
>>> type(theBoard['top-R'])
# <class 'str'>

# The type() and with pprint() functions are handy to use in the interactive shell when you just need to see what data you have 
>>> pprint.pprint(theBoard)
# {'low-L': ' ',
#  'low-M': ' ',
#  'low-R': 'X',
#  'mid-L': 'X',
#  'mid-M': 'X',
#  'mid-R': ' ',
#  'top-L': '0',
#  'top-M': '0',
#  'top-R': '0'}