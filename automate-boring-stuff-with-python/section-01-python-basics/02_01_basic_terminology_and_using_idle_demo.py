############# Expressions #############
>>> 2 + 2
# 4

>>> 2
# 2

>>> 5 - 3
# 2

>>> 3 * 7
# 21

>>> 22 / 7
# 3.142857142857143

>>> 2 + 3 * 6
# 20

>>> (2 + 3) * 6
# 30

>>> (5 - 1) * ((7 + 1) / (3 - 1))
# 16.0

# Get an error if Python doesn't understand 
>>> 5 +
# SyntaxError: invalid syntax

############# Data Types #############
# Whole number values are called integers 
>>> -2
# -2

>>> 30
# 30

# Values with decimals are called floating point numbers
>>> 3.14 
# 3.14

# This is an integer  
>>> 42
# 42

# This is a float
>>> 42.0 
# 42.0

# Data type for text values are called strings
>>> 'Hello world'
# 'Hello world'

# When the plus operator is used on strings it's the string concatenation operator
>>> 'Alice' + 'Bob'
# 'AliceBob'

>>> # When multiplication operator is used on an integer and a string that is String Replication
>>> 'Alice' * 3
# 'AliceAliceAlice'

# String concatenation and replication
>>> 'Hello' + '!' * 10
# 'Hello!!!!!!!!!!'

############# Variables #############
>>> spam = 42
>>> spam
# 42

# String concatenation with variable 
# Can use variables in expressions anywhere you would use values, a variable just evaluates to the value it contains
>>> spam = 'Hello'

# Regular string concatenation because variable `spam` evaluates down to the string inside of it "Hello"
# and that gets concatenated with the "World" 
>>> spam + 'World'
# 'HelloWorld'

# With space 
>>> spam + ' World'
# 'Hello World'

>>> spam = 'Goodbye'
>>> spam + ' World'
# 'Goodbye World'

# Can use expressions to assign to a variable 
>>> spam = 2 + 2
>>> spam
# 4

>>> # Can use the variable itself to set the variable's new value
>>> spam = 10
>>> spam = spam + 1