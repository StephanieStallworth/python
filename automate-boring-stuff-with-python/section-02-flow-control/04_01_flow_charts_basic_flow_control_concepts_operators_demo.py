################# Comparison Operators #####################
# Expressions With Comparison Operators 
# These are just expressions, just like `2+2` math problem we did before
# Have a value and an operator 
>>> True
# True

>>> False
# False

>>> 42 == 42
# True

>>> 42 == 'Hello'
# False

>>> 42 == 41
# False

>>> 2 != 3
# True

>>> 42 < 100
# True

>>> 42 >= 100
# False

>>> 42 < 42
# False

>>> 42 <= 42
# True

>>> myAge = 26
>>> myAge < 30
# True

# Equals To Operator 
# Integers and strings will always not be equal to each other
>>> 42 == '42'
# False

# Float values and integer values can be equal to each other
>>> 42.0 == 42
# True

################# Boolean Operators #####################
#### and operator ####
# Evaluates to True if both Boolean values are True, otherwise it evaluates to False
>>> True and True
# True

# If one or both are False, entire expression evaluates to False
>>> False and True
# False

>>> False and False
# False

#### or operator ####
# Evaluates to True if either are True
>>> True or True
# True

>>> True or False
# True

# Only time is False is when both are false
False or False 
# False

#### not operator ####
# Just evaluates to the opposite Boolean value
>>> not True
# False

>>> not False
# True

# Often mix Boolean and comparison operators together in the same expression
>>> myAge = 26
>>> myPet = 'cat'

# Evaluates to True because both sides of the and operator evaluate to True
>>> myAge > 20 and myPet == 'cat'
# True