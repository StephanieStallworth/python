############# import module #############
>>> import random

# Call the randint() function from this module
# Random integer between 1 and 10
>>> random.randint(1,10)
# 6
 
>>> random.randint(1,10)
# 4

>>> random.randint(1,10)
# 9

>>> random.randint(1,10)
# 2

>>> random.randint(1,10)
# 2

>>> random.randint(1,10)
# 10

############# import multiple modules #############
# import multiple modules at the same time
>>> import random, sys, os, math

############# alternative form with import * #############
# Alternative form of import 
# import everything from the random module
# Now don't have to type `random.` in front in order to call the random module's functions
>>> from random import *

# Can just type the function name to call it 
# However, using the full name `random.randin()` makes for more readable code
# Better to use the normal form of the import statement
>>> randint(1,10)
# 8