########## Escape Characters ##########
# String value with single quotes
>>> 'Hello'
# 'Hello'

# Problem is if try to use a single quote inside of a string, will get an error 
# Python thinks string ends after "Alice" 
>>> 'That is Alice's cat.'
# SyntaxError: invalid syntax

# There are multiple ways to type strings
# Strings can begin and end with double quotes just as they do with single quotes
# Benefit of using double quotes is that a string can have a single quote character in it
>>> "That is Alice's cat."
# "That is Alice's cat."

# However, if you need to use both single AND double quotes inside of a string, you'll need to use escape character `\`
>>> 'Say hi to Bob\'s mother.'
# "Say hi to Bob's mother."

# Single print() statement that prints out text across multiple lines
>>> print('Hello there!\nHow are you?\nI\'m fine.')
# Hello there!
# How are you?
# I'm fine.

########## Raw Strings  ##########
# If you have text that includes many backslashes that you don't want seen as the beginning of an escape character
# Can use a raw string, exact same as a normal string except it begins with a lower case `r` right before it
>>> r'Hello'
# 'Hello'

# The "That is Carol\\'s cat." string is a regular, non-raw string value
>>> r'That is Carol\'s cat'
# "That is Carol\\'s cat"

# Print function call
# Backslash is literally interpreted as part of the string
>>> print(r'That is Carol\'s cat')
# That is Carol\'s cat
 
####### Multi-line Strings With Triple Quotes ###########
# A multi-line string in Python begins and ends with either 3 single quote characters or 3 double quote characters
# Any quotes, tabs, or new lines in between the triple quotes are considered part of the string
>>> print("""Dear Alice,
Eve's cat has been arrested for catnapping, cat burglary,
and extortion.
Sincerely,
Bob.""")
# Dear Alice,
# Eve's cat has been arrested for catnapping, cat burglary,
# and extortion.
# Sincerely,
# Bob.

# Notice we're breaking the rules of having one instruction per line
# Python is smart enough to realize that this is one "line of code"
# Just one print() function call, it doesn't matter it's being split across multiple lines
# Python thinks "until I see another triple quote, everything here is part of the string"

#  Can store into a variable 
>>> spam = """Dar Alice,
Eve's cat has been arrested for catnapping, cat burglary,
and extortion.
Sincerely,
Bob."""

# If we printed the variable, Python will automatically format using the `\n`
>>> spam
# "Dear Alice,\nEve's cat has been arrested for catnapping, cat burglary,\nand extortion.\nSincerely,\nBob."

# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# Really handy if you have some gigantic string
# For example if spam variable was assigned to text from :https://automatetheboringstuff.com/files/rj.txt
# Output of print(len(spam))
# 174128

####### Similarities Between Strings And Lists ###########
# Can think of a string as a list-like value with each character being an item in that list
>>> 'Hello world!'
# 'Hello world!'

# Can do all the same things with strings that I can do with lists
>>> spam = 'Hello world!'

>>> # Indexes to pick out a single letter from it
>>> spam[0]
# 'H'

>>> # Use slices to pick out some substring from it
>>> spam[1:5]
# 'ello'

>>> # Negative indexes to get the very last character
>>> spam[-1]

>>> # `in` and `not in` operators
>>> 'Hello' in spam
# True

>>> 'x' in spam
# False

>>> # Note these opeators are case sensitive
>>> 'HELLO' in spam
# False