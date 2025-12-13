################ Groups in RegEx ################
# Create RegEx Object 
>>> import re
>>> phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

# Use search() method to search a string for this pattern
# Returns a Match Object
# phoneNumRegex.search('My number if 415-555-4242')
# <re.Match object; span=(13, 25), match='415-555-4242'>

# Store match object in a variable
>>> mo = phoneNumRegex.search('My number if 415-555-4242')

# Use the group() method to return the entire pattern/text
>>> mo.group()
# '415-555-4242'

# Groups
# If you just want a specify part, can do this by using parenthesis to mark out groups
# Note the parenthesis goes INSIDE the string
>>> phoneNumRegex = re.compile(r'(\d\d\d)-(\d\d\d-\d\d\d\d)') 

# Use the search() method to create a Match Object
>>> mo = phoneNumRegex.search('My number if 415-555-4242')

# Now when we call the group() method, does the same thing as before but has separate individual groups
# Full matching string
>>> mo.group()
# '415-555-4242'

# Returns just the first group that we marked
>>> mo.group(1)
# '415'

# Returns the second group that we marked 
>>> mo.group(2)
# '555-4242'

############### Parenthesis in RegEx ################
# Parenthesis have a special meaning in Regular Expressions, they mark out where the group begins and ends
# If you want to find literal parenthesis as part of the text in the pattern
# Would need to escape the opening and closing parenthesis characters with a backslash

>>> phoneNumRegex = re.compile(r'\(\d\d\d\) \d\d\d-\d\d\d\d')

# No parenthesis in string 
>>> mo = phoneNumRegex.search('My number if 415-555-4242')

# Parenthesis in string 
>>> mo = phoneNumRegex.search('My number if (415) 555-4242')

>>> mo.group()
# '(415) 555-4242'

############### The | Pipe Character ################
# Create ReGex Object
>>> batRegex = re.compile(r'Bat(man|mobile|copter|bat)')

# Search text using the RegEx object
>>> mo = batRegex.search('Batmobile lost a wheel')

# Print matches 
>>> mo.group()
# 'Batmobile'

# Search text that does not contain pattern 
mo = batRegex.search('Batmotorcycle lost a wheel')

# The search() method would return a value None
>>> mo == None
# True

# If the search() method can't find the Regular Expression pattern in the string that you passed it, going to return None
# So if you blindly save that to the `mo` variable then try to call group() on that None value will get an error message
# Because the None value doesn't have a method called group()
>>> mo.group()
    
# Traceback (most recent call last):
#   File "<pyshell#80>", line 1, in <module>
#     mo.group()
# AttributeError: 'NoneType' object has no attribute 'group'

# Just want to find which suffix was found the in the pattern
>>> mo = batRegex.search('Batmobile lost a wheel')
>>> mo.group(1)
# 'mobile'