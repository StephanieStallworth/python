###############  ? (zero or one) ###############

##### Bat Example #####
>>> import re

# Match the preceding group zero or one times
# It's an optional group that can either appear once or not appear at all
# "?" says "this group can appear in the text zero or 1 times in order to match this pattern"
>>> batRegex = re.compile(r'Bat(wo)?man') 
>>> mo = batRegex.search('The Adventures of Batman')
>>> mo.group()
# 'Batman'

# Will also match this 
>>> mo = batRegex.search('The Adventures of Batwoman')
>>> mo.group()
# 'Batwoman'

# But doesn't match this because "wo" can only appear once or zero times
>>> mo = batRegex.search('The Adventures of Batwowowoman')
>>> mo == None
# True

##### Phone Example #####
# This version requires an area code to match 
>>> phoneRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

# Text with area code would match 
>>> mo = phoneRegex.search('My phone number is 415-555-1234. Call me tommorow')

>>> mo.group()
# '415-555-1234'

# Text with no area code would not match 
>>> mo = phoneRegex.search('My phone number is 555-1234. Call me tommorow')

>>> mo == None
# True

# Look for phone numbers that do or do not have an area code, make area code optional 
# "?" to say "preceding group (area code) is optional"
# It can appear once or zero times
>>> phoneRegex = re.compile(r'(\d\d\d-)?\d\d\d-\d\d\d\d')

# Now both would match with or without area code
>>> phoneRegex.search('My phone number is 415-555-1234. Call me tommorow')
# <re.Match object; span=(19, 31), match='415-555-1234'>

>>> phoneRegex.search('My phone number is 555-1234. Call me tommorow')
# <re.Match object; span=(19, 27), match='555-1234'>

############### * (zero or more) ###############
# The "*" means it can appear any number of times
>>> batRegex = re.compile(r'Bat(wo)*man') 

# Matches if group doesn't appear at all 
>>> batRegex.search('The Adventures of Batman')
# <re.Match object; span=(18, 24), match='Batman'>

# Matches if group appears once
>>> batRegex.search('The Adventures of Batwoman')
# <re.Match object; span=(18, 26), match='Batwoman'>

# Also matches if group appears multiple times
>>> batRegex.search('The Adventures of Batwowowowowowowoman')
# <re.Match object; span=(18, 38), match='Batwowowowowowowoman'>

############### + (one or more) ###############
# The "wo" group is required to appear one or more times
>>> batRegex = re.compile(r'Bat(wo)+man')

# Won't find that in this string
>>> batRegex.search('The Adventures of Batman')

>>> # This function call will return None
>>> batRegex.search('The Adventures of Batman') == None
# True

# However this will match
>>> batRegex.search('The Adventures of Batman')

# Also matches because group can appear one or more times
>>> batRegex.search('The Adventures of Batwowowowowowowoman')
# <re.Match object; span=(18, 38), match='Batwowowowowowowoman'>

############### Escaping ?, *, and + ###############
# To literally match characters that otherwise have special meaning in Regular Expression strings, precede them with a backslash to escape them
>>> regex = re.compile(r'\+\*\?')

# This will match 
>>> regex.search('I learned about +*? regex syntax')
# <re.Match object; span=(16, 19), match='+*?'>

# Putting in a group and saying "it has to appear one or more times"
# The "\+" means "search for plus sign that is part of the text pattern"
# "+" by itself is saying "this is a Regular Expression instruction, match one or more of this preceding group in the parenthesis"
# Note this plus sign at the end goes INSIDE the string
>>> regex = re.compile(r'(\+\*\?)+')
>>> regex.search('I learned about +*?+*?+*?+*?+*? regex syntax')
# <re.Match object; span=(16, 31), match='+*?+*?+*?+*?+*?'>

############### {x} (exactly x) ###############
>>> haRegex = re.compile(r'(Ha){3}')

>>> haRegex.search('He said "HaHaHa"')
# <re.Match object; span=(9, 15), match='HaHaHa'>

# Looking for 3 phone numbers in a row	
# Which may or may not have an area code  
# And may or may not have a comma that separates them
>>> phoneRegex = re.compile(r'((\d\d\d-)?\d\d\d-\d\d\d\d(,)?){3}')

# Note there are no spaces 
>>> phoneRegex.search('My numbers are 415-555-1234,555-4242,212-555-0000')
# <re.Match object; span=(15, 49), match='415-555-1234,555-4242,212-555-0000'>

############### {x,y} (at least x, at most y) ###############
# Specify a minimum and maximum number of repetitions
>>> haRegex = re.compile(r'(Ha){3,5}') 

# If we had 3 "Ha"
>>> haRegex.search('He said "HaHaHa"')
# <re.Match object; span=(9, 15), match='HaHaHa'>

# If we had 4 "Ha"
>>> haRegex.search('He said "HaHaHaHaHa"')
# <re.Match object; span=(9, 15), match='HaHaHa'>

# If we had more than 5 "Ha"
# Would still match but would only match the first 5
>>> haRegex.search('He said "HaHaHaHaHaHaHa"')
# <re.Match object; span=(9, 15), match='HaHaHa'>

# Like slices, can leave off the first or second number
# Same as saying "0 to 5"
>>> haRegex = re.compile(r'(Ha){,5}')

# Same as saying "3 or more", an unbounded maximum
# Can have any number of them as long as it's at least 3
>>> haRegex = re.compile(r'(Ha){3,}')

############### Greedy vs Non-Greedy Match ###############
##### Greedy Match #####
>>> digitRegex = re.compile(r'(\d){3,5}')

# Matches the first 5 digits
# Python starts at the earliest match it can find
# Tries to match the longest possible string that matches this pattern
# In an ambiguous situation like this string where it could match 3, 4, or 5 digits, regular expression is going to match the longest possible string

>>> digitRegex.search('1234567890')
# <re.Match object; span=(0, 5), match='12345'>

##### Non-Greedy Match #####
# Non-Greedy Match, specify "?" after curly brace
# This question mark is different from the question mark that means "0 or 1" when it comes after a PATTERN (not directly after a GROUP)
>>> digitRegex = re.compile(r'(\d){3,5}?')

# This will do a non-greedy match
# Matches the smallest possible string this time
>>> digitRegex.search('1234567890')
# <re.Match object; span=(0, 3), match='123'>

# Happens in alot of RegEx syntax if you add a question mark it'll do a NON-GREEDY match
# Without it would do normal greedy match