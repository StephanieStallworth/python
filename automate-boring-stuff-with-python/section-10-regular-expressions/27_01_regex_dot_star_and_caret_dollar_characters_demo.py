############### Putting `^` at the Start ###############
>>> import re

# Caret `^` at the start of a regular expression to indicate the match has to occur at the beginning of the searched text
>>> beginsWithhelloRegex = re.compile(r'^Hello')

# Text pattern has to be at the beginning of the string
>>> beginsWithhelloRegex.search('Hello there!')
# <re.Match object; span=(0, 5), match='Hello'>

# If not at the beginning, not a match 
>>> beginsWithhelloRegex.search('He said "Hello!"')

# Doesn't start at start of string, returns None value
>>> beginsWithhelloRegex.search('He said "Hello!"') == None
# True

############### Putting `$` At The End ###############
# Dollar sign `$` at the end of a regular expression to indicate that the match has to occur at the end 
>>> endswithWorldRegex = re.compile(r'world!$')

# Matches if string ends with pattern 
>>> endswithWorldRegex.search('Hello world!')
# <re.Match object; span=(6, 12), match='world!'>

# Does not match if there are extra letters after the pattern 
# Returns the None value
>>> endswithWorldRegex.search('Hello world!a;dslfja;dlfjas')

############### Both `^` and `$` ##################
# Using both `^` and `$` means the entire string must match the pattern

# String that we're searching for has to begin and end with just this digit pattern 
# `^\d` means it has to begin with one or more digits 
# `\d$` means it has to end with one or more digits 
>>> allDigitsRegex = re.compile(r'^\d+$')

# Begins and ends with digit pattern
>>> allDigitsRegex.search('25874328597234587')
# <re.Match object; span=(0, 17), match='25874328597234587'>

# This has a non-digit character, so returns None value  
>>> allDigitsRegex.search('25874328x597234587')

############### `.` (anything except new line) ##################
# The dot `.` means any character EXCEPT for the new line

# This means match anything that is followed by the pattern "at"
>>> atRegex = re.compile(r'.at')

>>> atRegex.findall('The cat in the hat sat on the flat mat.')
# ['cat', 'hat', 'sat', 'lat', 'mat']

# Modify to look for pattern "at" that can be preceded by 1 or 2 letters of anything
>>> atRegex = re.compile(r'.{1,2}at')

# Finding two characters that can be anything, that includes whitespace characters in front of it 
>>> atRegex.findall('The cat in the hat sat on the flat mat.')
# [' cat', ' hat', ' sat', 'flat', ' mat']

############### `.*` To Match Anything ################
# `.` dot character means any character (except new lines)
# `*` star character means zero or more
# `*.` so dot star syntax means anything, any pattern whatsoever

>>> 'First Name: Al Last Name: Sweigart'
# 'First Name: Al Last Name: Sweigart'

# If you wanted to pull out first and last name
# Would have to find the index with multiple steps
>>> 'First Name: Al Last Name: Sweigart'.find(':')
# 10

>>> 'First Name: Al Last Name: Sweigart'.find(':') + 2
# 12

>>> 'First Name: Al Last Name: Sweigart'[12:]
# 'Al Last Name: Sweigart'

# Create a regular expression that does that for us using `.*`
>>> nameRegex = re.compile(r'First Name: (.*) Last Name: (.*)')

# Then use it to search text 

>>> nameRegex.findall('First Name: Al Last Name: Sweigart')

# Saying...
# "Look for text 'First Name: ' and whatever you find after that will be the first name"
# "Going up to the 'Last Name: ' part and then whatever comes after that is the last name"
# [('Al', 'Sweigart')]

###############  `(.*)` is greedy,  `(.*?)` is non-greedy ###############

###### Non-Greedy version with `.*?` with question mark ######
>>> serve = '<To serve humans> for dinner.>'

# This does a non-greedy match, "look for anything as long as we have the opening and closing angled bracket"
# In-between that can be anything, but as little "anything" as possible
>>> nongreedy = re.compile(r'<(.*?)>')

# Python says, "here is an opening angled bracket and we're going to be matching anything until we see a closed angled bracket"
# Stops at the FIRST closing angled bracket because NOT GREEDY 
>>> nongreedy.findall(serve)
# ['To serve humans']
 
###### Greedy version with `.*` with no question mark ######
>>> greedy = re.compile(r'<(.*)>')
>>> greedy.findall(serve)

# Python says, "here is that opening angled bracket"
# Then find anything
# But we can match even more text if we go past the first closing bracket
# Goes up to the second closing bracket instead 
# ['To serve humans> for dinner.']

############### Making Dot Match Newlines too (with `re.DOTALL`) ###############
# String with new line characters  
>>> prime = 'Serve the public trust. \nProtect the innocent. \nUphold the law.'
>>> print(prime)
# Serve the public trust. 
# Protect the innocent. 
# Uphold the law.

# The `.*` by itself matches any character EXCEPT the new line and zero or more occurrences of it 
>>> dotStar = re.compile(r'.*')

# Matches until it reaches a new line, because `.*` can be any character except for a new line
# Once it reaches a new line, it says "this is the first match that we found"
>>> dotStar.search(prime)
# <re.Match object; span=(0, 24), match='Serve the public trust. '>

# Pass a second argument `re.DOTALL` to the re.compile() function
# Configuration you can pass to the re.compile() function
# In this regular expression, dots here truly means everything including new line characters
>>> dotStar = re.compile(r'.*',re.DOTALL)

# The dot here truly means everything including new line characters
# Match everything, and also as much as possible because it is a greedy match
>>> dotStar.search(prime)
# <re.Match object; span=(0, 63), match='Serve the public trust. \nProtect the innocent. \>

############### `re.IGNORECASE` ###############
# Case sensitive
>>> vowelRegex = re.compile(r'[aeiou]')

>>> vowelRegex.search('Al, why does your programming book talk about Robocop so much?')
# <re.Match object; span=(9, 10), match='o'>

# Returns only lowercase vowels
>>> vowelRegex.findall('Al, why does your programming book talk about Robocop so much?')
# ['o', 'e', 'o', 'u', 'o', 'a', 'i', 'o', 'o', 'a', 'a', 'o', 'u', 'o', 'o', 'o', 'o', 'u']

# Pass in `re.I`   
# Or for case in-sensitive matching (ignore all casing and match both uppercase and lowercase)
# Pass in `re.I` or `re.IGNORECASE` argument to re.compile()
>>> vowelRegex = re.compile(r'[aeiou]', re.I) # Could also pass long version re.IGNORECASE

# Capital vowels are now also included in the matched text
>>> vowelRegex.findall('Al, why does your programming book talk about Robocop so much?')
# ['A', 'o', 'e', 'o', 'u', 'o', 'a', 'i', 'o', 'o', 'a', 'a', 'o', 'u', 'o', 'o', 'o', 'o', 'u']