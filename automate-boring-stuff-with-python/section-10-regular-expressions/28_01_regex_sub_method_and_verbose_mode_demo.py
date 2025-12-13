############### The sub() Method ###############
>>> import re

# Agent with one or more word characters
>>> namesRegex = re.compile(r'Agent \w+')

# Stops at a non-word character so stops at the space
>>> namesRegex.findall('Agent Alice gave the secret documents to Agent Bob.')
# ['Agent Alice', 'Agent Bob']

# sub() method returns string with substitutions made to it
>>> namesRegex.sub('REDACTED','Agent Alice gave the secret documents to Agent Bob.')
# 'REDACTED gave the secret documents to REDACTED.'

############### Using `\1`, `\2`, etc in sub() ###############
# `\number` syntax tells Python, "in the substituted string, I want some part of the original matching string"
# Don't want to give the full name, just a little of it

# Find first letter of Agent's name and put in group
# Followed by zero or more other letters 
>>> namesRegex = re.compile(r'Agent (\w)\w*')

# Retuns just the group (first letter of their name), not the entire match 
>>> namesRegex .findall('Agent Alice gave the secret documents to Agent Bob.')
# ['A', 'B']

# Now use the `/number` syntax in the sub() call 
#`\1` means inside of that match, whatever was in the first group (in the RegEx we marked the first group as the first letter)
>>> namesRegex.sub(r'Agent \1****','Agent Alice gave the secret documents to Agent Bob.')
# 'Agent A**** gave the secret documents to Agent B****.'

############### Verbose Mode with `re.VERBOSE` ###############
# In verbose mode, can break pattern like this into different lines to make it easier to read  
re.compile(r'\d\d\d-\d\d\d-\d\d\d\d', re.VERBOSE)  

# Add new lines, spaces and comments inside of the string that aren't going to be part of the pattern itself
# Great for really complicated regular expression strings 
>>> phoneRegex = re.compile(r'''
(\d\d\d-)|    # area code (without parens, with dash)
(\(\d\d\d\) ) # -or- with parens and no dash
\d\d\d        # first 3 digits 
-             # second dash
\d\d\d\d      # last 4 digits
\sx\d{2,4}    # extension, like x1234''', re.VERBOSE)

############### Using Multiple Options (re.I, re.DOTALL, re.VERBOSE) ###############
# Using Multiple Options  
>>> (re.I, re.DOTALL, re.VERBOSE)  
re.compile(r''' 
\d\d\d  # area code  
-   # first dash  
\d\d\d  # first 3 digits  
-    #second dash  
\d\d\d\d # last 4 digits''', re.VERSBOSE)  
\sx\d{2,4}  # extension, like x1234''', re.IGNORECASE | re.DOTALL | re.VERBOSE) 