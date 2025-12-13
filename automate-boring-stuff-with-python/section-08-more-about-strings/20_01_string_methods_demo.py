###############  upper(), lower() ###############
>>> spam = 'Hello world!'
   
>>> spam.upper()
# 'HELLO WORLD!'

>>> spam # `spam` still contains the lower case values   
# 'Hello world!'

# To actually modify the `spam` variable
>>> spam = spam.upper() 
  
>>> spam.lower()  
# 'hello world!'

# Helpful for case insensitive comparisons	
# Use can input anything 

# all lowercase
>>> answer = input()
# yes

>>> answer
# 'yes'

# all uppercase
>>> answer = input()
# YES

>>> answer
# 'YES'

# Won't return anything because `answer` is uppercase
>>> if answer == 'yes':
	print('Playing again')

>>> answer == 'yes'
# False

# Call the lower() method
>>> answer.lower() == 'yes'
# True

# # Works with any response
>>> answer = 'yES'
>>> if answer.lower() == 'yes':
	print('Playing again') 
# Playing again

###############  isupper(), islower() ###############
>>> spam = 'Hello world!'
>>> spam.islower()
# False

>>> spam = 'hello world!'
>>> spam.islower()
# True

>>> spam = 'HELLO WORLD!'
>>> spam.isupper()
# True

# Blank string
>>> spam = ''

>>> spam.isupper()
# False

>>> spam.islower()
# False

# Need to be at least uppercase or lowercase character for isupper() or islower() to return True
>>> '12345'.islower()
# False
 
# Since upper() and lower() return strings, you can call string methods on those return string values as well
>>> 'Hello'.upper()
# 'HELLO'

>>> 'Hello'.upper().isupper()
# True

###############  isaplha(), isalnum(), isdecimal(), isspace(), istitle(), title() ###############
# isalpha()
>>> 'hello'.isalpha()
# True

>>> 'hello123'.isalpha()
# False

# isalnum()
>>> 'hello123'.isalnum()
# True

>>> '123'.isdecimal()
# True

# isplace()
# Returns True if there is nothing but space 
>>> '     '.isspace()
# True

# Returns False if it is not just space
>>> 'Hello world!'.isspace()
# False

# Returns True because this index returns a single space only
>>> 'Hello world!'[5].isspace()
# True

>>> 'Hello world!'[5]
# ' '

# istitle()
# Returns True only if every letter starts with an uppercase followed by lowercase only
>>> 'This Is Title Case.'.istitle() 
# True

# title()
# Modify string to title case
>>> 'hello world'.title()
# 'Hello World'

############### startswith(), endswith() ###############
# Returns True if the string value that they're called on begins or ends respectively with the string that was passed to it
>>> 'Hello world!'.startswith('Hello')
# True

>>> 'Hello world!'.startswith('H')
# True

>>> 'Hello world!'.startswith('ello')
# False

>>> 'Hello world!'.endswith('world!')
# True

# No exclamation mark will return False
>>> 'Hello world!'.endswith('world')
# False

############### join() ###############
# Each of the individual strings are combined together and joined by the ',' string
>>> ','.join(['cats','rats','bats'])
# 'cats,rats,bats'

# Can have a nothing that joins them (blank string)
>>> ''.join(['cats', 'rats', 'bats'])
# 'catsratsbats'

# Single space
>>> ' '.join(['cats', 'rats', 'bats'])
# 'cats rats bats'

# New line characters
>>> '\n\n'.join(['cats', 'rats', 'bats'])
# 'cats\n\nrats\n\nbats'

# When we pass it to a print() function call they will display on new lines 
>>> print('\n\n'.join(['cats', 'rats', 'bats']))
# cats
# rats
# bats

############### split() ###############
# Splits on whitespace by default 
>>> 'My name is Simon'.split()
# ['My', 'name', 'is', 'Simon']

# Can split on other characters
>>> 'My name is Simon'.split('m')
# ['My na', 'e is Si', 'on']

############### ljust(), rjust() ###############
# rjust()
# Add white space to the beginning 
>>> 'Hello'.rjust(10) 
# '     Hello'

# Total length of return string is 10, the argument that was passed to it 
>>> len('     Hello')
# 10

# ljust()
# Add white space to the end 
>>> 'Hello'.ljust(10)
# 'Hello     '

>>> 'Hello'.ljust(20)
# 'Hello               '

# Optional second argument to specify a different fill character other than a space
>>> 'Hello'.rjust(20,'*')
# '***************Hello'

>>> 'Hello'.ljust(25,'-')
# 'Hello--------------------'

############### center() ###############
# Works just like ljust() and rjust() but centers the text rather than justifying to the left or right
>>> 'Hello'.center(20)
# '       Hello        '

>>> 'Hello'.center(20,'=')
# '=======Hello========'

# Nice because can use the same code for different variables
>>> name = 'Al'
>>> name.center(20, '=')
# '=========Al========='

>>> name = 'Wendya;sdlfkja;dfj'
>>> name.center(20, '=')
# '=Wendya;sdlfkja;dfj='

############### strip(), rstrip(), lstrip() ###############
# strip()
# Strip whitespace from either side 
>>> spam = 'Hello'.rjust(10)
>>> spam.strip()
# 'Hello'

# Just returns a brand new string
# Doesn't modify `spam` in place, it still has all the spaces in there
>>> spam
# '     Hello'

# To change the variable would have to reassign it again
>>> spam = spam.strip()
>>> spam
# 'Hello'

# Removes whitespace from either side of the string
>>> '          x          '.strip()
# 'x'

# lstrip()
# Remove spaces from the left side of the string
>>> '          x          '.lstrip()
# 'x          '

# rstrip()
# Remove spaces from the right side of the string
>>> '          x          '.rstrip()
# '          x'

# Pass the string method characters we want to remove instead of whitespace
>>> 'SpamSpamBaconSpamEggsSpamSpam'.strip('ampS')
# 'BaconSpamEggs'

############### replace() ###############
# Takes two arguments: a string to look for and a string to replace it with 
>>> spam = 'Hello there!'
>>> spam.replace('e','XYZ')
# 'HXYZllo thXYZrXYZ!'

############### pyperclip.copy(), pyperclip.paste() ###############
# pyperclip.copy()
# Copy text to clipboard
# Then can paste the text Edit > Paste (CTRL + v)
>>> pyperclip.copy('Hello!!!!!!!!')

# pyperclip.paste()
# Returns the text that is already on the clipboard
>>> pyperclip.paste()
# 'Hello!!!!!!!!'