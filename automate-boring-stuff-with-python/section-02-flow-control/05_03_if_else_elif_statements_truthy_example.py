############### truthy.py ##################
## Truthy and Falsey Values ####
print('Enter a name.')
name = input()

# Will sometimes see a condition like this that is kind of weird
# The `name` variable is set to whatever the user has typed
# But the input() function will be returning a string value not a Boolean True or False value like examples above
# The reason this code works is that the condition can use Truthy and Falsy values for strings

if name:       # Any non-blank string is truthy, which is considered to a True condition 
    print('Thank you for entering a name.')
else:         # If nothing is entered, the blank string is considered a Falsey value so condition is considered be False
    print('You did not enter a name.')

# For strings, the blank string is a Falsy value
# If condition evaluates to a blank string, it's considered to be the same as the False Boolean value
# Blank string is falsey all others are truthy  
# Blank string is falsey all others are truthy  

# Good shortcut, but general better to be more explicit
print('Enter a name.')
name = input()
if name != '': 
    print('Thank you for entering a name.')
else:
    print('You did not enter a name.')