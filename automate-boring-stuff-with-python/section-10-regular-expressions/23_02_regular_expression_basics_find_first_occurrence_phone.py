######## Regular Expression Example: Find First Occurrence ##########
import re

message = 'Call me at 415-555-1011 tomorrow, or at 415-555-9999 for my office line.'

# 1. re.compile() to create Regular Expression Object with specified pattern  
phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

# 2. Call search() method to create a Match Object
# Pass it the variable containing the string you want to search 
mo = phoneNumRegex.search(message)  

# Or could type the string directly to make it more obvious 
mo = phoneNumRegex.search('Call me at 415-555-1011 tomorrow, or at 415-555-9999 for my office line.')

# 3. Call group() on the Match Object to print out the actual text  
print(mo.group())