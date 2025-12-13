########## Regular Expression Example: Find All Occurances ##########
import re
 
phoneNumRegex = re.compile(r'\d\d\d-\d\d\d-\d\d\d\d')

# Find every occurance of phone number in string
print(phoneNumRegex.findall('Call me at 415-555-1011 tommorow, or at 415-555-9999 for my office line.'))