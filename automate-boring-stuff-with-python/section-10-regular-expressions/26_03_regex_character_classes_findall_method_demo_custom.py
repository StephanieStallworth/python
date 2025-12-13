#####################  Making Your Own Character Classes ##################
# Syntax is two square brackets and all of the characters we want to be inside of our character class 

# Pick out all the vowels from a string
>>> vowelRegex = re.compile(r'[aeiouAEIOU]') 
>>> vowelRegex.findall('Robocop eats baby food.')
# ['o', 'o', 'o', 'e', 'a', 'a', 'o', 'o']

# Look for not just vowels, but those that match exactly two of them in a row
>>> doubleVowelRegex = re.compile(r'[aeiouAEIOU]{2}')
>>> doubleVowelRegex.findall('Robocop eats baby food')
# ['ea', 'oo']

#################### Negative Character Classes ######################
# Adding caret at the start makes it a negative character class
# This means match every character that is NOT specified in the character class
>>> consonantsRegex = re.compile(r'[^aeiouAEIOU]')

# Returns consonants along with spaces, punctuation marks as well
# ANY character that is not the character class, including punctuation marks and numeric digits
>>> consonantsRegex.findall('Robocop eats baby food')
# ['R', 'b', 'c', 'p', ' ', 't', 's', ' ', 'b', 'b', 'y', ' ', 'f', 'd']