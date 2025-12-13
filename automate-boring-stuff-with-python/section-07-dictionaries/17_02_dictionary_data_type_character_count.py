######################## character_count.py ########################
############ Character Counting Program Example ############
# Works with any length string: https://automatetheboringstuff.com/files/rj.txt
# But would have to use triple quotes, Python's multi-line string
# Will have everything escape automatically and string can go across multiple lines
# Tripe quotes are covered in Lesson 19

message = 'It was a bright cold day in April, and the clocks were striking thirteen'
count = {} # 'r':12

# String is a list-like value so can use it in a for loop
for character in message.upper():  # upper() returns an uppercase form of the string 
    count.setdefault(character,0) # If you don't have letter as a key, create that key-value pair and make the value 0
    count[character] = count[character] + 1    

# Print dictionary 
# print(count)

# For a cleaner display of items in a dictionary
# pprint.pprint(count)

# Returns a string of what the pprint() function normally prints out
rjtext = pprint.pformat(count)

# Printing this string variable will to do what the pprint() function does
# But could also do something else with the string
print(rjtext) 