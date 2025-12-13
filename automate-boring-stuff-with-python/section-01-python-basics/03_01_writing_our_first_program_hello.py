################### hello.py ##########################
# This program says "Hello" and asks for my name

# Calling the print function
print('Hello World!')

# Another call to the print() function
print('What is your name?') # ask for their name
myName = input()
print('It is good to meet you, ' + myName) # single string gets passed to print() function call 
print('The lengh of your name is:')
print(len(myName)) # len() takes a string argument and evaluates to the integer value of the length of the string

print('What is your age?') # ask for their age
myAge = input()
print('You will be ' + str(int(myAge) + 1) + ' in a year.') # str(), int(), float() functions