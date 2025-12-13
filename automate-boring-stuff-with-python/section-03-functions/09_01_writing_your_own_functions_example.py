################ example.py ######################
########## Simple Function  ##########
# View in PythonTutor.com

# Use `def` statement to define a new function called `hello()`
# `def` statement only defines the function, doesn't execute the code inside of it
def hello(): 
    print('Howdy!') # body of the function
    print('Howdy!!!')
    print('Hello there.')

# Call function 3 times
# The executer goes to the top of the function and EXECUTES the code inside of it 
# At the very end, the execution returns to the function call and then proceeds down to the next line/call
hello()
hello()
hello()

########## Function With Parameters ##########
def hello(name): # arguments passed to the function will be assigned the name parameter 
    print('Hello' + name)

# Call function passing an argument to it
hello('Alice') 
hello('Bob')

########## Function With return ##########
# Define function and include `return` statement at the end
def plusOne(number):
    return number + 1

# Call function and pass it an integer value
newNumber = plusOne(5)
print(newNumber)

########## Keyword Arguments #############
# Output will appear on separate lines 
# The print() function automatically adds a new line character to the end of the string that is passed
print('Hello')
print('World')

# Can change the `end` keyword argument to change this to a different string besides the new line characters
print('Hello', end ='')
print('World')