######## Global vs Local Variable Example #############
spam = 42 # global variable

def eggs():
    spam = 42 # local variable

print('Some code here.')
print('Some more code.')

#########  Code in the global scope cannot use any local variables #########
def spam():
    eggs = 99 # Created a local scope, local variable eggs

# Will cause an error 
# Because when the spam() function returns, it's local scope is destroyed
# Any variables in that scope are forgotten (example: `eggs` variable no longer exists) 
spam()
print(eggs)

######### Code in one function's local scope cannot use variables in another local scope ##########
def spam():
    eggs = 99 
    bacon()
	print(eggs) # print spam() `eggs` variable because `eggs` variable from the bacon() function is destroyed when you came back out of the function

def bacon():
	ham = 101
	eggs = 0

# Run the program
# spam() is defined skip over the body of that function
# bacon() is defined skip over the body of that function
# now program calls the spam() function 
    # Local scope gets created for the spam() function
    # Local variable `eggs`
    # Call the bacon() from spam(), so execution moves into the bacon() function 
        # Creates new local scope for the bacon() function
        # Assigns local variables `ham` and `eggs`, `eggs` has the same name but are referring to different variables
    # Return back to the eggs() function, which means that bacon() function's local scope has been destroyed, and its local variables are now gone

########## No assignment statement in function ##########
#### Using a global variable ####
def spam():
	print(eggs) #  No assignment statement in the function, so this is referring to global `eggs` variable

eggs = 42 # Global `eggs` variable, assigned outside of all functions in the global scope

# Then call the spam() function which will print `eggs` 
# Since there no local variable called `eggs`, Python is smart enough to say maybe they are talking about a global variable `egg` and will check for it 
# Will print the value of the global variable `eggs`, which is 42
spam()

########## Assignment statement in function ##########
def spam():
	eggs = 'Hello'
	print(eggs) # Python will now treat this as a local variable

eggs = 42 # global variable

# When you call spam() function
spam()
print(eggs) # Once you come out of the spam() function, that local scope is destroyed and will print the global variable `eggs` 

########## Updating a global variable from inside a function ##########
def spam():
	global eggs # mark `eggs` as a global variable first at the top of the function, this tells Python "I'm referring to the global variable, don't create a new local variable"
	eggs = 'Hello' # Then assign the value you want to the global variable
	print(eggs)

eggs = 42 

# When you call spam() function, will print the value of the local variable `eggs` which is `Hello` 
spam()
print(eggs) # This will print the updated global variable for `eggs` which is `Hello`