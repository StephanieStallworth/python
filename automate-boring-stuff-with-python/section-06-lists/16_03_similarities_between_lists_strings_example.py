################### example.py #####################
### Passing Lists in Function Calls #######
# 1. Function is defined (but not executed)
def eggs(cheese):
	cheese.append('Hello') # 4. `cheese` references the same underlying list that is referenced by the global `spam` variable and modifies it

# 2. Global variable `spam` stores a reference to the list variable [1,2,3]
spam = [1, 2, 3]

# 3. We pass a copy of that reference stored in `spam` into the eggs() function 
# This reference gets assigned its parameter `cheese` 
# And the execution enters the eggs() function block 
eggs(spam)

# 5. Then come back out of the function
# It's true that the `cheese` variable gets destroyed after this function returns
# But since inside the function it was making a change to the same underlying list that spam refers to
# These changes get reflected in `spam` outside of the function also 
print(spam)