################### fivetimes.py ###################
######### for loop  #########
# Loops all the way up to but NOT including number passed to range function call
print('My name is')
for i in range(5):
      print('Jimmy Five Times ' + str(i))

### while loop equivalent ####
# Can use `while` loops to do the same thing as a `for` loop, but `for` loops are just more concise  
# By using a `for` loop, wouldn't have to worry about adding extra lines at the beginning and end  
print('My name is')
i = 0 
while i < 5:
    print('Jimmy Five Times ' + str(i))
    i = i + 1

######### for loop with two arguments passed to range #########
# Can enter multiple arguments separated by a comma
# Lets you change integers to range that follows any sequence of integers starting at numbers other than zero
print('My name is')
for i in range(12,16):
      print('Jimmy Five Times ' + str(i))

######### for loop with three arguments passed to range #########
print('My name is')
for i in range(5,-1,-1): # make the for loop count down instead of up 
      print('Jimmy Five Times ' + str(i))
