################## example.py ##########################  
##### ZeroDivisionError #####
# Remember: functions that return without a return statement return the None value
def div42by(divideBy):
    try:
        return 42/divideBy # when code returns a ZeroDivisionError
    except ZeroDivisionError: # name of error 
        print('Error: You tried to divide by zero.') # the execution moves here

print(div42by(2))
print(div42by(12))
print(div42by(0))
print(div42by(1))

def divide(a,b):  
  try:  
    return a/b   
  except ZeroDivisionError: 
	return "Zero division is meaningless" # this line is executed instead if the try block is not executed   		

print(divide(1,0)) 
# Output 
# Zero division is meaningless 
# If there is no return statement, will just print out "None" 

##### ValueError ######
# Input validation 
print('How many cats do you have?')
numCats = input()

try: 
    if int(numCats) >= 4:
        print('That is a lot of cats.')
    else:
        print('That is not that many cats.')
except ValueError:
    print('You did not enter a number.')