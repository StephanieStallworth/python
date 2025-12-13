############### yourName.py ###############
##### Standard while loop ######
# Example of input validation 
# Loops are a good way to ensure the program keeps asking the user until they've entered some valid input for your program
name = ''
while name != 'your name':
    print('Please type your name.')
    name = input()
print('Thank you!')

##### While loop with break statement ######
# This case, the code doesn't really do anything new
# But `break` statements are useful if you have several different places inside of a while loop that could possibly cause the execution to leave from that point 
name = ''
while True:
    print('Please type your name.')
    name = input()
    if name == 'your name':
        break
print('Thank you!')

##### While loop with continue statement ######
spam = 0
while spam < 5:
    spam = spam + 1
    if spam == 3:
        continue
    print('spam is ' + str(spam))