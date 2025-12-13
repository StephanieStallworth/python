############### while_example.py ##################
### While Statement ######
# This while loop iterates 5 times, on each iteration the `spam` variable was increased by 1
# Result is it will print 'Hello world!' 5 times 
# After the 5th time, `spam` is set to value 5 so the condition 5 < 5 would evaluate to False
# Then would continue on with the rest of the program
spam = 0
while spam < 5: # After the 5th time condition would be False, execution just continues with the rest of the program
    print('Hello world!')
    spam = spam + 1

### If Statement ######
# Looks very similar to an `if` statement, except at the end
# The difference is in how they behave 
# At the end of an `if`` block the program execution continues on with the rest of the program
# At the end of a `while` block, the execution jumps back to the start of the `while` statement and re-checks that condition until the first time this condition is False
# Then it contineus on with the rest of the program
if spam < 5:
    print('Hello world!')
    spam = spam + 1    