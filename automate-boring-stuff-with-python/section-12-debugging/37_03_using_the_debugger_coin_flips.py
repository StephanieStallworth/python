################## coinFlips.py ###############
# Breakpoints example 
#! python3
import random

heads = 0

for i in range(1,1001):
    if random.randint(0,1) == 1:
        heads = heads + 1
    if i == 500: # Would have to click "Over" until `i` is 500 before it will enter this block 
        print('Halfway done!') # Instead right-click the first line of the block and select "Set Breakpoint"

print('Heads came up ' + str(heads) + ' times.')