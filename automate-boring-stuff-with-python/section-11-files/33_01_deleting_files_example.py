############## example.py ###############
###### Dry Run ##### 
import os

# os.chdir('C:\\Users\\Al\\Desktop')
os.chdir('C:\\Users\\sstallworth\\Desktop')

for filename in os.listdir():
    if filename.endswith('.txt'): # corrected from ".rxt" 
        os.unlink(filename)
        # print(filename)