############### The open() Function ###############
# The call to open() returns a file object, save to a variable
>>> helloFile = open('c:\\users\\sstallworth\\hello.txt')

# Then can read contents of file 
>>> helloFile.read()
# 'Hello world!\nHow are you?'

# Then close it when you're done 
>>> helloFile.close()

# Can only read through the file contents once, best to save it to a variable
# Otherwise, if we want to read it again, would have to call open() again then read it
>>> content = helloFile.read()

>>> print(content)
# Hello world!
# How are you?

>>> helloFile.close()

############### the readlines() Method ###############
# Returns all of the lines as STRINGS inside of a LIST
>>> helloFile.readlines()
# ['Hello world!\n', 'How are you?']

>>> helloFile.close()

############### Open in Write Mode ###############
# Pass `w` as the second argument to the open() function 
>>> helloFile = open('c:\\users\\sstallworth\\hello.txt', 'w')

############### Open in Append Mode ###############
# Pass `a` as the second argument to the open() function 
>>> helloFile = open('c:\\users\\sstallworth\\hello.txt', 'a')

>>> helloFile.write('Hello!!!!')

>>> helloFile.close() 

############### The write() Method ############### 
# Open/create new file 
helloFile = open('c:\\users\\sstallworth\\hello2.txt', 'w')

# Then write to it with the write() Method
# Returns the bytes (characters) that it wrote, not including spaces
>>> helloFile.write('Hello!!!!!!!')
# 12

# Can call multiple times
>>> helloFile.write('Hello!!!!!!!')
# 12

>>> helloFile.write('Hello!!!!!!!')
# 12

# Then finally close the file
>>> helloFile.close()

# Then open newly created file from Notepad
# Will notice that the write() method doesn't automatically add a newline character at the end of each string that we passed it
# Not like how print() function automatically adds new lines when it prints things to the screen

# Would have to add new lines ourselves at the end of the strings
# helloFile.write('Hello!!!!!!!\n')

# Open/create new
>>> baconFile = open('bacon.txt','w')

# Write to file 
>>> baconFile.write('Bacon is not a vegetable.')
# 25

# Then close
# >>> baconFile.close()

# Pass the open() function a relative path, so will open and write to file in current working directory 
# Locate with `os` module 
>>> import os

# Here is where we'll find the file
>>> os.getcwd()
# 'C:\\Users\\sstallworth\\AppData\\Local\\Programs\\Python\\Python37-32'
 
# Pass it to print() to get version you can copy and paste into file explorer
>>> print(os.getcwd())
# C:\Users\sstallworth\AppData\Local\Programs\Python\Python37-32

# Open file in append mode
>>> baconFile = open('bacon.txt','a')

# Writes to the end of the file 
>>> baconFile.write('\n\nBacon is delicious.')
# 21

# Then close like usual 
>>> baconFile.close()

############### The shelve Module ##############
>>> import shelve

##### The shelve.open() Method #####
# Will return a shelf file object
>>> shelfFile = shelve.open('mydata')

# Can make changes to the shelf value as if it were a dictionary
>>> shelfFile['cats'] = ['Zophie','Pooka','Simon','Fat-tail','Cleo']

# Then close the shelf value when you're done
>>> shelfFile.close()

# When I run program again in the future, can just have code that reopens the shelf
>>> shelfFile = shelve.open('mydata')

# Then grab the value like a dictionary
# Variables in Python are kind of like key-value pairs in a dictionary
# cats = ['Zophie', 'Pooka', 'Simon', 'Fat-tail', 'Cleo']
# Cats variable is the key, list to be a value
# That is why shelf file has a dictionary-like structure like this 
>>> shelfFile['cats']
# ['Zophie', 'Pooka', 'Simon', 'Fat-tail', 'Cleo']

# Make sure to close when you're done
shelfFile.close()

############### The keys() and values() Shelf Methods ###############
# Return list-like values for all the keys and values inside them
>>> shelfFile = shelve.open('mydata')

# To convert to an actual list
>>> list(shelfFile.keys())
# ['cats']

>>> list(shelfFile.values())
# [['Zophie', 'Pooka', 'Simon', 'Fat-tail', 'Cleo']]