############### os.walk() Function ###############
>>> import os

# Walk through entire folder tree  
>>> for folderName, subfolders, filenames in os.walk('c:\\delicious'):
	print('The folder is ' + folderName)
	print('The subfolders in ' " + folderName + ' are: ' + str(subfolders))
	print('The filenames in ' " + folderName + ' are: ' + str(filenames))
	print() # blank print() to print out new line 

## Output 
## Tells us what is inside the `delicious` folder  
# The folder is c:\delicious
# The subfolders in c:\delicious are: ['foo','walnut'] # folders inside of `delicious` folder 
# The filenames in c:\delicious are: ['spam.txt', 'spamspamspam.txt'] # files inside of `delicious` folder 

## Then looks at each of the 2 folders underneath the `delicious` folder
# The folder is c:\delicious\foo
# The subfolders in c:\delicious\foo are: []
# The filnames in c:\delicious\foo are: ['spam.txt']

# The folder is c:\delicious\walnut
# The subfolders in c:\delicious\walnut are: ['waffles']
# The filenames in c:\delicious\walnut are: ['eggs.txt']

# And goes deeper and deeper inside those folders 
# The folder is c:\delicious\walnut\waffles
# The subfolders in c:\delicious\walnut\waffles are: []
# The filenames in c:\delicious\walnut\waffles are: ['bacon.txt, 'ham.txt']