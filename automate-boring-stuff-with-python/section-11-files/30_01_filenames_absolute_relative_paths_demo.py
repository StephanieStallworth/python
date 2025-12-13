############################################################################
######################### WITHOUT OS MODULE ################################
############################################################################
# Use strings to represent filepaths
# Need to escape the backslash to have a literal backslash in string
>>> 'c:\\spam\\eggs.png'
# 'c:\\spam\\eggs.png'

# Or begin string with `r` to begin a raw string
>>> r':\spam\eggs.png'
# ':\\spam\\eggs.png'

>>> print(r':\spam\eggs.png')
# :\spam\eggs.png

# Can join a bunch of names for different folders
# This would only work on Windows, ideally want Python scripts to work on all systems
>>> '\\'.join(['folder1','folder2','folder3','file.png'])
# 'folder1\\folder2\\folder3\\file.png'

############################################################################
######################### OS MODULE BASICS  ################################
############################################################################
# `os` module contains different filepath related functions we can use
>>> import os

################ `os.path.join()` ################
# Takes several string arguments 
# Returns a string value of a path that is appropriate for the operating system you're running
>>> os.path.join('folder1','folder2','folder3','file.png')
# 'folder1\\folder2\\folder3\\file.png'

# Value that the join() function uses is stored in a `sep` variable inside the `os` module
>>> os.sep
# '\\'

################ `os.getcwd()` ################
# The current working directory tells the program what folder it should look in when we just give it a filename without a filepath
# Use `os.getcwd()` to get the current working directory as a string value
>>> os.getcwd()
# 'C:\\Users\\sstallworth\\AppData\\Local\\Programs\\Python\\Python37-32'

>>> 'spam.png'
# 'spam.png'

################ `os.chdir()` ################
# Change current working directory by passing it a new filepath 
>>> os.chdir('c:\\')

# Now check where you are
>>> os.getcwd()
# 'c:\\'

################ Absolute and Relative Paths ################
# Absolute filepaths always begin with the root folder
>>> 'c:\\folder1\\folder2\\spam.png'
# 'c:\\folder1\\folder2\\spam.png'

# Relative filepath is relative to the current working directory
# Does NOT begin with root folder but can begin with other folders
>>> 'spam.png'
# 'spam.png'

>>> 'folder1\\folder2\\spam.png'
# 'folder1\\folder2\\spam.png'

################ The `.` and `..` Folders ################
# "This" folder
# './spam.png'  

# In the parent folder of the parent folder
# '..\\..\\spam.png'

############################################################################
################# FUNCTIONS FOR STRING MANIPULATION ########################
############################################################################

########## `os.path.abspath()` and `os.path.isabs()` ##############
##### `os.path.abspath()` #####
# Pass it a relative path and it will return an absolute path version
>>> os.chdir('C:\\Users\\sstallworth\\AppData\\Local\\Programs\\Python\\Python37-32')

# Give it a filename only, will assume it's in the current working directory
# And return an absolute path of that 
>>> os.path.abspath('spam.png')
# 'C:\\Users\\sstallworth\\AppData\\Local\\Programs\\Python\\Python37-32\\spam.png'

# Can also give it shortcut folders and it will give you the absolute path from that location 
# In the parent folder of the parent folder there is a `spam.png` file
>>> os.path.abspath('..\\..\\spam.png') 
# 'C:\\Users\\sstallworth\\AppData\\Local\\Programs\\spam.png'

##### `os.path.isabs()` #####
# Pass in filepath and it will return True or False depending if it is an absolute path
>>> os.path.isabs()

>>> os.path.isabs('..\\..\\spam.png')
# False

>>> os.path.isabs('c:\\folder\\folder')
# True

########## `os.path.relpath()` ##########
# Gives you a relative path to get to a target path (String Argument 1)
# Given the a path starting path (String Argument 2)
>>> os.path.relpath('c:\\folder1\\folder2\\spam.png','c:\\folder1')

# To get to `c:\\folder1\\folder2\\spam.png`
# From `c:\\folder1`
# Would need to go to this folder 
# 'folder2\\spam.png'

# Want to pull out just the directory part or just the file name part

########## `os.path.directname()` and `os.path.basename()` ##########
##### os.path.directname() #####
# Pull out just the directory part 
>>> os.path.dirname('c:\\folder1\\folder2\\spam.png')
# 'c:\\folder1\\folder2'

##### `os.path.basename()` #####
# Pull out just the filename 
>>> os.path.basename('c:\\folder1\\folder2\\spam.png')
# 'spam.png'

# Can also use to pull out the last folder of a file path (anything after the last slash)		    
>>> os.path.basename('c:\\folder1\\folder2')
# 'folder2'

############################################################################
################# FUNCTIONS TO EXAMINE HARD DRIVE ##########################
############################################################################

########## `os.path.exists()` ##########
# Made up file that doesn't exist
>>> os.path.exists('c:\\folder1\\folder2\\spam.png')
# False

# Calcuator program that DOES exist
>>> os.path.exists('c:\\windows\\system32\\calc.exe')
# True
   
########## `os.path.isfile()` and `os.path.isdir()` ##########
##### `os.path.isfile()` #####
# Pass it a file will return True
>>> os.path.isfile('c:\\windows\\system32\\calc.exe')
# True

# Pass it just the folder part, will return False
>>> os.path.isfile('c:\\windows\\system32')
# False

##### `os.path.isdir()` #####
# Pass it a file will return False
>>> os.path.isdir('c:\\windows\\system32\\calc.exe')
# False

# Pass it a directory, will return True 
>>> os.path.isdir('c:\\windows\\system32')
# True
  
########## `os.path.getsize()` and `os.listdir()` ##########
##### `os.path.getsize()` #####
# Returns size in bytes as a integer
>>> os.path.getsize('c:\\windows\\system32\\calc.exe') 
# 26112

##### `os.listdir()` #####
# Returns list of file and directory names inside folder you passed it 

# os.listdir('c:\\automatebook')
>>> os.listdir('C:\\Users\\sstallworth\\Desktop')
['BORING', 'desktop.ini', 'PROJECTS', 'STEPHANIE', '~$01372568 Personas Notes.docx', '~$CASE01401838_seg_danone_activatortransactions20181023_20181106.xlsx', '~$CASE01411902_miqplus_absco_ecomm.xlsx', '~$dg_pem_exclusion_list.xlsx', '~$lcome to BAR.docx', '~$miqplus_codebook.xlsx', '~$otient Training.docx', '~$RiQ YOY Automation Template - final - 2018 Q2 Runbb.xlsm', '~$RiQ YOY Automation Template - final - 2018 Q3 Runbb absco.xlsm', '~$riqactivedigital_banner_mmstatic_yoymergev02 year ending 2018q3.xlsx', '~$SF01372568 Personas Purchase Behavior.xlsx', '~$SF01389151_DG_segmentation_refresh_20181025.xlsx', '~$S_Paper_Submission.docx', '~$tail SIT_BAR Q4 Meeting Agenda.docx', '~$USER_SID COUNT.xlsx', '~WRL0003.tmp', '~WRL0005.tmp', '~WRL0247.tmp', '~WRL1989.tmp']

########## Example Code: Finding the total size of all files in a folder ##########
# My version 
>>> totalSize = 0
>>> for filename in os.listdir('C:\\Users\\sstallworth\\Desktop'):
	if not os.path.isfile(os.path.join('C:\\Users\\sstallworth\\Desktop', filename)):
		continue
	totalSize = totalSize + os.path.getsize(os.path.join('C:\\Users\\sstallworth\\Desktop', filename))

>>> totalSize
# 145507

# Original example 
>>> totalSize = 0   
>>> for filename in os.listdir('c:\\automatebook'):  
		if os.path.isfile(os.path.join('c:\\automatebook',filename)):  
			continue  
		totalSize = totalSize + os.path.getsize(os.path.join('c:\\automatebook',filename))  
		os.makedirs():to create new folders, pass this a relative or absolute file path 
		os.markdirs('c:\\delicious\\walnut\\waffles')	  

########## `os.makedirs()` ##########
# Pass it relative or absolute file paths and will create all of the folders that you specify 	      
>>> os.makedirs('c:\\delicious\\walnut\\waffles')