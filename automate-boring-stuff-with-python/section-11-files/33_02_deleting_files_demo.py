############### os.unlink() ###############
# Passing it a relative filepath, doesn't begin with root folder
>>> import os
>>> os.unlink('bacon.txt')

# To know which `bacon.txt` file its deleting
>>> os.getcwd()

# Will delete the `bacon.txt` that is inside this folder
# 'C:\\Users\\sstallworth\\AppData\\Local\\Programs\\Python\\Python37-32'

############### os.rmdir() ###############
# Delete folder that is completely empty   
>>> os.rmdir('c:\\delicious') 

############ shutil.rmtree() #############
# Delete a NON-EMPTY folder 
>>> import shutil
>>> shutil.rmtree('c:\\delicious')

############### Dry Run ##################
# Version with typo 
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# IMPORTANT FILE!!!.rxt

# After correction
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# Switching to Python.txt
  
############### send2trash ###############
# Send to recycling bin instead of deleting permanently 
>>> import send2trash
>>> send2trash.send2trash('c:\\users\\al\\desktop\\IMPORTANT FILE!!!.rxt')