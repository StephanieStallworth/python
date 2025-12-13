############### shutil.copy() ###############
>>> import shutil

# The copy() function will copy a file to a destination folder
>>> shutil.copy('c:\\spam.txt','c:\\delicous')

# Copy and rename at the same time by specifying a file name for the destination
>>> shutil.copy('c:\\spam.txt','c:\\delicous\\spamspamspam.txt')

############### shutil.copytree() ###############
# Copy entire folder and its contents 
>>> shutil.copytree('c:\\delicious','c:\\delicious_backup')

############### shutil.move() ###############
# Move file to a new location
>>> shutil.move('c:\\spam.txt','c:\\delicious\\walnut')

# Rename files
>>> shutilmove('c:\\delicious\\walnut\\spam.txt', 'c:\\delicious\\walnut\\eggs.txt')