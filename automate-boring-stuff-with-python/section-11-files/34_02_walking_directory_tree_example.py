import os

for folderName, subfolders, filenames in os.walk('c:\\delicious'): # Can pass in the root folder `c:\\` to do on all files on your computer 
    print('The folder is ' + folderName)
    print('The subfolders in ' " + folderName + ' are: ' + str(subfolders))
    print('The filenames in ' " + folderName + ' are: ' + str(filenames))
    print()

    for subfolder in subfolders:
        # print(subfolder)
        # os.unlink(subfolder)
        if 'fish' in subfolder:
            # os.rmdir(subfolder)
            print('rmdir on ' + subfolder) # Do dry run first 
    for file in filenames:
        if file.endswith('.py'):
            # shutil.copy(os.path.join(folderName,file), os.path.join(folderName, file + '.backup')
            print('copying ' + shutil.copy(os.path.join(folderName,file), os.path.join(folderName, file + '.backup') # Do dry run first 