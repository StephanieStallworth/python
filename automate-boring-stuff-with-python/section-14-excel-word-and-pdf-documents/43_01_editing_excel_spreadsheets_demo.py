# Create blank Excel spreadsheet
>>> import openpyxl
>>> wb = openpyxl.Workbook()

# See it is a workbook object  
>>> wb
# <openpyxl.workbook.workbook.Workbook object at 0x0DC10AB0>

# Get sheet names 
>>> wb.get_sheet_names()
# ['Sheet']

# Create sheet object from sheet name and save to variable
>>> sheet = wb.get_sheet_by_name('Sheet')
# <Worksheet "Sheet">

# Use square bracket syntax to get a cell
# Then read that cell object's value
>>> sheet['A1'].value

# See there is nothing in this cell yet 
>>> sheet['A1'].value == None
# True

# Add data to cell, use in assignment statement like any variable
>>> sheet['A1'] = 42
>>> sheet['A2']= 'Hello'

# This only exists in Python memory, in our program
# To save to the hard drive:
# Navigate to folder you want to save to 
# This is the `Documents` folder on my computer, use whatever folder you're going to save the document to on your computer
>>> import os
>>> os.chdir('c:\\Users\\sstallworth\\Documents')

# If you open an existing worksheet, good to save to a different file name
# That way you still have the original file
>>> wb.save('example.xlsx')

# Add new worksheets to your Excel file
# This creates and adds new worksheet object to workbook
# Also returns that worksheet object
sheet2 = wb.create_sheet()

# See the new sheet has been added 
>>> wb.get_sheet_names()
# ['Sheet', 'Sheet1']

# Get sheet name 
>>> sheet2.title
# 'Sheet1'

# Assign new name
>>> sheet2.title = 'My New Sheet Name'

# See that the name has been changed now 
>>> wb.get_sheet_names()
# ['Sheet', 'My New Sheet Name']

# Save to a different workbook (like "Save As")
>> wb.save('example2.xlsx')

# create_sheet() function will add new sheet at the end by default
# To create a sheet and make it the first sheet, pass an index value 
# Indexes start at 0 
>>> wb.create_sheet(index = 0, title = 'My Other Sheet')
# <Worksheet "My Other Sheet">

# Save to new workbook 
>>> wb.save('example3.xlsx')