>>> import openpyxl
>>> import os

# Navigate to where workbook is saved 
>>> os.chdir('c:\\users\\sstallworth\\Documents\\')

# Create workbook object
>>> workbook = openpyxl.load_workbook('example.xlsx')

# See this is a workbook object type
>>> type(workbook)
# <class 'openpyxl.workbook.workbook.Workbook'>

# There are sheet objects you can obtain from the workbook if you know the sheet name
>>> sheet = workbook.get_sheet_by_name('Sheet1')
>>> type(sheet)
# <class 'openpyxl.worksheet.worksheet.Worksheet'>

# If you don't know the names of the sheet
>>> workbook.get_sheet_names()

# Get cell object 
>>> cell = sheet['A1']

# Evaluates to a cell object
# That cell about has a member variable called `value` that has the actual value
>>> cell.value
# datetime.datetime(2014, 4, 5, 13, 14)

# Because the cell is formatted with Excel's datetime data type
# In Python, this translates to a datetime object
# Datetime is a built-in module inside of Python
# Just want the string value representation pass to str() function
>>> str(cell.value)
# '2014-04-05 13:14:00'

# Can do this directory without having to create an extra `cell` variable first 
>>> str(sheet['A1'].value)
# '2014-04-05 13:14:00'

# Get value of text formatted cell 
>>> sheet['B1'].value
# 'Apples'

# Get value of numeric format cell 
>>> sheet['C1'].value
# 73

# Can convert integers to strings also 
>>> str(sheet['C1'].value)
# '73'

# Get cells using only numbers
# Row and columns begin at 1 not 0
>>> sheet.cell(row = 1, column = 2) 
# <Cell 'Sheet1'.B1>

# Evaluates to the same thing as this 
>>> sheet['B1'] 
# <Cell 'Sheet1'.B1>

# Nicer with code to use the cell() method
# Can pass a number directly to the cell() method in a `for` loop 
>>> for i in range(1,8):
	print(i, sheet.cell(row = i, column = 2).value)
	
# 1 Apples
# 2 Cherries
# 3 Pears
# 4 Oranges
# 5 Apples
# 6 Bananas
# 7 Strawberries