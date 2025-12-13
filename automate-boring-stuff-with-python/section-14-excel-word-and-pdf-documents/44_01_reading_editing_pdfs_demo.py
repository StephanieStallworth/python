########################## Reading PDF Files #############################
>>> import PyPDF2
>>> import os

# Change working directory to folder where PDF docs are saved
# C:\Users\Al\Documents ("My Documents" folder on my computer.)
>>> os.chdir('c:\\users\\sstallworth\\Documents')

# Open will open files in read mode by default, need to open in "read binary" mode since this is a binary file
pdfFile = open('meetingminutes1.pdf', 'rb')

# Pass this file object to PyPDF2 to create PDRReader object 
# Original 
# reader = PyPDF2.PdfFileReader(pdfFile)

# Working 
>>> reader = PyPDF2.PdfReader(pdfFile)

# Returns new PDF reader object of the file
>>> reader
# <PyPDF2._reader.PdfReader object at 0x03CB8550>

# Reader objects all have a member variable called `numPages` or pages
# Stores an integer of how many pages are inside of the PDF document
# Original 
# reader.numPages

# Working 
>>> len(reader.pages)
# 19

# Also have a getPage() method
# Note pages start at 0, not 1

# Original 
>>> reader = reader.getPage(0)

# Working 
>>> page = reader.pages[0]

# Extract text and return as Python String
# Original 
>>> page.extractText()

# Working 
>>> page.extract_text()

# Extract all the text in document and print
>>> for pageNum in range(len(reader.pages)):
	     print(reader.getPage(pageNum).extractText())

# Working 
>>> for pageNum in range(len(reader.pages)):
            print(reader.pages[pageNum].extract_text())

########################## Create PDFs By Combining PDFs ##########################
>>> import PyPDF2

# Open files in Read Binary mode
# Make sure current working directory is where the PDF files are saved
# C:\User\Al\Documents ("My Documents" folder on my computer)
pdf1File = open('meetingminutes1.pdf','rb') 
pdf2File = open('meetingminutes2.pdf','rb') 
  
# Create reader object for each PDF file
# Original 
# reader1 = PyPDF2.PdfFileReader(pdf1File)
# reader2 = PyPDF2.PdfFileReader(pdf2File)	  

# Working 
>>> reader1 = PyPDF2.PdfReader(pdf1File)
>>> reader2 = PyPDF2.PdfReader(pdf2File)
  
# Create writer object of blank PDF  
# Original
# writer = PyPDF2.PdfFileWriter()

# Working 
writer = PyPDF2.PdfWriter()

# Loop through all of the pages
# Loop through all the pages in reader1 and add to new PDF document
# Then loop through all of the pages in reader2 and add to end of document 

# Original 
# for pageNum in range(reader1.numPages):
	  # page = reader1.getPage(pageNum)
	  # writer.addPage(page)
# for pageNum in range(reader2.numPages):
	  # page = reader2.getPage(pageNum)
	  # writer.addPage(page)

# Working  
>>> for pageNum in range(len(reader1.pages)):
    page = reader1.pages[pageNum]
    writer.add_page(page)

>>> for pageNum in range(len(reader2.pages)):
    page = reader2.pages[pageNum]
    writer.add_page(page)

# These are just objects that exist in Python's memory, for your program
# Have to actually save to a file on the hard drive
# Just like we had to open file in Read binary mode to read the original files in  
# Have to open output file in Write Binary mode before you can write to it 
>>> outputFile = open('combinedminutes.pdf','wb') 

# Then write to file 
>>> writer.write(outputFile)

# Clean up
>>> outputFile.close()
>>> pdf1File.close()
>>> pd2File.close()