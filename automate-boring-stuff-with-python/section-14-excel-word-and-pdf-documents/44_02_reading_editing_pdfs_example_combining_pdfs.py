########################## Create PDFs By Combining PDFs (Depreciated Version) ##########################
# Creating a PDF 
# pip install PyPDF2							
import pyPDF2							
import os							
os.chdir('c:\\users\\al\\documents')  

pdfFile = open('meetingminutes1.pdf','rb') # `rb` is read binary mode  
reader  = PyPDF2.PdfFileReader(pdfFile)  

reader.numPages							
page = reader.getPage(0)							
page.extractText()  

for pageNum in range(reader.numPages):  
  print(reader.getPage(pageNum).extractText())  
  
# Combining PDFs							
import PyPDF2							
pdf1File = open('eetingminutes1.pdf', 'rb')  
pdf2File = open('meetingminutes2.pdf,'rb')  

reader1 = PyPDF2.PdfFileRader(pdf1File)	  
reader2 = PyPDF2.PdfFileRader(pdf2File)  

writer = PyPDF2.PdfFileWriter()  

for pageNum in range(reader1.numPages):  
   page = reader1.getPage(pageNum)
   writer.addPage(page)							

for pageNum in range(reader2.numPages):	  
    page = reader2.getPage(pageNum)  
    writer.addPage(page)   
    
outputFile = open('combinedminutes.pdf', 'wb')  

writer.write(outputFile)
outputFile.close()
pdf1File.close()
pdf2File.close()  