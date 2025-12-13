#################### example.py #################################
# Get a single string of all the text inside a Word document

import docx

def getText(filename):
    doc = docx.Document(filename) # open blank document
    fullText = []
    for para in doc.paragraphs: # list of paragraph objects inside of document object
        fullText.append(para.text) # text of each paragraph object
    return '\n'.join(fullText) # single string with all of the text from all the paragraphs

# Test it out
print(getText('c:\\users\\sstallworth\\Documents\\demo.docx'))