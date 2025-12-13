#################### Updating Word Documents ####################
# Create a document object 
>>> import docx
>>> d = docx.Document('c:\\users\\sstallworth\\Documents\\demo.docx')

# Document objects have a `paragraphs` member variable that contains a list of paragraphs objects
>>> d.paragraphs
# [<docx.text.paragraph.Paragraph object at 0x03F46A50>, <docx.text.paragraph.Paragraph object at 0x03F46DB0>, <docx.text.paragraph.Paragraph object at 0x03F46B10>, <docx.text.paragraph.Paragraph object at 0x03F46CB0>, <docx.text.paragraph.Paragraph object at 0x03F46B50>, <docx.text.paragraph.Paragraph object at 0x03F46BB0>, <docx.text.paragraph.Paragraph object at 0x03F46BF0>]

# Look at the first paragraph object
>>> d.paragraphs[0]
# <docx.text.paragraph.Paragraph object at 0x03F469D0>

# Each of these paragraph objects has a text member `variable` containing a string of the text inside the paragraph
# Text of first paragraph 
>>> d.paragraphs[0].text
# 'Document Title'

# Text of second paragraph
>>> d.paragraphs[1].text 
# 'A plain paragraph having some bold and some italic.'

# Each paragraph object contains one or more runs in a member variable called `run`, this is a list of run objects
# A new run starts whenever there is a change in the style
>>> p = d.paragraphs[1]
>>> p.runs
# [<docx.text.run.Run object at 0x03F46AF0>, <docx.text.run.Run object at 0x03F46A10>, <docx.text.run.Run object at 0x03F469D0>, <docx.text.run.Run object at 0x03F58490>]

# Run objects also have a `text` member variable
>>> p.runs[0].text
# 'A plain paragraph having some '

>>> p.runs[1].text
# 'bold'

>>> p.runs[2].text
# ' and some '

>>> p.runs[3].text
# 'italic.'

# Each of these run objects also have `bold`, `italic`, and `underline` member variables
# Returns True if it is formatted that way 
>>> p.runs[1].bold
# True

# Returns None type if not formatted that way 
>>> p.runs[0].bold
>>> p.runs[0].bold == None
# True

>>> p.runs[3].italic
# True

# Can changes these `run` member variables to whatever we want
>>> p.runs[3].underline = True
>>> p.runs[3].text = 'italic and underlined'

# Take the document object and call its save method
>>> d.save('c:\\users\\sstallworth\\Documents\\demo2.docx')

# `CTRL + ALT + SHIFT + S` in Word brings up the styles 
# Every paragraph and run as its own style
# Word documents come with several of these built in styles

# Paragraph and run objects also have a `style` member variable that is set to the string of the style name
>>> # Set entire paragraph to Title Style
>>> p.style = 'Title'

# Save update to file 
>>> d.save('c:\\users\\sstallworth\\Documents\\demo3.docx')

#################### Creating Word Documents ####################
# Create new Word document object 
# This is nowhere on the hard drive, it only exists inside the Python program
>>> d = docx.Document()

# All document objects also have an add_paragraph() method
# Paragraph that gets created is added to the document object
# Object is also returned for convenience
>>> d.add_paragraph('Hello this is a paragraph.')
# <docx.text.paragraph.Paragraph object at 0x00AF75B0>

# Create another paragraph 
>>> d.add_paragraph('This is another paragraph')
# <docx.text.paragraph.Paragraph object at 0x00AF7D90>

# Save to hard drive
>>> d.save('c:\\users\\sstallworth\\Documents\\demo4.docx')

# Add more text to the first paragraph
>>> p = d.paragraphs[0]

# Just like document objects have an add_paragraph() method
# Paragraph objects have an add_run() method
>>> p.add_run('This is a new run')
# <docx.text.run.Run object at 0x03F46BF0>

# Can see there are now two run objects inside the paragraph object
>>> p.runs
# [<docx.text.run.Run object at 0x00B131D0>, <docx.text.run.Run object at 0x00B13250>]

# Grab the second one and set its bold to True
>>> p.runs[1].bold = True

# Save
>>> d.save('c:\\users\\sstallworth\\Documents\\demo5.docx')