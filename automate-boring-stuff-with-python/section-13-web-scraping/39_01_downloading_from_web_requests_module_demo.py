############## get() function ##############
import requests
# Download file
>>> res = requests.get('https://automatetheboringstuff.com/files/rj.txt')

# Check status of download
# 200 = everything went ok  
>>> res.status_code
# 200

# If the request succeeded the downloaded web page is stored as a string in the response object's text variable
# Large string of the entire play
>>> len(res.text) 
# 178978

# Slice just the first 500 lines 
>>> print(res.text[:500])
# The Project Gutenberg EBook of Romeo and Juliet, by William Shakespeare

# This eBook is for the use of anyone anywhere at no cost and with

# almost no restrictions whatsoever.  You may copy it, give it away or

# re-use it under the terms of the Project Gutenberg License included

# with this eBook or online at www.gutenberg.org/license

# Title: Romeo and Juliet

# Author: William Shakespeare

# Posting Date: May 25, 2012 [EBook #1112]

# Release Date: November, 1997  [Etext #1112]

# Language: Eng

############## `raise_for_status()` ##############
# raise_for_status() raises an exception if there was an error downloading the file  
# Simpler way to check for success 
# Nothing happens if download was successful
>>> res.raise_for_status()

# Bad request get back an exception    
>>> badRes = requests.get('http://automatetheboringstuff.com/a;sldfjkad;sfj')
>>> badRes.raise_for_status()

# Traceback (most recent call last):
#   File "<pyshell#12>", line 1, in <module>
#     badRes.raise_for_status()
#   File "C:\Users\sstallworth\AppData\Local\Programs\Python\Python37-32\lib\site-packages\requests\models.py", line 953, in raise_for_status
#     raise HTTPError(http_error_msg, response=self)
# requests.exceptions.HTTPError: 404 Client Error: Not Found for url: http://automatetheboringstuff.com/a;sldfjkad;sfj

############## `iter_content()` method ##############
# Save the webpage to a file on your hard drive
# Must open the file in write-binary mode
>>> playFile = open('RoemoandJuliet.txt','wb')

# Then write a for-loop with the response object's iter_content() method
>>> for chunk in res.iter_content(1000000):
	playFile.write(chunk) # Each chunk is just a piece of that downloaded file in the response object
# 178978

>>> playFile.close()