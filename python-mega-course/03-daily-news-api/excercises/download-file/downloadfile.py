import requests

# Right click image > Open image in New Tab > Copy URL in address bar
# Make sure ends with image extension ".jpg" (or ".png")
# Same technique can be used for any file
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Cat_August_2010-4.jpg/362px-Cat_August_2010-4.jpg"

# Make a "response" object because the get() method gives us a response
# <Response [200]> is an HTTP code which means the request was successful
response = requests.get(url)

# text property only works when you are loading text data such as webpages: response.text
# Image is not text, need to use content() method: response.content()
# Don't see the actual image, see the raw code that makes up the image (bytes data)

# Take this raw code and write to a file
with open("image.jpg","wb") as file: # need to open in "write binary" mode
    file.write(response.content) # points to bytes data of URL
