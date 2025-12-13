################### SELENIUM EXAMPLE 1 ###################
# How Selenium can be used to interact with the browser's web pages and interact with the browser itself
from selenium import webdriver
browser = webdriver.Firefox()

# Can now control by calling methods on the browser object
browser.get('https://automatetheboringstuff.com/')

# Python code will now control the browser

# Use CSS selectors to find elements on page 
# Right-click 'Introduction' at the bottom of the webpage and select "Inspect Element"
# Right-click element and select "Copy Unique Selector" 

# Selenium's find_element_by_css_selector() is kind of like Beautiful Soup's select()
elem = browser.find_element_by_css_selector('body > div > main > div > ul:nth-child(19) > li:nth-child(1) > a')

# Now have a web element object stored in the elem variable
elem 

# Once you have this element object, that represents a single element on the webpage
# Can call the click method to simulate "clicking" on that link/element on the web browser 
elem.click()

# Can specify a more general CSS selector that will match multiple elements
# Then call find_elements_by_css_selector() 
elems = browser.find_elements_by_css_selector('p') # get all the paragraph elements from HTML page
len(elems)

# To select a search field 
searchElem = browser.find_element_by_css_selector('.search-field')

# Pass any string that will be typed into that field
searchElem.send_keys('zophie')

# Then submit
searchElem.submit()

# Control the browser itself
browser.back()
browser.forward()
browser.refresh()
browser.quit()

################### SELENIUM EXAMPLE 2 ###################
# How Python scripts can use Selenium to read content of webpages
# Open new browser
browser.get('https://automatetheboringstuff.com/')

# Copy new CSS Selector
elem = browser.find_elements_by_css_selector('.entry-content>p:nth-child(4)')

# All web elements have a text member variable that contains a string of the text inside that element
elem.text 

# Entire text for the website
# <html> or <body> element 
# If you press CTRL + U you'll see html is at the very top 
elem = browser.find_elements_by_css_selector('html')