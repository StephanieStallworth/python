################### Beautiful Soup Demo #####################
# Code to download a website and find price information for us 
>>> import bs4
>>> import requests

# Amazon blocks scripts from scraping their site 
# You will need to change the user agent when making a request 
# Simply paste this variable into your code and make a second argument in the `request.get()`, this code will make Amazon think you're a user coming from a browser 
>>> headers = {
     'User-Agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
     }

>>> link = 'https://www.amazon.com/Automate-Boring-Stuff-Python-Programming/dp/1593275994'
>>> res = requests.get(link, headers=headers)
>>> res.raise_for_status()

# Returns a Beautiful Soup object
# Soup Object is now ready to find HTML elements in the webpage that we downloaded
soup = bs4.BeautifulSoup(res.text) 

# To hide warnings, tells BS4 "yes we want to parse HTML"
soup = bs4.BeautifulSoup(res.text,'html.parser')

# Next find the CSS Selector
# CSS Selectors are like Regular Expression syntax, but answers the question of "how do I specify a particular part of the HTML document that I want to look at?"

# To find the CSS Selector:
# Right-click price information on webpage and select "Inspect element"
# Then right click on the element and select "Copy selector" or "Copy CSS path"

# Pass CSS Selector path to the `soup.select()` method
# select() returns a list of matching elements for the CSS Selector 
# Since we passed it the unique selector, should only contain one element 
soup.select('#corePriceDisplay_desktop_feature_div > div.a-section.a-spacing-none.aok-align-center.aok-relative > span.a-price.aok-align-center.reinventPricePriceToPayMargin.priceToPay > span:nth-child(2)')

elems = soup.select('#corePriceDisplay_desktop_feature_div > div.a-section.a-spacing-none.aok-align-center.aok-relative > span.a-price.aok-align-center.reinventPricePriceToPayMargin.priceToPay > span:nth-child(2) > span.a-price-symbol')

# Just like a response object, element objects also have a member variable called `text` that contains their content 
# The `text` variable holds a string value of the text that holds the HTML element 
# Only text of the first matching element 
elems[0].text

# Kind of messy, clean it up with the strip() string method
elems[0].text.strip()