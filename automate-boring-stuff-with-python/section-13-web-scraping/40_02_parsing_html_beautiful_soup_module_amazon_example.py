############### Amazon Price example.py ###############
# Code might fail if there are changes to the webpage
# Would have to figure out the new CSS selector for that new web page
# Try adding a try and except block to gracefully handle errors that come up
# Or put a debugger breakpoint to see exactly how it failed

# Other web scraping ideas: www.weather.gov, https://xkcd.com

import bs4, requests

def getAmazonPrice(productURL):
    res = requests.get(productURL) # Download page
    res.raise_for_status() # Bad download, will raise an exception and crash our program

    # Create Soup Object by passing it the HTML text we downloaded
    # Also pass `html.parser` as the second argument to avoid ugly error message
    soup = bs4.BeautifulSoup(res.text, 'html.parser')

    # Pass the CSS selector to the select() method 
    # To find CSS selector, right-click price and select "Inspect element"
    # Then right-click on the element and select "Copy CSS path"
    elems = soup.select('#corePriceDisplay_desktop_feature_div > div.a-section.a-spacing-none.aok-align-center.aok-relative > span.a-price.aok-align-center.reinventPricePriceToPayMargin.priceToPay > span:nth-child(2)')

    # Returns a list matching elements for the CSS Selector
    # Only want the first one
    # Just like response object, element object will also have a member variable called `text` that contains their content 
    return elems[0].text.strip() 
  
price = getAmazonPrice('https://www.amazon.com/Automate-Boring-Stuff-Python-Programming/dp/1593275994')
print('The price is' + price)
