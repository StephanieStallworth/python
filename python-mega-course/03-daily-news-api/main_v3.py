import requests
from send_email import send_email

##### API #####
# Instead of passing URL of webpage
# url= "https://finance.yahoo.com"

# Improvement 5: Control category
# Fix hardcoded values to make more dynamic using f-strings
# Note: String is spread across multiple lines, the "f" in an f-string should on line where the placeholder is located
topic = "tesla"

# Pass in URL of API
# Sign up for account at newsapi.org to API key
api_key = '890603a55bfa47048e4490069ebee18c'

# Copy and paste URL of API you want
# Please change the part after "apiKey=" to reflect your own API which you will get from newsapi.org after creating a free account there.
url = "https://newsapi.org/v2/everything?"\      
        f"q={topic}&"\ 
        "sortBy=publishedAt&"\
        "apiKey=890603a55bfa47048e4490069ebee18c&"\
        "language=en" # Improvement 2: Specify language

##### Make request #####
# Create request object type
request = requests.get(url)

##### Get text property #####
# The text property returns what looks like a dictionary but is actually a plain string
# content = request.text

##### Get dictionary with data #####
# Instead need a data structure that is more flexible such as dictionary or list
content = request.json()
print(type(content))

# Print out the text
# Same content if looked at the page from the browser
# Difference is we can make use of these data since we're working with a programming language
# print(content)

##### Access the article titles and description #####
body = ""
for article in content["articles"][:20]: # Improvement 3: Limits news
        if article["title"] is not None:
                body = "Subject: Today's news"\ # Improvement 4: Add subject to email
                        + "\n" + body + article["title"] + "\n" \
                       + article["description"] \
                       + "\n" + article["url"] + 2*"\n" # Improvement 1: Add links

body = body.encode("utf-8")
send_email(message=body)
