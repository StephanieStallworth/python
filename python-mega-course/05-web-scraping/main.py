import requests
import selectorlib
import smtplib, ssl
import os
import time
import sqlite3

URL = "http://programmer100.pythonanywhere.com/tours/"

# Tell the web server that this script behaves like a browser
# Some servers don't like script programs, having headers can sometimes fix this issue
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# Establish connection once then work on top of that connection
connection = sqlite3.connect("data.db")

def scrape(url):
    """Scrape the page source from the URL"""
    response = requests.get(url, headers=HEADERS)
    source = response.text
    return source

def extract(source):
    extractor = selectorlib.Extractor.from_yaml_file("extract.yaml")
    value = extractor.extract(source)["tours"] # has to match extract.yaml
    return value

def send_email():
    host = "smtp.gmail.com"
    port = 465

    username = "app8flask@gmail.com"
    password = "here_goes_your_gmail_password"

    receiver = "app8flask@gmail.com"
    context = ssl.create_default_context()

    with smtplib.SMTP_SSL(host, port, context=context) as server:
        server.login(username, password)
        server.sendmail(username, receiver, message)
    print("Email was sent!")

# Text file version
# def store(extracted):
#     with open("data.txt", "a") as file:
#         file.write(extracted + "\n")

# def read(extracted):
#     with open("data.txt","r") as file:
#         return file.read()

# Database version
def store(extracted):
    row = extracted.split(",")
    row = [item.strip() for item in row]
    cursor = connection.cursor() # have separate cursors for each function
    cursor.execute("INSERT INTO events VALUES(?,?,?)", row) # list of 3 items, not a tuple here

def read(extracted):
    row = extracted.split(",")
    row = [item.strip() for item in row]
    band, city, date = row
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM events WHERE band=? and city=? AND date=?", (band, city, date))
    rows = cursor.fetchall()
    print(rows)

if __name__ == "__main__":
    while True:
        # print(scrape(URL))
        scraped = scrape(URL)
        extracted = extract(scraped)
        print(extracted)
        # content = read(extracted)

        # Store data and send email only if event is new
        if extracted != "No upcoming tours":

            # Check database when script value is different from 'No upcoming tours'
            row = read(extracted)

            # This checks for value in the actual string "data.txt" (not what we want)
            # if not extracted in "data.txt":

            # This checks the actual file
            # if extracted not in content:
            if not row: # if not empty
                store(extracted)
                send_email(message="Hey, new event was found!") # may need to empty data.txt first to get email
            time.sleep(2)