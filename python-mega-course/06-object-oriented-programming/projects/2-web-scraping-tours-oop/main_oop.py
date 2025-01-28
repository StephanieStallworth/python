import requests
import selectorlib
import smtplib, ssl
import os
import time
import sqlite3

# "INSERT INTO events VALUES ('Tigers', 'Tiger City', '2088.10.14')"
# "SELECT * FROM events WHERE date='2088.10.15'"

URL = "http://programmer100.pythonanywhere.com/tours/"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

# Move into Database class
# connection = sqlite3.connect("data.db")

class Event:
    def scrape(self, url):
        """Scrape the page source from the URL"""
        response = requests.get(url, headers=HEADERS)
        source = response.text
        return source

    def extract(self, source):
        extractor = selectorlib.Extractor.from_yaml_file("extract.yaml")
        value = extractor.extract(source)["tours"]
        return value


class Email:
    # rename so it makes more sense when you call the method on the "email" class object later
    # def send_email(message):
    def send(self, message):
        host = "smtp.gmail.com"
        port = 465

        username = "app8flask@gmail.com"
        password = "qyciukmocfaiarse"

        receiver = "app8flask@gmail.com"
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL(host, port, context=context) as server:
            server.login(username, password)
            server.sendmail(username, receiver, message)
        print("Email was sent!")

class Database:
    # __init__ is a special method that is executed EVERY TIME you create an instance of the class ("instantiate the class")
    # It is called automatically, don't have to call __init__ specifically
    # Whereas other class methods, you have to point to class object and call it explicitly

    # "self" is a special argument/variable inside the class that holds the particular instance of the class the user is creating

    # When you create a class object, they are DIFFERENT instances with different addresses in memory
    # "self" is the variable you use to point to THAT particular class instance
    # Outside of the class, it's not called "self" it's called whatever assigned variable name that holds that instance

    # Inside the __init__ method, we can define other methods to go inside of the class so you can automatically access them when you create a class object

    # Can also assign properties to the "self" variable to access variables when you create a class object
    # "self"  is a special variable that all methods have access to so using it as a middle man to be able to access variables

    # Can also add parameters to the __init__ method definition so you can pass arguments to class methods
    def __init__(self, database_path):

        # Adding "self" to the variable assignment makes connection a property of the self argument
        self.connection = sqlite3.connect(database_path)

    def store(self, extracted):

        row = extracted.split(",")
        row = [item.strip() for item in row]
        cursor = self.connection.cursor() # add "self" to point to the connection property of the self variable
        cursor.execute("INSERT INTO events VALUES(?,?,?)", row)
        self.connection.commit()

    def read(self, extracted):
        row = extracted.split(",")
        row = [item.strip() for item in row]
        band, city, date = row
        cursor = self.connection.cursor()
        cursor.execute("SELECT * FROM events WHERE band=? AND city=? AND date=?", (band, city, date))
        rows = cursor.fetchall()
        print(rows)
        return rows


if __name__ == "__main__":
    while True:

        ##### Event Class #####
        # Create an event instance by calling the class with parenthesis
        # Without the parenthesis it is just the class itself
        event = Event()

        # Need to change this are called because they are not functions anymore in the global namespace
        # scraped = scrape(URL)
        # extracted = extract(scraped)

        # Update to point to the class instance
        scraped = event.scrape(URL)
        extracted = event.extract(scraped)

        print(extracted)

        if extracted != "No upcoming tours":
            # Create instance of class then call methods on the object
            database = Database(database_path="data.db") # may need empty data.db file
            row = database.read(extracted)
            if not row:
                database.store(extracted)

                ##### Email Class #####
                # Create instance of class then call method on the object
                email = Email()
                email.send(message="Hey, new event was found!")
                # send_email(message="Hey, new event was found!")
        time.sleep(2)