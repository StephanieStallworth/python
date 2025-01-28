import pandas
from abc import ABC, abstractmethod

df = pandas.read_csv("hotels.csv", dtype={"id": str})


class Hotel:
    watermark = "The Real Estate Company"
    def __init__(self, hotel_id):
        self.hotel_id = hotel_id
        self.name = df.loc[df["id"] == self.hotel_id, "name"].squeeze()

    def book(self):
        """Book a hotel by changing its availability to no"""
        df.loc[df["id"] == self.hotel_id, "available"] = "no"
        df.to_csv("hotels.csv", index=False)

    def available(self):
        """Check if the hotel is available"""
        availability = df.loc[df["id"] == self.hotel_id, "available"].squeeze()
        if availability == "yes":
            return True
        else:
            return False

    ##### Class Methods #####
    # Class method that will return length of dataframe
    # Add @classmethod decorator just above the function with no breaklines to make it a class variable
    # Then specify "cls" as the first parameter
    # "cls" could also be called "self" because its just a variable
    # But best practice is to use "cls" as the first parameter for class variables
    # "cls" will be the variable that will hold the CLASS object, not the instance object
    @classmethod
    def get_hotel_count(cls, data):
        return len(data)

    ##### Special Methods #####
    # Treat two instances as being equal
    # Need to refer to the second instance by the "other" variable instead of "self"
    def __eq__(self, other):
        if self.hotel_id == other.hotel_id:
            return True
        else:
            return False

    def __add__(self, other):
        total = self.price + other.price
        return total

##### Abstract Classes #####
# ABC class imported from abc standard module
# Every child inheriting from this class should implement its generate() method
# Basically defining rules, forcing other collaborators to include these methods for every new class they add
class Ticket(ABC):

    @abstractmethod
    def generate(self):
        pass


class ReservationTicket(Ticket): # Inherit Abstract class
    def __init__(self, customer_name, hotel_object):
        self.customer_name = customer_name
        self.hotel = hotel_object

    # def generate(self):
    #     content = f"""
    #     Thank you for your reservation!
    #     Here are you booking data:
    #     Name: {self.customer_name}
    #     Hotel name: {self.hotel.name}
    #     """
    #     return content

    # Use property to get customer name instead
    def generate(self):
        content = f"""
        Thank you for your reservation!
        Here are you booking data:
        Name: {self.the_customer_name}
        Hotel name: {self.hotel.name}
        """
        return content

    ##### Properties #####
    # Looks like a regular instance method, but adding the "@property" decorator make it a property that acts like a variable
    # Name them with nouns instead of verbs as it will make more sense when you call the property later
    # Use a property when you think the the value a instance method returns looks more like an instance variable but the value needs some processing first (by the method)
    # Could technically do this in the class "__init__" method but would be too much for it
    # Makes more sense to create a property and access it it was a normal instance variable
    # Benefit is you keep this code enclosed inside a method, keeps things more organized
    @property
    def the_customer_name(self):
        name = self.customer_name.strip()
        name = name.title()
        return name

    ##### Static Methods #####
    # Static methods are like a function so doesn't have any reference to the class or its instance
    # So don't need "self" or "cls" parameter
    # Utility function to convert the price from Euros to USD
    @staticmethod
    def convert(amount):
        return amount * 1.2

    # Not the best practice, better way is to create an "Abstract Class"
    # From which other normal classes inherit
    # class DigitalTicket(ReservationTicket):
    #     def generate(self):
    #         return "Hello, this is your digital ticket"
    #
    #     def download(self):
    #         pass

    # Create Abstract class above then inherit it here
    class DigitalTicket(Ticket):
        def generate(self):
            return "Hello, this is your digital ticket"

        def download(self):
            pass

# ##### Class Instances #####
# # Create instances of the Hotel class
# hotel1 = Hotel(hotel_id="188")
# hotel2 = Hotel(hotel_id="134")
#
# ##### Instance Variables #####
# # Variables that are specific to THAT particular instance
# print(hotel1.name)
# print(hotel2.name)
#
# ##### Class Variables #####
# # "watermark" is a class variable because value of the variable is shared among ALL instances
#
# # Can access it from an instance
# print(hotel1.watermark)
# print(hotel2.watermark)
#
# # Or access directly from the class
# print(Hotel.watermark)
#
# ##### Instance Methods #####
# # Instance method that checks if THIS particular hotel is available of not
# print(hotel1.available())
#
# ##### Class Methods #####
# # Class methods are not related to a particular instance
# # Can access using the class
# print(Hotel.get_hotel_count(data=df))
#
# # Or access using an instance
# print(hotel1.get_hotel_count(data=df))
#
# ##### Properties #####
# ticket = ReservationTicket(customer_name="john smith", hotel_object=hotel1)
#
# # Defined as a method, but the @property makes it act like a variable
# # Benefit is you don't need to "call" it
# # That's what makes it a property
# print(ticket.the_customer_name)
#
# ##### Static Method #####
# converted = ReservationTicket.convert(10)
# print(converted)



