import pandas

df = pandas.read_csv("hotels.csv", dtype={"id": str})
df_cards = pandas.read_csv("cards.csv", dtype=str).to_dict(orient="records")
df_cards_security = pandas.read_csv("card_security.csv", dtype=str)

##### 1. Class definitions #####

# Don't need anymore this class anymore
# class User:
#     def view_hotels(selfself):
#         pass


class Hotel:
    #### 3A. Implement Methods #####
    def __init__(self,hotel_id): # id is a occupied name, need to change to something else
        self.hotel_id = hotel_id
        self.name = df.loc[df["id"] == self.hotel_id, "name"].squeeze()  # using hotel ID inputed by the user to find the hotel name and "squeeze" to extract the actual string

    def book(self):
        """Books a hotel by changing its availability to no"""
        df.loc[df["id"] == self.hotel_id, "available"] = "no"
        df.to_csv("hotels.csv", index=False) # Update actual file on disk

    def available(self):
        """Check if the hotel is available"""
        availability = df.loc[df["id"] == self.hotel_id, "available"].squeeze()
        if availability == "yes":
            return True
        else:
            return False
        pass


class ReservationTicket:
    #### 3B. Implement Methods #####
    def __init__(self, customer_name, hotel_object):

        # "customer_name" and "hotel_object" are local variables of the __init__ method
        # Need to make them attributes of the ReservationTicket instance by using "self"
        self.customer_name = customer_name # whatever user passes when they instantiate the class below
        self.hotel = hotel_object

    def generate(self):
        # pass
        # content = f"Name of the customer hotel"
        content = f"""
        Thank you for your reservation! 
        Here are your booking data
        Name: {self.customer_name} 
        Hotel name: {self.hotel.name}
        """
        return content

# When defining parameters think... what is the minimum set of parameters you need to instantiate the class or call the method?
class CreditCard:
    def __init__(self, number): # just need credit card to instatiate the CreditCard class, but some programmers prefer to put all the parameters here
        self.number = number

    def validate(self,expiration, holder, cvc): # but can't validate with just cc number, need these parameters too
        card_data = {"number": self.number, "expiration": expiration,
                     "holder": holder, "cvc":cvc}
        print("Validation Data:", card_data)
        if card_data in df_cards:
            return True
        # Adding this to be more explicit, technically don't need this part as it will just return None if condition is False
        else:
            return False

    # Can extend with feature to pay with credit
    # Add "balance" column to cards.csv
    # def pay(self):
    #     pass

##### Child Class #####
# Inherit everything from a parent class (including its __init__ method)
# This child will be more advanced because it will have other methods in addition
class SecureCreditCard(CreditCard):
    def authenticate(self, given_password):
        password = df_cards_security.loc[df_cards_security["number"] == self.number, "password"].squeeze()
        if password == given_password:
            return True
        else:
            return False

##### 2. Program Main Loop #####

# Not interacting with classes yet
print(df)
hotel_id = input("Enter the id of the hotel: ")

# Create Hotel Instance
hotel = Hotel(hotel_id)
if hotel.available(): # new method we just realized we need, add to Hotel class above
    # credit_card = CreditCard(number="123456789123456")
    # Made sure card number in cards.csv match card_security.csv!!!!
    credit_card = SecureCreditCard(number="123456789123456") # More advanced child class that inherited CreditCard class
    if credit_card.validate(expiration="12/26", holder="JOHN SMITH", cvc="123"):
        if credit_card.authenticate(given_password="mypass"):
            hotel.book()
            name = input("Enter your name: ") #"name" is an instance of the string class
            # Create instance of ReservationTicket with name  then point to instance and call methods on it
            # Hierarchy of objects and attributes
            # ReservationTicket instance class --> has a Hotel Class instance as a attribute --> which has a string instance as an attribute
            reservation_ticket = ReservationTicket(customer_name=name # need to add these as parameters to the class __init__ method
                                                   , hotel_object=hotel # ReservationTicket instance has an instance of the Hotel class as an attribute
                                                   )
            print(reservation_ticket.generate())
        else:
            print("Credit card authentication failed.")
    else:
        print("There was a problem with your payment")
else:
    print("Hotel is not free.")