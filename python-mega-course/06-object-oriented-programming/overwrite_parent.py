class Ticket:
    def __init__(self):
        pass
    def generate(self):
        return "Hello, this is your ticket."

class DigitalTicket(Ticket):
    def download(self):
        pass
    # Overwrite this method from the parent class
    def generate(self):
        return "Hi, this is your ticket"

# Child class
dt = DigitalTicket()
print(dt.generate())

# Only the method in the Child class is modified
# Parent Class itself is not modified at all
t = Ticket()
print(t.generate())