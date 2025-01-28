feet_inches = input("Enter feet and inches: ")

# This function is doing to much
# Parsing the string then converting it to meters
# However, the name of the function is "convert"
# Functions should do one thing and do it well
# Solution is decoupling by creating two different functions
# def convert(feet_inches):
#     parts = feet_inches.split(" ")
#     feet = float(parts[0])
#     inches = float(parts[1])
#
#     meters = feet * 0.3048 + inches * 0.0254
#
#     # Instead of this
#     # return f"{feet} feet and {inches} inches is equal to {meters} meters."
#
#     # Return single value
#     return meters

def parse(feet_inches):
    parts = feet_inches.split(" ")
    feet = float(parts[0])
    inches = float(parts[1])
    # return feet, inches # make these numbers available to be accessed by other components of the script
    return {"feet": feet, "inches":inches} # improve this further by using a dictionary

def convert(feet,inches):

    meters = feet * 0.3048 + inches * 0.0254
    # Instead of this
    # return f"{feet} feet and {inches} inches is equal to {meters} meters."

    # Return single value
    return meters

parsed = parse(feet_inches)

result = convert(parsed['feet'], parsed['inches'])

print(f"{parsed['feet']} feet and {parsed['inches']} is equal to {result}")

if result < 1:
    print("Kid is too small.")
else:
    print("Kid can use the slide.")

