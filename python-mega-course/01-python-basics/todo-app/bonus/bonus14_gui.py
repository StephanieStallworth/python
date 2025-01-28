# Could have put both functions in the same file
# But using two files because the functions are different from each other
# Foresee having other parser-type functions later on, allows you extend the program easier


# import statements added by PyCharm automatically
# folderName.moduleScript import functionName
from bonus.converters14 import convert
from bonus.parsers14 import parse

# Could have also done this: from parsers14 import parse
# Because parsers14 and bonus14 scripts are in the same directory
# And 'bonus' directory is located in the PyCharm project directory which is also included in the namespace

feet_inches = input("Enter feet and inches: ")

parsed = parse(feet_inches)

result = convert(parsed['feet'], parsed['inches'])

print(f"{parsed['feet']} feet and {parsed['inches']} is equal to {result}")

if result < 1:
    print("Kid is too small.")
else:
    print("Kid can use the slide.")