try:
    width = input("Enter rectangle width: ")
    length = float(input("Enter rectangle length: "))

    if width == length:
        exit("That looks like a square.") # interrupt/break the program and print out string

    area = width * length
    print(area)
except ValueError:
    print("Please enter a number.")
    # don't need continue keyword here because we're not in a while loop
