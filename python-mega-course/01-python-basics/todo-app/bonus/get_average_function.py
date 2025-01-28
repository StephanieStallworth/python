# Function to get values from a text file and return an output
# Name function with "get"
# So when you can call them, can write a variable that represents the returned value of that function and the function name is like the process
# the naming makes sense this way
# sum() built in function that will sum together elements in a list
# Right click > "Refactor" > "Rename" to change all occurrences of the LOCAL variable within the function (doesn't rename the global variable)
def get_average():
    with open("files/data.txt","r") as file:
        data = file.readlines() # using readlines() to get a list of separate strings, read() will return the entire text as a single string
    values = data[1:] # intermediate variable, better to have reading and slicing as separate steps
    values = [float(i) for i in values] # updating the value of the varible as the updated list

    average_local = sum(values) /len(values)
    return average_local

average = get_average()
print(average)