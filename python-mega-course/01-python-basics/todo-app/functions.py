# Define variable
# Benefit is if we decide to change the filename later, can just change it here
# Have to close Python console to refresh the session and see the FILEPATH variable in dir(functions)
FILEPATH = "todos.txt"

# Default parameter only
def get_todos(filepath=FILEPATH):
    """Read a text file and return the list of
    to-do items.
    """
    with open(filepath,'r') as file_local:
        todos_local = file_local.readlines()
    return todos_local

# Default parameter with non-default parameter
def write_todos(todos_arg,file_path=FILEPATH): # non-default parameters have to come before default parameters
    """Write the to-do items list in the text file."""
    with open(file_path, 'w') as file:
        file.writelines(todos_arg)

if __name__== "__main__":
    print("Hello from functions")
    print(get_todos())