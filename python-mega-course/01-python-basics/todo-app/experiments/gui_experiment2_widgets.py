
import functions

import FreeSimpleGUI as sg

# Create instances of different types
label = sg.Text("Type in a to-do")
input_box = sg.InputText(tooltip="Enter todo")

# Won't do anything when you click the button because it's not connect to any action yet (next video)
add_button = sg.Button("Add")


# Create instance of window type that will be the parent of the instances above
# This mother instance contains the other objects
# Made possible with the "layout" argument, which expects a list of lists
# Lists are not a sequence of numbers and strings
# Lists can contain anything, here each of the lists contain FreeSimpleGUI object instances
# Makes the window instance aware of the other instances created above

# instances in the same list are displayed on one row
# window = sg.Window('My To-Do App', layout=[[label,input_box]])

# Instances in different lists are displayed on separate rows
# window = sg.Window('My To-Do App', layout=[[label],[input_box]])

########## Experiments ##########
# 1. Can't have a flat list of items, need to pass list of lists
window = sg.Window('My To-Do App', layout=[[label, input_box, add_button]])

# 2. Have to be PySimpleGUI widget types not number or text
window = sg.Window('My To-Do App', layout=[["hey", 1, add_button]])

# 3. Placing each widget in its own list, produces a GUI with 3 separate rows
# Each list represents a row in the window
window = sg.Window('My To-Do App', layout=[["hey", 1, add_button]])

################################

# Once we create the objects, display them
window.read()

# Print line shows it waits for user action up to this line
# Then will execute from here down
print("HelLo")

# Close the program
window.close()


