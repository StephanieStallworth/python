
import functions

import FreeSimpleGUI as sg

# Create instances of different types
label = sg.Text("Type in a to-do")
input_box = sg.InputText(tooltip="Enter todo", key="todo")
add_button = sg.Button("Add")
list_box = sg.Listbox(values=functions.get_todos(), key='todos',  # don't use the same key for input_box above
                      enable_events=True, size=[45,10])
edit_button = sg.BUtton("Edit")

########## Experiment 2: layout argument ##########
# Can pass values directly into layout argument
# window = sg.Window('My To-Do App',
#                    layout=[[label],[input_box, add_button],[list_box,edit_button]],
#                    font=('Helvetica',20))

# Or define in variable first then pass it in to the layout argument
layout = [[label],[input_box, add_button],[list_box,edit_button]]

# Beneficial when you want to dynamically construct the list of widgets
button_labels = ["Close", "Apply"]

layout = []
for bl in button_labels:
    layout.append([sg.Button(bl)])

[[sg.Button("Close")],[sg.button("Apply")]] # Each is a button instance

# # Create instance of window type that will be the parent of the instances above
window = sg.Window('My To-Do App',
                   layout=layout,
                   font=('Helvetica',20))

# Once we create the objects, display them
# Keeps the window open by reading it over and over again
# Every event that happens creates another run of the loop
while True:
    event,values = window.read()
    print(event)
    print(values)
    print(values['todos'])
    match event:
        case "Add":
            todos = functions.get_todos()
            new_todo = values['todo'] + "\n"
            todos.append(new_todo)
            functions.write_todos(todos)
            window['todos'].update(values=todos)
        case "Edit":
            todo_to_edit = values['todos'][0]
            new_todo = values['todo']

            todos = functions.get_todos()
            index = todos.index(todo_to_edit)
            todos[index] = new_todo
            functions.write_todos(todos)
            window['todos'].update(values=todos)
        case 'todos':
            window['todo'].update(value=values['todos'])
        case sg.WIN_CLOSED: # variable defined in PySimpleGUI files
            # break
            exit()

########## Experiment 1: break statement ##########
# exit() stops the program completely
# break statement only stops the while loop from executing again
# Recommended to use break because it makes more sense, allows for better control of program
# Allowed to execute other things between while loop and window.close()

# This line is rinted out with the break statement
# But not printed with exit() because this line is never executed
print("Bye")

# Close the program
window.close()


