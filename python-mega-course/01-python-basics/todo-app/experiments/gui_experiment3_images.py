
import functions
import FreeSimpleGUI as sg
import time

# https://docs.pysimplegui.com/en/latest/documentation/module/themes/
sg.theme("Black")

# Create instances of different types
clock = sg.Text('',key='clock') # don't use time because we're using the time module, don't use variable namess that are same as keywords, objects, functions ,etc
label = sg.Text("Type in a to-do")
input_box = sg.InputText(tooltip="Enter todo", key="todo")
add_button = sg.Button(size=2, image_source="add.png",
                       mouseover_colors='LightBlue2',  # https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Color_Names.py
                       tooltip = "Add Todo",
                       key="Add" )
list_box = sg.Listbox(values=functions.get_todos(), key='todos',  # don't use the same key for input_box above
                      enable_events=True, size=[45,10])
edit_button = sg.BUtton("Edit")
complete_button = sg.Button("Complete")
exit_button = sg.Button("Exit")

window = sg.Window('My To-Do App',
                   layout=[[clock],
                           [label],
                           [input_box, add_button],
                           [list_box,edit_button,complete_button],
                           [exit_button]],
                   font=('Helvetica',20))

# Once we create the objects, display them
# Keeps the window open by reading it over and over again
# Every event that happens creates another run of the loop
while True:
    event,values = window.read(timeout=200) # have to add timeout to execute every N milliseconds otherwise loop happens whenver an event happens
    window["clock"].update(value=time.strftime("%b %d, %Y %H:%M:%S"))
    # print(event)
    # print(values)
    # print(values['todos'])
    match event:
        case "Add":
            todos = functions.get_todos()
            new_todo = values['todo'] + "\n"
            todos.append(new_todo)
            functions.write_todos(todos)
            window['todos'].update(values=todos)
        case "Edit":
            try:
                todo_to_edit = values['todos'][0]
                new_todo = values['todo']

                todos = functions.get_todos()
                index = todos.index(todo_to_edit)
                todos[index] = new_todo
                functions.write_todos(todos)
                window['todos'].update(values=todos)
            except IndexError: # paste error you get
                # print("Please select an item first.") # user will see the GUI not the command line interface
                sg.popup("Please select an item first.", font=("Helvetica",20)) #  should show in GUI or popup window
        case "Complete":
            try:
                todo_to_complete = values['todos'][0]
                todos = functions.get_todos()
                todos.remove(todo_to_complete)
                functions.write_todos(todos)
                window['todos'].update(values=todos)
                window['todo'].update(value='')
            except IndexError:
                sg.popup("Please select an item first.", font=("Helvetica", 20))
        case "Exit":
            break
        case 'todos':
            window['todo'].update(value=values['todos'])
        case sg.WIN_CLOSED: # variable defined in PySimpleGUI files
            break

# Print line shows it waits for user action up to this line
# Then will execute from here down
# print("HelLo")

# Close the program
window.close()


