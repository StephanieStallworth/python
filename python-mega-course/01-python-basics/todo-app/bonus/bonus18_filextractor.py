# import PySimpleGUI as sg
import FreeSimpleGUI as sg
from zip_extractor import extract_archive

sg.theme("Black")

# Widgets
label1 = sg.Text("Select archive:")
input1 = sg.Input()
choose_button1 = sg.FileBrowse("Choose",key="archive") # user selects one file only

label2 = sg.Text("Select dest dir:")
input2 = sg.Input()
choose_button2 = sg.FolderBrowse("Choose",key="folder")

extract_button = sg.Button("Extract")
output_label = sg.Text(key="output", text_color="green")

# Window instances with all the widgets
window = sg.Window("Archive Extractor",
                   layout=[[label1, input1, choose_button1],
                           [label2, input2, choose_button2],
                           [extract_button, output_label]])

# Connecting frontend and backend
while True:
        event, values = window.read()
        print(event,values)
        archivepath=values["archive"]
        dest_dir = values["folder"]
        extract_archive(archivepath, dest_dir) # backend function that is imported
        window["output"].update(value="Extraction Completed")

window.close()