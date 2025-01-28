import FreeSimpleGUI as sg
from zip_creator import make_archive  # since file contains one function only

label1 = sg.Text("Select files to compress:")
input1 = sg.Input()
choose_button1 = sg.FilesBrowse("Choose", key="files") # add key argument so makes more sense than "Choose" (label of the widget Python uses as a key)

label2 = sg.Text("Select destination folder:")
input2 = sg.Input()
choose_button2 = sg.FolderBrowse("Choose", key="folder") # add key argument so it makes more sense than "Choose0" (label of the widget Python uses as a key)

compress_button = sg.Button("Compress")
output_label = sg.Text(key="output", text_color="green") # key is the identifier for this widget

window = sg.Window("File Compressor",
                   layout=[[label1, input1, choose_button1],
                           [label2, input2,choose_button2],
                           [compress_button, output_label]])
while True:
    event,values = window.read()
    print(event,values)
    filepaths = values["files"].split(";")
    folder = values["folder"]
    make_archive(filepaths,folder)
    window["output"].update(value="Compression completed")

window.close()



