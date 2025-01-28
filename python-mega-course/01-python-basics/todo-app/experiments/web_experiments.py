import streamlit as st
import functions

########## Experiment 3: Wide layout argument ##########
# Horizontal length of the webpage expands to the entire width of the browser
st.set_page_config(layout="wide")

todos = functions.get_todos()
def add_todo():
    todo = st.session_state["new_todo"] + "\n"
    todos.append(todo)
    functions.write_todos(todos)

st.title("My Todo App")
st.subheader("This is my todo app.")

########## Experiment 2: Provide HTML instead of a plain string ##########
# HTML only allowed for the .write() method
# title() and subheader() don't have HTML enabled because they're already set to a fixed font

# Note HTML tags are inside the quotes
# To Python its just a string, but streamlit interprets it as HTML
st.write("This app is to increase your <b>productivity</b>." # bold tags
         , unsafe_allow_html=True) # set to false by default

# To render in font, but can just use title for this
# st.write("<h1>This app is to increase your <b>productivity</b>.</h1>"
#          , unsafe_allow_html=True) # set to false by default

########## Experiment 1: Order of the widgets matter ##########
st.text_input(label="", placeholder="Add new todo...",
              one_change=add_todo, key='new_todo')

# Use enumerate function to get index of todo item to remove
for index, todo in enumerate(todos):
    checkbox = st.checkbox(todo, key=todo) # need to add a key to the check boxes so session_state will know
    if checkbox:
        todos.pop(index)
        functions.write_todos(todos)
        del st.session_state[todo]
        st.experimental_rerun() # needed for checkbox, probably just need to do st.rerun()

# Original
# st.text_input(label="", placeholder="Add new todo...",
#               one_change=add_todo, key='new_todo')

print("Hello")

st.session_state

