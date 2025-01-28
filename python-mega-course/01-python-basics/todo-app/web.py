import streamlit as st
import functions

todos = functions.get_todos()

# The session state type is a specific object type streamlit
# Contains pairs of data the user enters in your app (looks like a dictionary)
# Contains widget information
def add_todo():
    todo = st.session_state["new_todo"] + "\n"
    todos.append(todo)
    functions.write_todos(todos)

# The order of components matters
# Just need to refresh page when adding them, don't need to re-run script from the command line to view changes
st.title("My Todo App")
st.subheader("This is my todo app.")
st.write("This app is to increase your productivity")

# Use enumerate function to get index of todo item to remove
for index, todo in enumerate(todos):
    checkbox = st.checkbox(todo, key=todo) # need to add a key to the check boxes so session_state will know
    if checkbox:
        todos.pop(index)
        functions.write_todos(todos)
        del st.session_state[todo]
        st.experimental_rerun() # needed for checkbox, probably just need to do st.rerun()

st.text_input(label="", placeholder="Add new todo...",
              one_change=add_todo, key='new_todo')

print("Hello")

st.session_state