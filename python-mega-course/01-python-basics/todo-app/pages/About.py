########## Experiment 4: Muiltiple pages for web apps ##########
# Go to root directory of your project
# Create new folder "pages" and add Python files in this folder
# Then refresh browser to see page in sidebar (reflecting names of the Python files)

# Rename web.py to Home.py to reflect in the app
# But the first time you refresh it, you've lost the execution of web.py
# Press CTRL+ C to interupt and change command to : streamlit run Home.py
# Because this is the entry point of the app now

import streamlit as st

st.write("Hello")