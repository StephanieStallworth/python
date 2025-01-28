# Run webpage by executing command in terminal: run bonus/bonus19-camera.py

import streamlit as st
from PIL import Image

with st.expander("Start Camera:"):
    # Camera hidden under expander component, when expanded...
    # Start the camera
    camera_image = st.camera_input("Camera")  # camera_input is a specific streamlit object

    # If camera image not captured, code will not be executed yet
    if camera_image:
        # Create a pillow image instance
        img = Image.open() # Needs the .open() method to create instances

        # Convert the pillow image to grayscale
        gray_img = img.convert("L")

        # Render the grayscale image on the webpage
        st.image(gray_img)







