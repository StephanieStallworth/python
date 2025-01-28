import streamlit as st
from send_email import send_email

st.header("Contact Me")

with st.form(key="email_forms"): # some components like this, key are mandatory
    user_email = st.text_input("Your email address")
    raw_message = st.text_area("Your Message")
    message=f"""\
    Subject: New email from {user_email}
    From: {user_email}
    {raw_message}
    """
    message = message + "\n" + user_email
    button = st.form_submit_button("Submit") # special button designed to submit the form it belongs to
    print(button)
    if button:
        # print(button)
        # print("Pressed")
        send_email(message) # "throws to" send_email.py
        st.info("Your email was sent successfully")
