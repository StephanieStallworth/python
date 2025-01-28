import smtplib, ssl
import os

def send_email(message): # "Throws to" message variable at the bottom
    host = "smtp.gmail.com"
    port = 465

    username = "app8flask2@gmail.com"
    password = os.getenv("PASSWORD")

    receiver="app8flask2@gmail.com" # could another email that you own, doesn't have to be the same
    context = ssl.create_default_context() # Create a secure context

    # Comment out because we have message variable in Contact Us.py
    # message ="""\ # Need backslash here, without it implying there is a breakline which should not be with subject line
    # Subject: Hi!
    # How are you?
    # Bye!
    # """

    with smtplib.SMTP_SSL(host, port, context=context) as server:
        server.login(username, password)
        server.sendmail(username, receiver, message)