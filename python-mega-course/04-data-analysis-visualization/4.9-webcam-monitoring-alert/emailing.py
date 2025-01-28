import smtplib
import imghdr
import email.message import EmailMessage

PASSWORD = "nfsfpzlmxkaggmtj"
SENDER = "app8flask@gmail.com"
RECEIVER = "app8flask@gmail.com"

def send_email(image_path):
    # Add print statement to better understand what thread is doing when it calls the function
    print("send_email function started")

    # Create EmailMessage object
    email_message = EmailMessage()

    # EmailMessage object behaves like a dictionary
    email_message["Subject"] = "New customer showed up!"
    email_message.set_content("Hey, we just saw a new customer!")

    # Attachments
    # Open/create Python file object in "rb" mode (becuase image is a binary file)
    with open(image_path, "rb"):
        content = file.read()
    # Add attachment to EmailMessage object
    email_message.add_attachment(content, maintype="image", subtype=imghdr.what(None, content))

    # Send out email
    # Create SMTP server object
    gmail = smtplib.SMTP("smtp.gmail.com", 587)

    # Routines to start the email server parameters
    gmail.ehlo()
    gmail.starttls()
    gmail.login(SENDER, PASSWORD)
    gmail.sendmail(SENDER, RECEIVER, email_message.as_string())
    gmail.quit()
    print("send_email function ended")

# Test to see if it works as a standalone function first
# Need to turn off VPN, will not work with VPN enabled
if __name__ == "__main__":
    send_email(image_path="images/19.png")
