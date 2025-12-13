################### Sending Emails with SMTP ###################
>>> import smtplib

# Create connection object with email service domain name 
>>> conn = smtplib.SMTP('smtp.gmail.com',587)

>>> type(conn)
# <class 'smtplib.SMTP'>

>>> conn
# <smtplib.SMTP object at 0x03FF9D30>

# Connect to SMTP server 
# Returns a code, anything that begins with `200` means it's connected
# Also returns values of the `bytes` data type, looks like strings but begin with a "b" when typed as code
>>> conn.ehlo()
# (250, b'smtp.gmail.com at your service. [67.180.35.169]\nSIZE 35882577\n8BITMIME\nSTARTTLS
# nENHANCEDSTATUSCODES\nPIPELINING\nCHUNKING\nSMTPUTF8')

# Begin encryption 
# Most email servers require you start encryption before you can log in 
>>> conn.starttls()
# (220, b'2.0.0 Ready to start TLS')

# Call login() method, passing in username and password as strings
>>> conn.login('aweigar@gmail.com,'klxkjfvcrlxxiinmj')
# (235, b'2.7.0 Accepted')

# Send email
# Returns dictionary of all the emails that it failed to send 
# Blank dictionary means it was all sent correctly 
>>> conn.sendmail('asweigart@gmail.com','asweigar@gmail.com', 'Subject: So long...\n\n\Dear Al,\nSo long and thanks for the all the fish.\n\n-Al')
# {} 

# Disconnect from SMTP server 
>>> conn.quit()