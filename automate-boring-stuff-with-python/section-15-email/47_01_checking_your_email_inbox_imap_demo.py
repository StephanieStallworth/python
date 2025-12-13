################ Working With Your Email Inbox with IMAP ################

########## LOG INTO EMAIL ##########
>>> import imapclient, pyzmail 

# Connect to IMAP domain name 
# Usually `imap.domain.com` 
# Encrpytion algorithm, tells it we want to use SSL encryption
>>> conn = imapclient.IMAPClient('imap.gmail.com', ssl = True)  

# Login with user credentials 
# By the time you're watching this, I've since changed my password
>>> conn.login('asweigart@gmail.com', 'lxkjfvcrlxxiinmj') 

# Get a bytes object that can basically interpreted like a string 
# b'asweigart@gmail.com authenticated (Success)'

########## CHECK INBOX ##########
# Select a folder, 99% of the time will be INBOX 
# Want to set read-only as True so you don't accidently delete any emails during logged in session
>>> conn.select_folder('INBOX', read-only = True)  
# {b'READ-ONLY':[b''], b'UIDVALIDITY: 596471934', b'EXISTS':12296, b'HIGHESTMODESEQ':6543426, b'RECENT': 0, b'UIDNEXT': 47475, b'FLAGS': (b'\\Answered',b'\\Flagged', b'\\Draft', b'\\Deleted', b'\\Seen', b'$Forwarded',b'$MDNSent', b'$Not Phishing', b'$Phishing', b$label4', b'$label5', b'Junk', b'NonJunk'), b'PERMANENTLFAGS':()}

# Call the connection object's list_folders() method to view all folders
# Will return a list of tuple values, name of folder will be the third item inside that tuple 
>>> conn.list_folders()

########## READ EMAILS ##########
# Find emails based on search criteria 
# Returns list of Unique ID's of particular emails   
>>> uid = conn.search(['SINCE 20-Aug-2015']) 
# [47416,	47417,	47418,	47419,	47420,	47421,	47422,	47423,	47424,	47425,	47426,	47427,	47428,	47429,	47430,	47431,	47432,	47433,	47434,	47435,	47436,	47437,	47438,	47439,	47440,	47441,	47442,	47443,	47444,	47445,	47446,	47447,	47448,	47449,	47450,	47451,	47452,	47453,	47454,	47455,	47456,	47457,	47458,	47459,	47460,	47461,	47462,	47463,	47464,	47465,	47466,	47467,	47468,	47469,	47470,	47471,	47472,	47473,	47474]

# Translate UID into actual email for the connection object
# First argument is the email ID we want 
# Second argument is the part of the email we want, 99% of the time: ['BODY[]','FLAGS']  
>>> rawMessage = conn.fetch([47474],['BODY[]','FLAGS'])  

# Don't want to parse this on our own, use the `pyzmail` module to do this for us  
>>> import pyzmail  

# Returns a Pyz object  
# pyzmail.PyzMessage.factory(rawMessage[47474][b'BODY[]'])  
# <pyzmail.parse.PyzMessage object at 0x00000000038C95F8

# Save to a variable 
>>> message = pyzmail.PyzMessage.factory(rawMessage[47474][b'BODY[]'])  

# Now have we have a message object, can work with it esaily 
>>> message.get_subject()  
# 'So long...'

# Returns a tuple of email sender's name (if set) and their email address
>>> message.get_addresses('from')  
# [('asweigart@gmail.com','asweigart@gmail.com')]

# Returns blank list, may be something weird with the smtplib method calls made earlier 
>>> message.get_addresses('to')  
# []

# Blind Carbon Copy 
>>> message.get_addresses('bcc')  
# [('asweigart@gmail.com','asweigart@gmail.com')]

# Look at the `text_part` member variable
# To see if it was a text email, HTML email, or if it had both text and HTML parts 

# Check if it was an text email 
>>> message.text_part
# MailPart<*text/plain len=56>

# Check if it was an HTML email
# Will return None value, which are is displayed by IDLE 
>>> message.html_part 

# To confirm there is no HTML 
>>> message.html_part == None 
# True

# Get main message of the email as a string
# 99% of the time pass in 'UTF-8'
>>> message.text_part.get_payload().decode('UTF-8')  
# 'Dear Al,\r\nSo long, and thanks for all the fish.\r\n\r\n-Al\r\n'

# If it doesn't work, can try to figure out what encoding the email was sent as 
# This time, it was set to None value but 99% of the time will be'UTF-8'
>>> message.text_part.charset
>>> message.text_part.charset == None
# True

########## DELETE EMAILS ##########
# Select folder but have read-only keyword set to False
# This will allow us to modify the folder 
>>> conn.select_folder('INBOX',readonly = False)

# Search for emails you want to delete based on some criteria 
>>> UIDs = conn.search(['ON 24-Aug-2015'])

# Returns a list of Email IDs
>>> UIDs
# [47467, 47468, 47469, 47470, 47471, 47472, 47473 47474]

# Delete a specific email
>>> conn.delete_messages([47474])

# Pass list itself to delete all emails received on that date
>>> conn.delete_messages(UIDs)