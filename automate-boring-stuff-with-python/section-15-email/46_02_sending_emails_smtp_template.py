# Sending Emails With SMTP Template 
import smtplib							
conn = smtplib.SMTP('smtp.gmail.com',587) # port of that server		
conn.ehlo() # connects to server							
conn.starttls() # begins encryption							
connlogin('user@email.com','password1234') # username and password		
conn.sendmail('user@email.com','user@email.com','Subject: Hello…\n\nDear User,\nHello.\n\n-Al')							
conn.quit()  