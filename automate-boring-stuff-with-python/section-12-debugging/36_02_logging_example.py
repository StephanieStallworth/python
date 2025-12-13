################### Logging Example: example.py #############################
########## CONFIGURATION ########## 
import logging
# Configure logging to display on screen
# Remember to specify log level you want here! 
# Just CONFIGURING message, do not write actual message here 
# logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s')

# Configure log messages to a WRITE TO FILE
logging.basicConfig(filename='myProgramLog.txt', level=logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s')

########## TURN OFF (OPTIONAL) ########## 
# Turn off logging at a specific level or lower  
# logging.disable(logging.CRITICAL) # Turn off CRITICAL level log messages and below, CRITICAL is the highest so disables all logging messages
# logging.disable(logging.DEBUG) # Turn off DEBUG log messages and below
# logging.disable(logging.WARNING) # Turn off WARNING log level messages and below

########## CALL LOGGING FUNCTION ########## 
# Can call the different logging functions based on the priority of that message
# logging.info('Put log message here')
logging.debug('Start of program')

# Write output of fuction/method
# https://stackoverflow.com/questions/42515493/write-or-log-print-output-of-pandas-dataframe
# logging.debug('Column Check - {}'.format(str(dat_offers.columns)))

logging.debug((str(dat_offers.columns)))

def factorial(n):
    logging.debug('Start of factorial (%s)' % (n))
    total = 1
    for i in range(1, n + 1):
        total *= i
        logging.info('i is %s, total is %s' % (i,total)) # Slightly higher level

    logging.debug('Return value is %s' % (total))
    return total

# Print out an example
print(factorial(5))

logging.debug('End of program')