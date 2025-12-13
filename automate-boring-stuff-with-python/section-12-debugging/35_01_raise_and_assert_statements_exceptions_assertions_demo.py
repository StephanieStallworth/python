##########################################################################
####################### EXCEPTIONS #######################################
##########################################################################

############### Exceptions ###############
##### Python Exceptions #####
# Python raises an Exception whenever it tries to execute invalid code
>>> 42/0
# Traceback (most recent call last):
#   File "<pyshell#1>", line 1, in <module>
#     42/0
# ZeroDivisionError: division by zero

##### Raising Your Own Exceptions #####
# Exceptions can be raised with a `raise` statement, where you can provide a custom message
# This will return an exception object

>>> raise Exception('This is the error message.')
# Traceback (most recent call last):
#   File "<pyshell#3>", line 1, in <module>
#     raise Exception('This is the error message.')
# Exception: This is the error message.

##### Exception Example: Box Function #########
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# ***************
# *             *
# *             *
# *             *
# ***************
# OOOOO
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# O   O
# OOOOO

# Unexpected output, want to raise an exception if double symbols are entered 
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# ******************************
# **             **
# **             **
# **             **
# ******************************
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# Traceback (most recent call last):
#   File "C:/Users/sstallworth/Desktop/BORING/example.py", line 23, in <module>
#     boxPrint('**',15, 5)
#   File "C:/Users/sstallworth/Desktop/BORING/example.py", line 12, in boxPrint
#     raise Exception('"symbol" needs to be a string of length 1')
# Exception: "symbol" needs to be a string of length 1

# Unexpected output, want to raise an exception if height and width are not >=2
# Raise an exception if only 1 symbol is entered 
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# *
# *

# This entire error message is called a "Traceback"
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# Traceback (most recent call last):
#   File "C:/Users/sstallworth/Desktop/BORING/example.py", line 27, in <module>
#     boxPrint('*',1, 1)
#   File "C:/Users/sstallworth/Desktop/BORING/example.py", line 15, in boxPrint
#     raise Exception('"width" and "height" must be greater or equal to 2.')
# Exception: "width" and "height" must be greater or equal to 2.

############### The traceback.format_exec() Function ###############
>>> import traceback
>>> try:
      raise Exception('This is the error message.')
    except:
      errorFile = open('error_log.txt','a')
      errorFile.write(traceback.format_exc())
      errorFile.close()
      print('The traceback info was written error_log.txt')

# 116
# The traceback info was written error_log.txt

# Gave the open() function a relative file path
# So will open and write to file in current working directory 
>>> import os
>>> os.getcwd()
# 'C:\\Users\\sstallworth\\Desktop\\BORING'

# Copy this, then press `Windows Key + R` and paste into Run dialog to jump to directory 
>>> print(os.getcwd())
# C:\Users\sstallworth\Desktop\BORING

# Can run agiain and it will append the new traceback to end of file
>>> try:
      raise Exception('This is the error message.')
    except:
      errorFile = open('error_log.txt','a')
      errorFile.write(traceback.format_exc())
      errorFile.close()
      print('The traceback info was written error_log.txt')
	
# 116
# The traceback info was written error_log.txt

##########################################################################
####################### ASSERTIONS #######################################
##########################################################################

########## Assertions and the assert statement ##############
# The assert statement is the assert keyword followed by a condition
# This will always evaluate to False because it is the False value itself
>>> assert False, 'This is the error message.' 
# Traceback (most recent call last):
#   File "<pyshell#33>", line 1, in <module>
#     assert False, 'This is the error message.'
# AssertionError: This is the error message.

########## Assertion Example: Traffic Lights ###############
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example2.py ==========
# {'ns': 'green', 'ew': 'red'}
# {'ns': 'yellow', 'ew': 'green'}
 
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example2.py ==========
# Traceback (most recent call last):
#   File "C:/Users/sstallworth/Desktop/BORING/example2.py", line 14, in <module>
#     switchLights(market_2nd)
#   File "C:/Users/sstallworth/Desktop/BORING/example2.py", line 11, in switchLights
#     assert 'red' in intersection.values(), 'Neither light is red!' + str(intersection) 
# AssertionError: Neither light is red!{'ns': 'yellow', 'ew': 'green'}