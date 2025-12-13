############### The logging.basicConfig() Function ###############
>>> import logging
>>> logging.basicConfig(level=logging.DEBUG, format = '%(asctime)s - %(levelname)s - %(message)s')

############### Example: Buggy Factorial Program ##############
# Factorial 4
>>> 1*2*3*4

# Factorial 7
>>> 1*2*3*4*5*6*7
# 5040

# See there is a bug in our code somewhere
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# 0

# Enable logging and call the debug logging function to identify where the issue is  
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# 2024-02-11 09:35:32,669 - DEBUG - Start of program
# 2024-02-11 09:35:32,685 - DEBUG - Start of factorial (5)
# 2024-02-11 09:35:32,687 - DEBUG - i is 0, total is 0
# 2024-02-11 09:35:32,688 - DEBUG - i is 1, total is 0
# 2024-02-11 09:35:32,689 - DEBUG - i is 2, total is 0
# 2024-02-11 09:35:32,690 - DEBUG - i is 3, total is 0
# 2024-02-11 09:35:32,691 - DEBUG - i is 4, total is 0
# 2024-02-11 09:35:32,693 - DEBUG - i is 5, total is 0
# 2024-02-11 09:35:32,694 - DEBUG - Return value is 0
# 0

# Now see that the error has been corrected
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# 2024-02-11 09:36:55,432 - DEBUG - Start of program
# 2024-02-11 09:36:55,461 - DEBUG - Start of factorial (5)
# 2024-02-11 09:36:55,463 - DEBUG - i is 1, total is 1
# 2024-02-11 09:36:55,464 - DEBUG - i is 2, total is 2
# 2024-02-11 09:36:55,465 - DEBUG - i is 3, total is 6
# 2024-02-11 09:36:55,466 - DEBUG - i is 4, total is 24
# 2024-02-11 09:36:55,467 - DEBUG - i is 5, total is 120
# 2024-02-11 09:36:55,468 - DEBUG - Return value is 120
# 120
# 2024-02-11 09:36:55,472 - DEBUG - End of program

# Now disable debugging logging messages 
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# 120

# Writing the logging messages to a file instead of to the screen 
# ========== RESTART: C:/Users/sstallworth/Desktop/BORING/example.py ==========
# 120