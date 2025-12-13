#################### SCREEN COORDINATES ####################
import pyautogui

# Screen resolution 
>>> pyautogui.size()
# (1280, 720)

# Multiple assignment
>>> width, height = pyautogui.size()

>>> width 
# 1280

>>> height 
# 720

# Current coordindates of the mouse
>>> pyautogui.position()
# (324, 270)

# Move mouse to the top left corner
>>> pyautogui.position()
# (3,7)

# Bottom-most and right-most coordinates (bottom right corner)
# Will be 1 less than the full screen size (1280, 720)
# Because coordinates start at 0
>>> pyautogui.position()
# (1279, 719)

#################### MOVING THE MOUSE ####################
##### To Absolute position on screen #####
# Move mouse cursor instantly to upper-left corner
>>> pyautogui.moveTo(10,10)

# Move mouse cursor slowly over specified number of seconds 
>>> pyautogui.moveTo(10,10, duration = 1.5)	

##### Relative to where mouse already is #####
# Pass the `x` offset and `y` offset 
# Move cursor to the right by 20 pixels instantly 
>>> pyautogui.moveRel(20,0)

# Move cursor to the right by 200 pixels instantly
>>> pyautogui.moveRel(200,0)

# Move 200 pixels to the right over 2 seconds
>>> pyautogui.moveRel(20,0, duration = 2)	

# Move 100 pixels up the screen 
# Y-coodinates DECREASE going up so pass it `-100` for the `y` offset 

# Move instantly 
>>> pyautogui.moveRel(0, -100)

# Move up slowly 
>>> pyautogui.moveRel(0, -100, duration = 1)

################### CLICKING THE MOUSE ####################
# Click the "Help" button 							
# Move mouse cursor over "Help" button to identify position  
>>> pyautogui.gui.position()
# (339, 38)

# Click at those coordinates
>>> pyautogui.click(339,38)

# Could also do
# pyautogui.doubleClick(339,38)  
# pyautogui.rightClick(339,38)  
# pyautogui.middleClick(339,38)  

# No arguments to click wherever the mouse is currently 
>>> pyautogui.click() 	