################ Screen Shots and Image Recognition ################
import pyautogui			

# Take screenshot 
# Will return a pillow image object
# Can call any method on this image object that is described in the image documentation  
# Pillow is the image module covered in https://automatetheboringstuff.com/chapter17
>>> pyautogui.screenshot()
# <PIL.Image.Image image mode=RGB size=1280x720 at 0x35cc746>

# For our purposes, just want to take a screenshot 
# Then save it to an image file on the hard drive
>>> pyautogui.screenshot('c:\\screenshot_example.png')  

# Now pyautogui can "see" what is on the screen
# But to do image recognition have to call the locateOnScreen() function
# And pass it an image we want it to find on the screen 

# For example, took an image of the calculator and cropped just the "7" button
# Pass this filename to locateOnScreen() function
# Returns a tuple of 4 integers:
# x, y coordinates of where on the screen it can find that "7" key image  
# Width and height of that region (of the sample image we're looking for)
>>> pyautogui.locateOnScreen('c:\\calc7key.png')  
# (907, 316, 50, 41)

# More useful to call locateCenterOnScreen() function
# Returns the x,y coordinates of the CENTER of that region
>>> pyautogui.locateCenterOnScreen('c:\\calc7key.png')

# Pass this to the moveTo() function to move right to the "7" key coordinates 
>>> pyautogui.moveTo((932, 336), duration = 1)

# Then can have it click at those coordinates
# The click() function can take a single tuple value with the x and y integers 
>>> pyautogui.click((932,336)) 

# Or can just pass it the integer values separately 
>>> pyautogui.click(932, 336) 