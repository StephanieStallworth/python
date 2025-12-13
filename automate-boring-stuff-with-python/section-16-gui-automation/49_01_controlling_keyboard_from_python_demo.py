# Controlling the Keyboard							
import pyautogui  

# Click first where you want to focus, then type
>>> pyautogui.click(100,100);pyautogui.typewrite('Hello World!')

# Pass in the `interval` keyword with the number of seconds to pause between each character  
# To make the typing more human like
>>> pyautogui.click(100,100);pyautogui.typewrite('Hello World!', interval = 0.2)  

# pyautogui lets you send a list of the string to type out 
>>> pyautogui.click(100,100);pyautogui.typewrite(['a','b','left','left','X','Y'], interval = 1)						

# For hard-to-type keys, can use keyboard key strings 
# To get names of different keyboard keys you can use  
>>> pyautogui.KEYBOARD_KEYS

# Press a single key
# press() function and pass it one of the keyboard keys above
>>> pyautogui.press('f1') 

# Keyboard shortcuts  
# Call the hotkey() function and pass it a series of keys that it will press in combination
>>> pyautogui.hotkey('ctrl','o') 