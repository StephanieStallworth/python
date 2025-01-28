# pip install opencv-python
import cv2

array = cv2.imread("image.png")
print(array.shape)

# NumPy array (NumPy is installed by opencv)
# List of list of lists
# Used to store large amount of numbers
print(type(array))
