mport cv2
import time
import glob
import os
from emailing import send_email
from threading import Thread

# Create instance of VideoCapture class
# 0 = Main camera, 1 = Secondary USB camera
video = cv2.VideoCapture(0)

# Smooth out video
time.sleep(1)

# Video is just a series of images
# Move this into a while loop, don't need to repeat these lines
# check1, frame1 = video.read()
# time.sleep(1)
# check2, frame2 = video.read()
# time.sleep(1)
# check3, frame3 = video.read()
# print(frame3)

# Create while loop instead
first_frame = None

status_list = []

count = 1

def clean_folder():
    # Add print statement to better understand what thread is doing when it calls the function
    print("clean_folder function started")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean_folder function ended")

while True:
    # Always reset status value when loop starts
    status = 0
    # time.sleep(1) # gives camera time to load, this line moved to the top
    check, frame = video.read()

    # Convert all the frames to grayscale pixels to reduce the data in the matrices
    # Color is not necessary to compare differences in the frames
    # Alogrithm to be applied in all capital letters
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Don't need that much precision, amount of blur you want to apply
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21,21), 0)

    # First frame saved in one variable and current frame in another variable
    if first_frame is None:
        first_frame = gray_frame_gau

    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # Goal is to have only the object in white and the environment less white
    # So classify white pixels with a value
    # If pixel has a value of N or more, reassign value of 255 to that pixel
    # Move in and out of camera to test for the correct value of N
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Show frame
    # cv2.imshow("My video", gray_frame_gau)
    # cv2.imshow("My video", delta_frame_gau)
    # cv2.imshow("My video", delta_frame)
    cv2.imshow("My video", dil_frame)

    # Detect the contours around the white areas
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If its a small or "fake" object then continue, run the loop again
    # Otherwise move to the next line (Pro Tip: Don't always need 'else' keyword when using 'if')
    for contour in contours:
        if cv2.countourArea(contour) < 10000:  # pixels
            continue
        x, y, w, h = cv2.boundingRect(contour)

        # Trigger action when the webcam detects an object
        rectangle = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0)) # draw rectangle around frame
        if rectangle.any():
            status = 1
            # send_email() # moved below when object exits frame

            # Produce image onlyif an object in the frame
            # Videos are made frames (usually 30 frames per second)
            cv2.iamwrite(f"images/{count}.png", frame)
            count = count + 1
            # Get middle image
            all_images = glob.glob("images/*.png")
            index = int(len(all_images) / 2)
            image_with_object = all_images[index]

    status_list.append(status)
    status_list = status_list[-2:]

    # Object just exited frame, send email
    if status_list[0] == 1 and status_list[1] == 0:

        ########## Prepare thread instances (not executing them yet) ##########
        # Instead of calling the function directly
        # send_email(image_with_object)

        # Create a thread instance that will call the function
        email_thread = Thread(target=send_email, # function to be called by the thread
                              args=(image_with_object, ) # arguments you want to pass to the function (add comma to indicated this is a tuple not a single object)
                              )

        # Then point variable containing thread instance to daemon property
        # This line allows sent_email() function to be executed in the background
        email_thread.daemon = True

        # Create second thread instance for the clean_folder() function
        # clean_folder()
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        ########## Start threads ##########
        email_thread.start()

        # Moved to where the while loop ends
        # clean_thread.start()

    print(status_list)

    cv2.imshow("Video",frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
clean_thread.start()




