import cv2
import numpy as np
import os

from cv2 import WINDOW_NORMAL

from getColorRange import *

# os.environ["QT_QPA_PLATFORM"] = "xcb"

#video_path = "testvideos/main/ch01_20241022100000.mp4"
video_path = "../testvideos/main/ch03_20241022100000.mp4"
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps    = cap.get(cv2.CAP_PROP_FPS)
print("Num of frames: ", length, "FPS: ", fps)

# Define color in BGR
# modra_z_tricka_z_videa = np.uint8([[[94, 28, 19]]])
modra_z_tricka_ine_svetelne_podmienky = np.uint8([[[11, 4, 1]]])

# lower_bound, upper_bound = getHSVcolorRangeFromBGR(modra_z_tricka_z_videa, 2)
lower_bound, upper_bound = getDarkHSVcolorRangeFromBGR(modra_z_tricka_ine_svetelne_podmienky)
print("Lower bound: ", lower_bound, "Upper bound: ", upper_bound)

# Create a window for displaying frames
cv2.namedWindow('frame', WINDOW_NORMAL)
cv2.namedWindow('mask', WINDOW_NORMAL)

#--------------SLIDER na nastavenie minimalnej pixelarei----------------
# Initial minimum area for detection
min_pixel_area = 500
# max pixel area
max_pixel_area = 1500
# Callback function for trackbar (does nothing but is required by OpenCV)
def on_trackbar(val):
    global min_pixel_area
    min_pixel_area = val  # Update the minimum pixel area with trackbar value

# Create a trackbar for setting min_pixel_area in the 'frame' window
cv2.createTrackbar('Min Pixel Area', 'frame', min_pixel_area, max_pixel_area, on_trackbar)
#---------------------------------------------------------------------

is_paused = False
bounding_boxes = []
while True:
    if not is_paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_num = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print("current frame: ", current_frame_num)
        # Convert the frame to the HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask to extract yellow regions
        mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

        # Find contours in the maskq
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all detected contours
        for contour in contours:
            # Filter contours by area (size of the object in pixels)
            area = cv2.contourArea(contour)
            if area >= min_pixel_area:  # Only consider contours above the minimum area
                # Calculate the bounding box for each contour
                x, y, w, h = cv2.boundingRect(contour)

                # Draw a rectangle around each detected yellow object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Cierna', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Show the frame with the drawn rectangles
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        # cv2.imshow('contours', contours)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Toggle pause/play on space bar press
    if cv2.waitKey(30) & 0xFF  == ord(' '):  # Space bar
        is_paused = not is_paused  # Toggle between paused and playing

cap.release()
cv2.destroyAllWindows()
