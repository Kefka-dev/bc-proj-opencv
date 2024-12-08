from math import trunc

import cv2
import numpy as np
from PIL import Image
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
# source https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html

cap = cv2.VideoCapture(0)

#yellow
#yellow color values https://stackoverflow.com/questions/9179189/detect-yellow-color-in-opencv
lowerYellow = np.array([20,100,100])
print(lowerYellow)
upperYellow = np.array([30,255,255])

#black color
lowerBlack = np.array([ 0,  0, 0])
upperBlack = np.array([10,100,10])

#green
lowerGreen = np.array([50, 100, 100])
upperGreen = np.array([70, 255,255])
kernel = np.ones((5,5),np.float32)/25
while True:

    ret, frame = cap.read()

    #converts the frame into HSV color scheme
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv_frame, lowerYellow, upperYellow)

    # mask_filtered = cv2.filter2D(mask, -1, kernel)

    # maskGaussianBlur = cv2.GaussianBlur(mask, (11,11), 0)

    res = cv2.bitwise_and(frame, frame, mask= mask)

    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

    print(bbox)


    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    # cv2.imshow('mask2', ma)
    cv2.imshow('res', res)

    # close the window on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()