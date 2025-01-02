import cv2
from ultralytics import YOLO


MODEL = "yolov8x.pt"
model = YOLO(MODEL)

# results = model("/home/patrik/Pictures/Screenshots/Screenshot from 2024-12-09 14-25-25.png",show=True)
results = model("/home/patrik/Pictures/Screenshots/Screenshot from 2024-12-09 14-43-07.png",show=True)
# "0" will display the window infinitely until any keypress (in case of videos)
# waitKey(1) will display a frame for 1 ms
cv2.waitKey(0)