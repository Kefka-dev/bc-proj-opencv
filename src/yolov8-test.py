import cv2
from cv2 import WINDOW_NORMAL
from ultralytics import YOLO
import math
import torch

video_path = "../testvideos/ch01_20241022100000.mp4"
video_path2 = "../testvideos/ch03_20241022105505.mp4"
video_path3 = "../testvideos/ch02_20241022130405.mp4"
video_path4 = "../testvideos/ch15_20241022105615.mp4"
video_path5 = "../testvideos/ch14_20241022100000.mp4"
cap = cv2.VideoCapture(video_path5)

# MODEL = "yolov8x.pt"
#model trained in 10 epochs
MODEL = "E:\\Skola\\FEI\\bakalarka\\bc-proj-opencv\\src\\finetunning\\runs\\detect\\train3\\weights\\best.pt"
#model trained in 100 epochs
MODEL2= "E:\\Skola\\FEI\\bakalarka\\bc-proj-opencv\\src\\finetunning\\runs\\detect\\train\\weights\\best.pt"
model = YOLO(MODEL)  # Load the model normally (WITHOUT classes=0 here)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.9
font_thickness = 2
text_color = (0, 255, 0)

if torch.cuda.is_available():
    device = 'cuda'  # Use GPU if available
else:
    device = 'cpu'  # Fallback to CPU

print(f"Using device: {device}")
cv2.namedWindow('frame', WINDOW_NORMAL)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, classes=0)  # Predict without class filtering
    # annotated_frame = results[0].plot()
    for r in results:
        boxes = r.boxes
        names = r.names # Get class names

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls = int(box.cls[0]) # Get class index
            class_name = names[cls] # Get class name

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            confidence = math.ceil((box.conf[0] * 100)) / 100
            confidence_text = f"{class_name} {confidence:.1f}"  # Include class name

            text_offset_y = -5

            cv2.putText(frame, confidence_text, (x1, y1 + text_offset_y), font, font_scale, text_color, font_thickness) # Display class name and confidence

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()