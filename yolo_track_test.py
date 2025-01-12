import cv2

from ultralytics import YOLO
import torch


if torch.cuda.is_available():
    device = 'cuda'  # Use GPU if available
else:
    device = 'cpu'  # Fallback to CPU

print(f"Using device: {device}")

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Open the video file
video_path = "testvideos/main/ch01_20241022100000.mp4"
video_path2 = "testvideos/main/ch03_20241022105505.mp4"
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()