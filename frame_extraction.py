import cv2
import os
from ultralytics import YOLO

# Path to video
video_path = "testvideos/main/ch01_20241022100000.mp4"
video_path2 = "testvideos/main/ch03_20241022105505.mp4"
# Output folder for frames and annotations
output_folder = "video_frames"
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture(video_path2)
if not cap.isOpened():
    print("Failed to open video")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS: ", fps, ", Total frames: ", total_frames)

# Calculate interval to extract 1,000 frames
skip_frames = total_frames // 1000  # Extract every Nth frame
print("Extracting every", skip_frames, "th frame")

frame_count = 0
saved_count = 0

# Loop to extract frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extract every skip_frames frame
    if frame_count % skip_frames == 0:
        # Detect objects using YOLO
        results = model(frame)

        # Filter results to only include people (class 0 in COCO dataset)
        people_detections = []
        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # Class 0 is 'person' in COCO dataset
                    # Convert bounding box to YOLO format (normalized x_center, y_center, width, height)
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]

                    x_center = (xmin + xmax) / 2 / frame_width
                    y_center = (ymin + ymax) / 2 / frame_height
                    width = (xmax - xmin) / frame_width
                    height = (ymax - ymin) / frame_height

                    people_detections.append((0, x_center, y_center, width, height))  # Class 0 is 'person'

        # If people are detected, save the frame and create annotation file
        if people_detections:
            # Save frame as image
            frame_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            print("Frame num ", frame_count, "saved with people detected")

            # Create annotation file in YOLO format
            annotation_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.txt')
            with open(annotation_filename, 'w') as f:
                for detection in people_detections:
                    class_id, x_center, y_center, width, height = detection
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

            saved_count += 1

        # Stop if we've extracted 1,000 frames
        if saved_count >= 1000:
            break

    frame_count += 1

# Release resources
cap.release()
print(f"Extracted {saved_count} frames to '{output_folder}'.")