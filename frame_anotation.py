import cv2
import os

# Cesta k videu
video_path = "testvideos/main/ch01_20241022100000.mp4"
video_path2 = "testvideos/main/ch03_20241022105505.mp4"

output_folder = "video_frames"
os.makedirs(output_folder, exist_ok=True)

# Otvorenie videa
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Failed to open video")
    exit()

# Získanie FPS a celkového počtu snímkov
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS: ", fps, ", Total frames: ", total_frames)

# Výpočet intervalu pre extrahovanie 1 000 snímkov
skip_frames = total_frames // 1000  # Každý 17. snímok
print("Extracting every", skip_frames, "th frame")

frame_count = 0
saved_count = 0

# Slučka na extrahovanie snímkov
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Extrahujte každý skip_frames snímok
    if frame_count % skip_frames == 0:
        frame_filename = os.path.join(output_folder, f'frame_{saved_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        print("Frame num ", frame_count, "saved")
        saved_count += 1

        # Ukončenie, ak sme extrahovali 1 000 snímkov
        if saved_count >= 1000:
            break

    frame_count += 1

# Uvoľnenie zdrojov
cap.release()
print(f"Extracted {saved_count} frames to '{output_folder}'.")