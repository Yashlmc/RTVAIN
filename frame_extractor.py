import cv2
import os

# === Configuration ===
video_path = 'video_dataset/penalty-dataset.mp4'  # Replace with your video path
output_folder = 'frames2'     # Folder to save frames
frame_rate = 5               # Frames per second to save (adjust as needed)

# === Create output folder if it doesn't exist ===
os.makedirs(output_folder, exist_ok=True)

# === Load the video ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Video FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")

frame_interval = int(fps / frame_rate) if frame_rate else 1
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        frame_name = f"frame_{saved_count:06d}.jpg"
        cv2.imwrite(os.path.join(output_folder, frame_name), frame)
        saved_count += 1

    frame_count += 1

cap.release()
print(f"Done! {saved_count} frames saved in '{output_folder}'")
