import cv2
import os

video_path = "video_dataset/penalty-dataset.mp4"
output_dir = "frames_output"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video FPS: {fps}")

target_fps = 12   # keep high so ball motion isn’t skipped
frame_interval = int(round(fps / target_fps))

# Higher threshold since ROI is smaller → tune between 20k–60k
motion_threshold = 40_000  

prev_roi = None
frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---- ROI Crop (Penalty Area) ----
    roi = gray[250:900, 300:1600]   # y1:y2, x1:x2

    if prev_roi is None:
        prev_roi = roi
        continue

    diff = cv2.absdiff(prev_roi, roi)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)

    if motion_score > motion_threshold:
        save_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(save_path, frame)
        saved_count += 1

    prev_roi = roi

cap.release()
print(f"Done! Saved {saved_count} frames from {frame_count} checked.")
