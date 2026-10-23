import cv2
import os
import numpy as np

video_path = "video_dataset/sample.mp4"
output_dir = "frames_output"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(round(fps / 12))  # keep high fps ~12

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# ROI crop (penalty area)
prev_gray = prev_gray[250:900, 300:1600]

frame_count, saved_count = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = gray[250:900, 300:1600]

    # Optical Flow (detect local motion)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, roi, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    motion_score = np.mean(mag)

    # Threshold tuned for ball motion vs camera pan
    if motion_score > 2.5:   # try between 2.0–3.5
        cv2.imwrite(f"{output_dir}/frame_{frame_count}.jpg", frame)
        saved_count += 1

    prev_gray = roi

cap.release()
print(f"Saved {saved_count} motion frames from {frame_count} checked")
