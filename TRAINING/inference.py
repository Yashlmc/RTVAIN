import cv2
import time
from ultralytics import YOLO

# Load your trained YOLOv11 model
model = YOLO("runs/detect/yolov11_T2-50/weights/best.pt")

# Open video file or webcam
video_path = "sample.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video.")
    exit()

# Get FPS of the source video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
frame_time = 1 / fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, conf=0.5, verbose=False)

    # Process detections
    for r in results:
        h, w, _ = frame.shape

        for box in r.boxes:
            # xywh normalized format (by default)
            x_tl, y_tl, bw, bh = box.xywhn[0].tolist()  # normalized values
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Convert normalized top-left to absolute pixel coordinates
            x1 = int(x_tl * w)
            y1 = int(y_tl * h)
            x2 = int((x_tl + bw) * w)
            y2 = int((y_tl + bh) * h)

            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLOv11 (Top-left Boxes)", frame)

    # Maintain normal playback speed
    elapsed = time.time() - start_time
    delay = max(int((frame_time - elapsed) * 1000), 1)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
from ultralytics import YOLO

# Load your trained YOLOv11 model
model = YOLO("runs/detect/yolov11_T2-50/weights/best.pt")

# Open video file or webcam
video_path = "sample.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video.")
    exit()

# Get FPS of the source video
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback
frame_time = 1 / fps

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model.predict(frame, conf=0.5, verbose=False)

    # Process detections
    for r in results:
        h, w, _ = frame.shape

        for box in r.boxes:
            # xywh normalized format (by default)
            x_tl, y_tl, bw, bh = box.xywhn[0].tolist()  # normalized values
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Convert normalized top-left to absolute pixel coordinates
            x1 = int(x_tl * w)
            y1 = int(y_tl * h)
            x2 = int((x_tl + bw) * w)
            y2 = int((y_tl + bh) * h)

            label = f"{model.names[cls]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Show frame
    cv2.imshow("YOLOv11 (Top-left Boxes)", frame)

    # Maintain normal playback speed
    elapsed = time.time() - start_time
    delay = max(int((frame_time - elapsed) * 1000), 1)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
