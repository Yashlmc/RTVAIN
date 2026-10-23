# inference

import cv2
import time
import threading
import json
from ultralytics import YOLO


class YOLOVideoPlayer:
    def __init__(self, model_path: str, video_source=0, conf_threshold: float = 0.5, export_path: str = "detections.json"):
        """
        Initialize the YOLO video player.
        :param model_path: Path to YOLO model (.pt file)
        :param video_source: Video path or 0 for webcam
        :param conf_threshold: Confidence threshold for detections
        :param export_path: Path to save the final detection results (JSON or TXT)
        """
        self.model = YOLO(model_path)
        self.video_source = video_source
        self.conf_threshold = conf_threshold
        self.export_path = export_path

        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise Exception("❌ Error: Cannot open video source.")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30
        self.frame_time = 1 / self.fps

        self.latest_frame = None
        self.latest_detections = []
        self.all_detections = []  # ✅ store framewise detections
        self.frame_count = 0
        self.running = False
        self.lock = threading.Lock()

        self._thread = threading.Thread(target=self._detection_thread, daemon=True)

    def _detection_thread(self):
        """Background thread for running YOLO detection asynchronously."""
        while self.running:
            if self.latest_frame is not None:
                with self.lock:
                    frame_copy = self.latest_frame.copy()
                    frame_id = self.frame_count

                results = self.model.predict(frame_copy, conf=self.conf_threshold, verbose=False)
                detections = []

                for r in results:
                    h, w, _ = frame_copy.shape
                    for box in r.boxes:
                        x_tl, y_tl, bw, bh = box.xywhn[0].tolist()
                        cls = int(box.cls[0])

                        x1 = int(x_tl * w)
                        y1 = int(y_tl * h)
                        x2 = int((x_tl + bw) * w)
                        y2 = int((y_tl + bh) * h)

                        det = {
                            "class": self.model.names[cls],
                            "bbox": [x1, y1, x2 - x1, y2 - y1]
                        }
                        detections.append(det)

                # ✅ Save detections for this frame
                frame_data = {"frame": frame_id, "detections": detections}

                with self.lock:
                    self.latest_detections = detections
                    self.all_detections.append(frame_data)

                # ✅ Print detections live
                print(json.dumps(frame_data, indent=2))

    def start(self, window_name="YOLOv11 Real-time (Async Detection)"):
        """Start video playback and async detection."""
        self.running = True
        self._thread.start()

        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            with self.lock:
                self.latest_frame = frame.copy()
                detections_copy = self.latest_detections.copy()

            for det in detections_copy:
                x, y, w, h = det["bbox"]
                label = det["class"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, max(25, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            cv2.imshow(window_name, frame)

            elapsed = time.time() - start_time
            delay = max(int((self.frame_time - elapsed) * 1000), 1)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break

        self.stop()

    def stop(self):
        """Stop video playback, save detections, and release resources."""
        self.running = False
        if self._thread.is_alive():
            self._thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

        # ✅ Export detections to file
        self._export_detections()

    def _export_detections(self):
        """Export all detections to JSON or TXT file."""
        if self.export_path.lower().endswith(".json"):
            with open(self.export_path, "w") as f:
                json.dump(self.all_detections, f, indent=2)
            print(f"\n✅ Detections saved to {self.export_path}")
        else:
            with open(self.export_path, "w") as f:
                for frame_data in self.all_detections:
                    f.write(json.dumps(frame_data) + "\n")
            print(f"\n✅ Detections saved to {self.export_path}")



if __name__ == "__main__":
    player = YOLOVideoPlayer(
        model_path="models/yolov11_T1-n150/weights/best.pt",
        video_source="penalties/pen1.mp4",
        conf_threshold=0.5,
        export_path="detections.json"
    )
    player.start()
