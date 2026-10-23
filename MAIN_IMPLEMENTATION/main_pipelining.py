"""
RTVAIN real-time orchestrator:
YOLO → ContextEngine (sliding window) → Commentary → TTS (threaded)
"""

from yolo_video_player import YOLOVideoPlayer
from context_engine import ContextEngine
from narration_generator import CommentaryGenerator
from tts_engine import TextToSpeech
import cv2
import time
from collections import deque
import threading


class RTVAIN:
    def __init__(self,
                 model_path: str,
                 video_source: str,
                 conf_threshold: float = 0.3,
                 window_size: int = 8,
                 stride: int = 3):
        """
        Initializes the RTVAIN pipeline for real-time operation.
        """
        # 1️⃣ YOLO Detector
        self.model = YOLOVideoPlayer(model_path=model_path,
                                     video_source=video_source,
                                     conf_threshold=conf_threshold)

        # 2️⃣ Context Engine
        self.context_engine = ContextEngine(frame_window=window_size)

        # 3️⃣ Commentary + Speech
        self.commentary_gen = CommentaryGenerator()
        self.tts = TextToSpeech(voice="en-GB-RyanNeural", rate="+15%")
        self.tts_threads = []

        # Window logic
        self.window_size = window_size
        self.stride = stride
        self.buffer = deque(maxlen=window_size)
        self.frame_counter = 0

    # ----------------------------------------------------------------------

    def run(self):
        """
        Runs YOLO detection and context processing in parallel style.
        """
        print("\n🚀 Starting Real-time RTVAIN Pipeline...\n")
        cap = cv2.VideoCapture(self.model.video_source)
        # cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            raise Exception("❌ Error: Cannot open video source.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_time = 1 / fps

        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    break

                self.frame_counter += 1

                # Run YOLO inference for this frame
                results = self.model.model.predict(frame,
                                                   conf=self.model.conf_threshold,
                                                   verbose=False)

                detections = []
                for r in results:
                    h, w, _ = frame.shape
                    for box in r.boxes:
                        x_tl, y_tl, bw, bh = box.xywhn[0].tolist()
                        cls = int(box.cls[0])
                        x1 = int(x_tl * w)
                        y1 = int(y_tl * h)
                        x2 = int((x_tl + bw) * w)
                        y2 = int((y_tl + bh) * h)
                        detections.append({
                            "class": self.model.model.names[cls],
                            "bbox": [x1, y1, x2 - x1, y2 - y1]
                        })

                # Append current frame detections to buffer
                self.buffer.append({
                    "frame": self.frame_counter,
                    "detections": detections
                })

                # Process window only when ready
                if len(self.buffer) == self.window_size and self.frame_counter % self.stride == 0:
                    # ✅ Run context analysis + TTS in a separate thread
                    threading.Thread(
                        target=self._process_window,
                        args=(list(self.buffer),),
                        daemon=True
                    ).start()

                # Display video
                for det in detections:
                    x, y, w, h = det["bbox"]
                    label = det["class"]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, max(25, y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                cv2.namedWindow("RTVAIN - Real-time", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("RTVAIN - Real-time", 1344, 756)
                cv2.imshow("RTVAIN - Real-time", frame)

                # Maintain real-time FPS
                elapsed = time.time() - start_time
                delay = max(int((frame_time - elapsed) * 1000), 1)
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n🛑 Video stream ended. RTVAIN stopped.\n")
            for t in self.tts_threads:
                t.join()
            print("✅ All TTS threads finished. Program exiting.")

    # ----------------------------------------------------------------------

    def _process_window(self, window_frames):
        """
        Called automatically once buffer fills. Runs context analysis + commentary.
        Executes in its own thread to avoid blocking video playback.
        """
        print(f"\n--- Processing frames {window_frames[0]['frame']} "
              f"to {window_frames[-1]['frame']} ---")

        sequence_events = []
        for frame_data in window_frames:
            frame_id = frame_data["frame"]
            result = self.context_engine.update(frame_id, frame_data["detections"])
            if result:
                sequence_events.append(result)
                print(f"Result @ Frame {frame_id} → {result}")

        if sequence_events:
            match_context = {
                "player_kicker": "Shakes",
                "goalkeeper": "Valdes",
                "score": "2 - 2"
            }

            commentary = self.commentary_gen.generate_sequence_commentary(sequence_events, match_context)
            if commentary:
                print(f"\n🎙️ Commentary: {commentary}")

                # 🎤 Speak asynchronously so video loop isn't blocked
                tts_thread = threading.Thread(
                    target=self.tts.speak,
                    args=(commentary,),
                    daemon=False
                )
                tts_thread.start()
                self.tts_threads.append(tts_thread)

        # Reset context for next window
        self.context_engine.reset()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    app = RTVAIN(
        model_path="models/yolov11_T1-n150/weights/best.pt",
        video_source="penalties/pen2.mp4",
        conf_threshold=0.2,
        window_size=8,
        stride=3
    )
    app.run()
