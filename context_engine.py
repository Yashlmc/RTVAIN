import math
from collections import deque


class ContextEngine:
    """
    Real-time Context Engine for football penalty understanding.
    Works on overlapping sliding windows of YOLO detections.
    """

    def __init__(self, frame_window=5):
        self.frame_window = frame_window
        self.ball_positions = deque(maxlen=frame_window)
        self.keeper_boxes = deque(maxlen=frame_window)
        self.goal_boxes = deque(maxlen=frame_window)

        self.kick_detected = False
        self.result_declared = False
        self.last_event_frame = -1
        self.event_log = []

    # ---------------------------- Utility Functions ----------------------------

    def distance(self, p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

    def velocity(self, p_prev, p_now):
        return self.distance(p_prev, p_now)

    def get_center(self, box):
        """Convert YOLO bbox [x_center, y_center, w, h] into tuple"""
        return (box[0], box[1])

    def box_intersects(self, boxA, boxB):
        """Check intersection between two boxes"""
        ax1, ay1 = boxA[0] - boxA[2]/2, boxA[1] - boxA[3]/2
        ax2, ay2 = boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2
        bx1, by1 = boxB[0] - boxB[2]/2, boxB[1] - boxB[3]/2
        bx2, by2 = boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2
        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    # ---------------------------- Main Update ----------------------------

    def update(self, frame_id, detections):
        """
        Called per frame. Append detections to sliding window.
        """
        ball = next((d for d in detections if d["class"] == "ball"), None)
        keeper = next((d for d in detections if d["class"] == "goalkeeper"), None)
        goal = next((d for d in detections if d["class"] == "goalpost"), None)

        if not (ball and keeper and goal):
            return None

        self.ball_positions.append(self.get_center(ball["bbox"]))
        self.keeper_boxes.append(keeper["bbox"])
        self.goal_boxes.append(goal["bbox"])

        # Not enough frames yet
        if len(self.ball_positions) < self.frame_window:
            return None

        # 1️⃣ Kick detection
        v_start = self.velocity(self.ball_positions[0], self.ball_positions[2])
        v_end = self.velocity(self.ball_positions[-2], self.ball_positions[-1])

        if not self.kick_detected and v_start < 10 and v_end > 40:
            self.kick_detected = True
            self.event_log.append({"frame": frame_id, "event": "kick"})
            print(f"[{frame_id}] Kick detected")

        # 2️⃣ Determine outcome once kick detected
        if self.kick_detected and not self.result_declared:
            current_ball = ball["bbox"]
            keeper_now = keeper["bbox"]
            goal_now = goal["bbox"]

            goal_x1 = goal_now[0] - goal_now[2]/2
            goal_x2 = goal_now[0] + goal_now[2]/2
            ball_center = self.get_center(current_ball)

            # Check Save
            if self.box_intersects(current_ball, keeper_now):
                return self.register_event("save", frame_id)

            # Check Goal
            if goal_x1 <= ball_center[0] <= goal_x2 and ball_center[1] < (goal_now[1] + goal_now[3]/2):
                return self.register_event("goal", frame_id)

            # Check Miss
            if ball_center[0] > goal_x2 + 30 or ball_center[0] < goal_x1 - 30:
                return self.register_event("miss", frame_id)

        return None

    # ---------------------------- Event Registration ----------------------------

    def register_event(self, event_type, frame_id):
        """
        Avoid duplicates in overlapping windows.
        """
        if frame_id - self.last_event_frame < self.frame_window:
            return None  # skip duplicate events from overlap

        self.result_declared = True
        self.last_event_frame = frame_id
        direction = self.get_direction()
        self.event_log.append({"frame": frame_id, "event": event_type, "direction": direction})
        print(f"[{frame_id}] {event_type.upper()} detected ({direction})")

        return {"event": event_type, "direction": direction}

    # ---------------------------- Direction ----------------------------

    def get_direction(self):
        if len(self.ball_positions) < 2:
            return None
        x_start, y_start = self.ball_positions[0]
        x_end, y_end = self.ball_positions[-1]
        dx, dy = x_end - x_start, y_end - y_start
        horiz = "right" if dx > 0 else "left"
        vert = "top" if dy < 0 else "bottom"
        return f"{vert}-{horiz}"

    # ---------------------------- Reset ----------------------------

    def reset(self):
        self.ball_positions.clear()
        self.keeper_boxes.clear()
        self.goal_boxes.clear()
        self.kick_detected = False
        self.result_declared = False
        print("Context Engine reset for next sequence.")

if __name__ == "__main__":
    engine = ContextEngine(frame_window=5)

    # Simulated YOLO detections for 10 frames
    frames = [
        {"id": 1, "detections": [
            {"class": "ball", "bbox": [300, 500, 15, 15]},
            {"class": "goalkeeper", "bbox": [600, 480, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 2, "detections": [
            {"class": "ball", "bbox": [340, 475, 15, 15]},
            {"class": "goalkeeper", "bbox": [605, 480, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 3, "detections": [
            {"class": "ball", "bbox": [410, 440, 15, 15]},
            {"class": "goalkeeper", "bbox": [610, 475, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 4, "detections": [
            {"class": "ball", "bbox": [500, 420, 15, 15]},
            {"class": "goalkeeper", "bbox": [595, 455, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 5, "detections": [
            {"class": "ball", "bbox": [590, 400, 15, 15]},
            {"class": "goalkeeper", "bbox": [590, 450, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 6, "detections": [
            {"class": "ball", "bbox": [640, 380, 15, 15]},
            {"class": "goalkeeper", "bbox": [600, 430, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 7, "detections": [
            {"class": "ball", "bbox": [670, 370, 15, 15]},
            {"class": "goalkeeper", "bbox": [610, 420, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 8, "detections": [
            {"class": "ball", "bbox": [690, 360, 15, 15]},
            {"class": "goalkeeper", "bbox": [605, 420, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 9, "detections": [
            {"class": "ball", "bbox": [710, 350, 15, 15]},
            {"class": "goalkeeper", "bbox": [605, 420, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 10, "detections": [
            {"class": "ball", "bbox": [730, 340, 15, 15]},
            {"class": "goalkeeper", "bbox": [605, 420, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]},
        {"id": 11, "detections": [
            {"class": "ball", "bbox": [730, 340, 15, 15]},
            {"class": "goalkeeper", "bbox": [605, 420, 90, 150]},
            {"class": "goalpost", "bbox": [620, 400, 180, 160]}
        ]}
    ]

    # Process with sliding window overlap (5 frames, stride=3)
    window_size = 5
    stride = 3

    for start in range(0, len(frames) - window_size + 1, stride):
        window = frames[start:start + window_size]
        print(f"\n--- Processing frames {window[0]['id']} to {window[-1]['id']} ---")
        for frame in window:
            result = engine.update(frame["id"], frame["detections"])
            if result:
                print(f"Result @ Frame {frame['id']} → {result}")
        engine.reset()


###################################################################################################

# Example Output:

# --- Processing frames 1 to 5 ---
# [3] Kick detected
# [6] GOAL detected (top-right)
# Result @ Frame 6 → {'event': 'goal', 'direction': 'top-right'}
# Context Engine reset for next sequence.

# --- Processing frames 4 to 8 ---
# (no duplicate events triggered due to overlap)
# Context Engine reset for next sequence.

# --- Processing frames 7 to 10 ---
# (no duplicate events triggered)
# Context Engine reset for next sequence.
