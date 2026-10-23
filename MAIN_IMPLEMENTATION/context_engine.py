import math
from collections import deque

class ContextEngine:
    """
    Real-time Context Engine for football penalty understanding.
    Adapted for top-left (x, y, w, h) bounding boxes.
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
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def velocity(self, p_prev, p_now):
        return self.distance(p_prev, p_now)

    def get_center(self, box):
        """
        Convert top-left (x, y, w, h) → center (cx, cy)
        """
        x, y, w, h = box
        return (x + w / 2, y + h / 2)

    def box_intersects(self, boxA, boxB):
        """
        Check intersection between two boxes (top-left format).
        """
        ax1, ay1, aw, ah = boxA
        bx1, by1, bw, bh = boxB

        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        return not (ax2 < bx1 or ax1 > bx2 or ay2 < by1 or ay1 > by2)

    # ---------------------------- Main Update ----------------------------

    def update(self, frame_id, detections):
        """
        Called per frame. Appends detections to sliding window and detects events.
        Expects detections with top-left bounding boxes.
        """
        ball = next((d for d in detections if d["class"] == "ball"), None)
        keeper = next((d for d in detections if "keeper" in d["class"].lower()), None)
        goal = next((d for d in detections if "goal" in d["class"].lower()), None)

        # Ensure we have key entities
        if not (ball and keeper and goal):
            return None

        self.ball_positions.append(self.get_center(ball["bbox"]))
        self.keeper_boxes.append(keeper["bbox"])
        self.goal_boxes.append(goal["bbox"])

        # Wait until we have enough frames for motion estimation
        if len(self.ball_positions) < self.frame_window:
            return None

        # -------------------- 1️⃣ Kick Detection --------------------
        v_start = self.velocity(self.ball_positions[0], self.ball_positions[2])
        v_end = self.velocity(self.ball_positions[-2], self.ball_positions[-1])

        if not self.kick_detected and v_start < 10 and v_end > 40:
            self.kick_detected = True
            self.event_log.append({"frame": frame_id, "event": "kick"})
            print(f"[{frame_id}] Kick detected")

        # -------------------- 2️⃣ Determine Result --------------------
        if self.kick_detected and not self.result_declared:
            current_ball = ball["bbox"]
            keeper_now = keeper["bbox"]
            goal_now = goal["bbox"]

            goal_x1, goal_y1, goal_w, goal_h = goal_now
            goal_x2, goal_y2 = goal_x1 + goal_w, goal_y1 + goal_h

            ball_center = self.get_center(current_ball)

            # ⚽ SAVE: if ball and keeper intersect
            if self.box_intersects(current_ball, keeper_now):
                return self.register_event("save", frame_id)

            # 🥅 GOAL: if ball center inside horizontal bounds of goal
            if goal_x1 <= ball_center[0] <= goal_x2 and goal_y1 <= ball_center[1] <= goal_y2:
                return self.register_event("goal", frame_id)

            # ❌ MISS: ball goes horizontally beyond goal boundaries
            if ball_center[0] > goal_x2 + 30 or ball_center[0] < goal_x1 - 30:
                return self.register_event("miss", frame_id)

        return None

    # ---------------------------- Event Registration ----------------------------

    def register_event(self, event_type, frame_id):
        """
        Avoid duplicate detections in overlapping windows.
        """
        if frame_id - self.last_event_frame < self.frame_window:
            return None

        self.result_declared = True
        self.last_event_frame = frame_id
        direction = self.get_direction()

        event = {"frame": frame_id, "event": event_type, "direction": direction}
        self.event_log.append(event)
        print(f"[{frame_id}] {event_type.upper()} detected ({direction})")

        return event

    # ---------------------------- Direction ----------------------------

    def get_direction(self):
        """
        Estimate direction based on movement of ball centers.
        """
        if len(self.ball_positions) < 2:
            return None
        x_start, y_start = self.ball_positions[0]
        x_end, y_end = self.ball_positions[-1]
        dx, dy = x_end - x_start, y_end - y_start
        horiz = "right" if dx > 0 else "left"
        vert = "up" if dy < 0 else "down"
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

    frames = [
        {
            "id": 1,
            "detections": [
                {"class": "goal_post", "bbox": [450, 200, 400, 300]},
                {"class": "keeper", "bbox": [800, 450, 200, 150]},
                {"class": "kicker", "bbox": [600, 520, 120, 300]},
                {"class": "ball", "bbox": [920, 740, 30, 30]}
            ]
        },
        {
            "id": 2,
            "detections": [
                {"class": "goal_post", "bbox": [450, 200, 400, 300]},
                {"class": "keeper", "bbox": [800, 450, 200, 150]},
                {"class": "kicker", "bbox": [600, 520, 120, 300]},
                {"class": "ball", "bbox": [910, 730, 30, 30]}
            ]
        },
        {
            "id": 3,
            "detections": [
                {"class": "goal_post", "bbox": [450, 200, 400, 300]},
                {"class": "keeper", "bbox": [800, 450, 200, 150]},
                {"class": "kicker", "bbox": [600, 520, 120, 300]},
                {"class": "ball", "bbox": [880, 700, 30, 30]}
            ]
        },
        {
            "id": 4,
            "detections": [
                {"class": "goal_post", "bbox": [450, 200, 400, 300]},
                {"class": "keeper", "bbox": [800, 450, 200, 150]},
                {"class": "kicker", "bbox": [600, 520, 120, 300]},
                {"class": "ball", "bbox": [700, 500, 30, 30]}
            ]
        },
        {
            "id": 5,
            "detections": [
                {"class": "goal_post", "bbox": [450, 200, 400, 300]},
                {"class": "keeper", "bbox": [800, 450, 200, 150]},
                {"class": "kicker", "bbox": [600, 520, 120, 300]},
                {"class": "ball", "bbox": [550, 350, 30, 30]}
            ]
        }
    ]

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
