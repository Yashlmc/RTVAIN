import numpy as np
from collections import deque

# --- Helper functions ---
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def velocity(curr, prev, fps=10):
    if prev is None:
        return 0.0
    return np.linalg.norm(np.array(curr) - np.array(prev)) * fps

def intersects(obj1, obj2):
    """Check if two bounding boxes intersect"""
    x1, y1, w1, h1 = obj1
    x2, y2, w2, h2 = obj2
    return not (x1+w1 < x2 or x2+w2 < x1 or y1+h1 < y2 or y2+h2 < y1)

# --- Sliding Window Action/Understanding Engine ---
class SlidingWindowPenaltyEngine:
    def __init__(self, fps=10, window_size=5, overlap=2):
        self.fps = fps
        self.window_size = window_size
        self.overlap = overlap
        self.frame_buffer = deque(maxlen=window_size)
        self.prev_positions = {}
        self.events_detected = []

    def update_frame(self, detections):
        """
        Update engine with a new frame of detections.
        Returns detected event if any.
        """
        # Map classes to positions and bboxes
        objs = {}
        for d in detections:
            cls = d["class"]
            objs[cls] = {
                "center": (d["x"], d["y"]),
                "bbox": (d["x"] - d["width"]/2, d["y"] - d["height"]/2, d["width"], d["height"])
            }

        # Compute velocities
        for cls in ["ball", "kicker", "keeper"]:
            curr = objs.get(cls, {}).get("center")
            prev = self.prev_positions.get(cls)
            objs[cls]["v"] = velocity(curr, prev, self.fps)
            self.prev_positions[cls] = curr

        self.frame_buffer.append(objs)

        # Process window if full
        if len(self.frame_buffer) == self.window_size:
            event = self._process_window()
            # Slide window keeping overlap
            for _ in range(self.window_size - self.overlap):
                self.frame_buffer.popleft()
            if event:
                self.events_detected.append(event)
                return event
        return None

    def _process_window(self):
        """
        Analyze current window to detect events:
        waiting, run-up, kick, ball_in_air, save, goal, miss
        """
        objs = self.frame_buffer[-1]

        # Check required objects
        if "ball" not in objs or "goalpost" not in objs:
            return {"event": "no_major_event"}

        ball = objs["ball"]
        kicker = objs.get("kicker")
        keeper = objs.get("keeper")
        goal = objs["goalpost"]

        # Ball and keeper directions
        if len(self.frame_buffer) >= 2:
            prev_ball = self.frame_buffer[-2]["ball"]["center"]
            ball_dir = np.array(ball["center"]) - np.array(prev_ball)
            if keeper:
                prev_keeper = self.frame_buffer[-2]["keeper"]["center"]
                keeper_dir = np.array(keeper["center"]) - np.array(prev_keeper)
            else:
                keeper_dir = np.array([0,0])
        else:
            ball_dir = np.array([0,0])
            keeper_dir = np.array([0,0])

        # Distances
        dist_kicker_ball = distance(kicker["center"], ball["center"]) if kicker else None
        dist_ball_goal = distance(ball["center"], goal["center"])
        dist_ball_keeper = distance(ball["center"], keeper["center"]) if keeper else None

        # Ball bbox for intersection
        ball_bbox = (ball["center"][0]-5, ball["center"][1]-5, 10, 10)
        goal_bbox = goal["bbox"]

        # --- Action logic ---
        event = "no_major_event"

        # 1. Waiting: kicker far, ball stationary
        if kicker and dist_kicker_ball > 100 and ball["v"] < 1:
            event = "waiting"

        # 2. Run-up: kicker approaching ball
        elif kicker and dist_kicker_ball <= 100 and dist_kicker_ball > 20 and kicker["v"] > 1:
            event = "run-up"

        # 3. Kick: ball speed spikes
        elif ball["v"] > 5 and dist_kicker_ball <= 20:
            event = "kick"

        # 4. Ball in flight toward goal
        elif ball["v"] > 3 and dist_ball_goal < 200:
            event = "ball_in_air"

        # 5. Save: keeper intersects ball
        elif keeper and intersects(ball_bbox, keeper["bbox"]):
            event = "save"

        # 6. Goal: ball inside goal without keeper
        elif goal_bbox[0] <= ball["center"][0] <= goal_bbox[0]+goal_bbox[2] and \
             goal_bbox[1] <= ball["center"][1] <= goal_bbox[1]+goal_bbox[3] and \
             (not keeper or not intersects(ball_bbox, keeper["bbox"])):
            event = "goal"

        # 7. Miss: ball outside goal vertical bounds
        elif ball["center"][1] < goal_bbox[1] or ball["center"][1] > goal_bbox[1]+goal_bbox[3]:
            event = "miss"

        return {
            "event": event,
            "ball_dir": ball_dir.tolist(),
            "keeper_dir": keeper_dir.tolist(),
            "dist_kicker_ball": round(dist_kicker_ball or 0, 2),
            "dist_ball_goal": round(dist_ball_goal, 2),
            "dist_ball_keeper": round(dist_ball_keeper or 0, 2)
        }


if __name__ == "__main__":
    # Initialize engine
    engine = SlidingWindowPenaltyEngine(fps=12, window_size=5, overlap=2)

    # Simulated video frames
    # Each frame: list of detections (class, x, y, width, height)
    video_detections = [
        # Frame 1 - waiting
        [{"class":"kicker","x":300,"y":430,"width":80,"height":160},
        {"class":"ball","x":340,"y":420,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 2 - waiting
        [{"class":"kicker","x":305,"y":430,"width":80,"height":160},
        {"class":"ball","x":340,"y":420,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 3 - run-up
        [{"class":"kicker","x":320,"y":425,"width":80,"height":160},
        {"class":"ball","x":340,"y":420,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 4 - run-up
        [{"class":"kicker","x":335,"y":420,"width":80,"height":160},
        {"class":"ball","x":340,"y":420,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 5 - kick
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":345,"y":415,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 6 - ball in air
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":370,"y":395,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 7 - ball in air
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":400,"y":370,"width":20,"height":20},
        {"class":"keeper","x":505,"y":375,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 8 - save (keeper intercepts)
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":510,"y":365,"width":20,"height":20},
        {"class":"keeper","x":505,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 9 - ball in air again (for goal sequence)
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":400,"y":370,"width":20,"height":20},
        {"class":"keeper","x":505,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 10 - goal (ball inside goal without keeper)
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":540,"y":320,"width":20,"height":20},
        {"class":"keeper","x":505,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 11 - miss (ball outside goal bounds)
        [{"class":"kicker","x":350,"y":415,"width":80,"height":160},
        {"class":"ball","x":600,"y":500,"width":20,"height":20},
        {"class":"keeper","x":505,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
        
        # Frame 12 - reset/waiting
        [{"class":"kicker","x":300,"y":430,"width":80,"height":160},
        {"class":"ball","x":340,"y":420,"width":20,"height":20},
        {"class":"keeper","x":500,"y":370,"width":90,"height":170},
        {"class":"goalpost","x":520,"y":300,"width":250,"height":200}],
    ]


    for i, frame in enumerate(video_detections):
        event = engine.update_frame(frame)
        if event:
            print(f"Frame {i+1}: Event Detected = {event}")


###################################################################################################
###################################################################################################

"""
# Example Output: 
Frame 5: Event Detected = {'event': 'kick', 'ball_dir': [5, -5], 'keeper_dir': [0, 0], 'dist_kicker_ball': 5.0, 'dist_ball_goal': 209.4, 'dist_ball_keeper': 161.4}
Frame 8: Event Detected = {'event': 'ball_in_air', 'ball_dir': [110, -5], 'keeper_dir': [0, -5], 'dist_kicker_ball': 167.63, 'dist_ball_goal': 65.76, 'dist_ball_keeper': 7.07}
Frame 11: Event Detected = {'event': 'miss', 'ball_dir': [60, 180], 'keeper_dir': [0, 0], 'dist_kicker_ball': 264.05, 'dist_ball_goal': 215.41, 'dist_ball_keeper': 161.01}
"""