import cv2
import numpy as np
import random
from trackers.byte_tracker import BYTETracker

def xywh_to_xyxy(x, y, w, h):
    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)
    return x1, y1, x2, y2

cfg = { 
    "track_high_thresh":0.5,
    "track_low_thresh":0.1,
    "new_track_thresh":0.6,
    "track_buffer":30,
    "match_thresh":0.8,
}

tracker = BYTETracker(args=cfg) 

width, height = 640, 480
fps = 60
duration = 60 
num_frames = fps * duration
frames_per_new_circle = 120  

# Circle parameters
radius = 30
w, h = radius*3, radius*3 # for bbox
circle_thickness = -1  
max_step = 5
direction_change_probability = 0.05

# Initial list of circles (each circle has x, y, dx, dy, color)
circles = []

for frame_idx in range(num_frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Add a new circle every `frames_per_new_circle` frames
    if frame_idx % frames_per_new_circle == 0:
        x, y = random.randint(radius, width - radius), random.randint(radius, height - radius)
        dx, dy = random.choice([-1, 1]), random.choice([-1, 1])
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        circles.append([x, y, dx, dy, color])

    # Update and draw each circle
    boxes_xywh = []
    scores = []
    classes = []
    for circle in circles:
        x, y, dx, dy, color = circle
        
        boxes_xywh.append([x,y,w,h])
        scores.append(0.9) 
        classes.append(0.)
        
        # Update circle position
        if random.random() < direction_change_probability:
            dx, dy = random.choice([-1, 1]), random.choice([-1, 1])

        x += dx * random.randint(1, max_step)
        y += dy * random.randint(1, max_step)

        # Ensure the circle stays within the frame boundaries
        x = max(radius, min(width - radius, x))
        y = max(radius, min(height - radius, y))

        # Update the circle's position in the list
        circle[0], circle[1], circle[2], circle[3] = x, y, dx, dy
    
    boxes_xywh = np.array(boxes_xywh, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes)

    tracks = tracker.update(
        bboxes=boxes_xywh, 
        scores=scores, 
        cls=classes, 
        img=frame
    )
    idx = tracks[:, -1].astype(int) 
    
    for ids, circle in zip(idx, circles):
        x, y, dx, dy, color = circle
        x1, y1, x2, y2 = xywh_to_xyxy(x,y,w,h)
        cv2.circle(frame, (x, y), radius, color, circle_thickness)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, f"id:{ids}", (x1 , y1-5), 1, 1.5, (255,0,0), 2)

 

    cv2.imshow("", frame) 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break

cv2.destroyAllWindows()
