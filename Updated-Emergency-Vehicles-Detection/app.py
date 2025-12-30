from ultralytics import YOLO
import cv2
import os
from datetime import datetime
import numpy as np

model = YOLO("yolov8n.pt")

video_paths = [
    "nice - traffic signal.mp4",
    "partially ok.mp4",
    "nice - traffic signal.mp4",
    "partially ok.mp4"
]

caps = [cv2.VideoCapture(v) for v in video_paths]

for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Could not open video: {video_paths[i]}")
    else:
        print(f"Opened video: {video_paths[i]}")

output_dirs = ["output/cam1", "output/cam2", "output/cam3", "output/cam4"]
for d in output_dirs:
    os.makedirs(d, exist_ok=True)

next_cam_index = 0

def take_snapshot_sequential():
    """Capture a snapshot from the next camera in sequence."""
    global next_cam_index
    cap = caps[next_cam_index]

    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    if not ret:
        print(f"Camera {next_cam_index + 1}: Cannot read frame.")
        next_cam_index = (next_cam_index + 1) % 4
        return

    results = model(frame)
    annotated = results[0].plot()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dirs[next_cam_index]}/{timestamp}.jpg"
    cv2.imwrite(filename, annotated)
    print(f"Saved snapshot for Camera {next_cam_index + 1} → {filename}")

    next_cam_index = (next_cam_index + 1) % 4

def show_collage():
    """Display 4 video feeds in a 2x2 collage."""
    while True:
        frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            if not ret:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Feed {i+1} not available",
                            (80, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            frames.append(frame)

        frames = [cv2.resize(f, (640, 360)) for f in frames]
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        collage = np.vstack((top_row, bottom_row))

        cv2.imshow("Traffic Footage Collage", collage)

        key = cv2.waitKey(1)
        if key == ord('s'):
            take_snapshot_sequential()
        elif key == 27:  # ESC
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

show_collage()   #call the show_collage function to run the file