# Još malo refine-at promptove...
# Install stuff
!pip install -q pytube decord
!pip install -q torch torchvision
!pip install -q mediapipe

# Download video from youtube
from pytube import YouTube
import time

youtube_url = 'https://www.youtube.com/watch?v=vqqt5p0q-eU'  # tu stavin bilo koji link
yt = YouTube(youtube_url)
streams = yt.streams.filter(file_extension='mp4')
file_path = streams[0].download()

# pregled videa
from decord import VideoReader, cpu
import numpy as np

videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
video_fps = videoreader.get_avg_fps()

# Loadamo model
import torch
import cv2

start_time = time.time()
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# Inicijaliziramo model
import mediapipe as mp
from collections import deque, Counter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Definiranje parametara (ovo bi se još dalo malo popeglat)
motion_status_window = deque(maxlen=5)
MOTION_THRESHOLD = 500
SIGNIFICANT_VERTICAL_MOVEMENT = 8 # bilo je 10 i prepoznaje dosta slabo
MODERATE_VERTICAL_MOVEMENT = 4
LOW_VERTICAL_MOVEMENT = 1
SIGNIFICANT_HORIZONTAL_MOVEMENT = 11
MODERATE_HORIZONTAL_MOVEMENT = 4
LOW_HORIZONTAL_MOVEMENT = 1

prev_positions = []

def classify_movement(roi, x, y, prev_x, prev_y):
    global prev_positions, motion_status_window

    motion_status = "TBD"  # Default status (ako ne prepozna current kretanje kao neko od definiranih)

    if roi.size != 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_y, gray_roi)
        _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(frame_diff)

        if motion_pixels > MOTION_THRESHOLD:
            y_movement = abs(y - prev_y)
            x_movement = abs(x - prev_x)
            if y_movement > SIGNIFICANT_VERTICAL_MOVEMENT and x_movement > SIGNIFICANT_HORIZONTAL_MOVEMENT:
                motion_status = "Running"
            elif y_movement > SIGNIFICANT_VERTICAL_MOVEMENT:
                motion_status = "Jumping"
            elif x_movement > SIGNIFICANT_HORIZONTAL_MOVEMENT:
                motion_status = "Fast Walking"
            elif y_movement > MODERATE_VERTICAL_MOVEMENT or x_movement > MODERATE_HORIZONTAL_MOVEMENT:
                motion_status = "Walking"
            elif y_movement > LOW_VERTICAL_MOVEMENT and y_movement < MODERATE_VERTICAL_MOVEMENT and x_movement > LOW_HORIZONTAL_MOVEMENT and x_movement < MODERATE_HORIZONTAL_MOVEMENT:
                motion_status = "Slow Walking"
            elif x_movement == 0 and y_movement == 0:
                motion_status = "StandIng Still"
            else:
                motion_status = "Kinda Moving"

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = pose.process(roi_rgb)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
            right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
            left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
            right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

            avg_knee_y = (left_knee_y + right_knee_y) / 2
            avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
            avg_wrist_y = (left_wrist_y + right_wrist_y) / 2

            if abs(avg_knee_y - avg_wrist_y) < 0.1 and avg_ankle_y > 0.9:
                motion_status = "Crawling"

            if len(prev_positions) >= 2:
                prev2_x, prev2_y = prev_positions[-2]
                y_movement_2 = abs(prev_y - prev2_y)
                x_movement_2 = abs(prev_x - prev2_x)
                if (y_movement > SIGNIFICANT_VERTICAL_MOVEMENT and x_movement_2 > SIGNIFICANT_HORIZONTAL_MOVEMENT) or \
                   (x_movement > SIGNIFICANT_HORIZONTAL_MOVEMENT and y_movement_2 > SIGNIFICANT_VERTICAL_MOVEMENT):
                    motion_status = "Skipping"

            # asimetrično kretanje
            if abs(left_knee_y - right_knee_y) > 0.05 and abs(left_ankle_y - right_ankle_y) > 0.05:
                motion_status = "Limping"

    return motion_status

# Procesiramo frameove
annotated_frames = []
frame_count = len(videoreader)
print(f"Processing {frame_count} frames...")

for idx, frame in enumerate(videoreader):
    if idx % 1 == 0:  # ovo mi ne treba, doda san napušto, čisto da vidin kako nepreduje
        print(f"Processing frame {idx}/{frame_count}")

    frame_rgb = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_BGR2RGB)
   
    results = model(frame_rgb)
    
    frame_with_classification = frame_rgb.copy()
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = map(int, result)
        if results.names[cls] == 'person':
            roi = frame_rgb[y1:y2, x1:x2]
            prev_x, prev_y = prev_positions.pop(0) if prev_positions else (x1, y1)
            motion_status = classify_movement(roi, x1, y1, prev_x, prev_y)
            prev_positions.append((x1, y1))
            label = f'{motion_status}'
            cv2.rectangle(frame_with_classification, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame_with_classification, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    annotated_frames.append(cv2.cvtColor(frame_with_classification, cv2.COLOR_RGB2BGR))

# Save 
output_video_path = 'annotated_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width, _ = annotated_frames[0].shape
out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))

for frame in annotated_frames:
    out.write(frame)

out.release()

print(f'Annotated video saved to {output_video_path}')
