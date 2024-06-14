# Donekle dela, opet poboljšat prompt
# Imamo previše kinda moving => malo više precizirat klase, potencijalno ih dodat više

# Install required packages
!pip install -q pytube decord
!pip install -q torch torchvision
!pip install -q mediapipe

# Download the video using pytube
from pytube import YouTube
import time

youtube_url = 'https://youtu.be/WY9HKBe8dF0'  # Replace with your video URL
yt = YouTube(youtube_url)
streams = yt.streams.filter(file_extension='mp4')
file_path = streams[0].download()

# Read the video using decord
from decord import VideoReader, cpu
import numpy as np

videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
video_fps = videoreader.get_avg_fps()

# Load YOLOv5 model
import torch
import cv2

start_time = time.time()
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# Initialize MediaPipe Pose model
import mediapipe as mp
from collections import deque, Counter

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Define parameters for motion classification
motion_status_window = deque(maxlen=5)
MOTION_THRESHOLD = 500  # Increased sensitivity for standing still
SIGNIFICANT_VERTICAL_MOVEMENT = 15  # Increased sensitivity for jumping
MODERATE_VERTICAL_MOVEMENT = 3  # Increased sensitivity for kinda moving
SIGNIFICANT_HORIZONTAL_MOVEMENT = 15  # Sensitivity for horizontal movement

prev_positions = []

def classify_movement(roi, x, y, prev_x, prev_y):
    global prev_positions, motion_status_window

    motion_status = "really slow walking"  # Default status

    if roi.size != 0:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_y, gray_roi)
        _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        motion_pixels = cv2.countNonZero(frame_diff)

        if motion_pixels > MOTION_THRESHOLD:  # Significant motion detected
            y_movement = abs(y - prev_y)
            x_movement = abs(x - prev_x)
            if y_movement > SIGNIFICANT_VERTICAL_MOVEMENT:  # Significant vertical movement
                motion_status = "Jumping"
            elif y_movement > MODERATE_VERTICAL_MOVEMENT or x_movement > SIGNIFICANT_HORIZONTAL_MOVEMENT:  # Moderate vertical or significant horizontal movement
                motion_status = "faster Walking"
            else:
                motion_status = "kinda moving"
        else:
            motion_status = "standing still"  # No significant motion detected

        # Apply pose estimation for more precise crawling detection
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

    return motion_status

# Process each frame of the video
annotated_frames = []
frame_count = len(videoreader)
print(f"Processing {frame_count} frames...")

for idx, frame in enumerate(videoreader):
    if idx % 1 == 0:  # Print progress for each frame
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

# Save the annotated video
output_video_path = 'annotated_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width, _ = annotated_frames[0].shape
out = cv2.VideoWriter(output_video_path, fourcc, video_fps, (width, height))

for frame in annotated_frames:
    out.write(frame)

out.release()

print(f'Annotated video saved to {output_video_path}')
