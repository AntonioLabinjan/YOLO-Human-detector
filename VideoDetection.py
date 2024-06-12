# PORADIT NA FINE-TUNINGU
import cv2
import numpy as np
import threading
import mediapipe as mp

# Load YOLO model
net = cv2.dnn.readNet("yolo-coco/yolov4-tiny.weights", "yolo-coco/yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

person_class_id = classes.index("person")

video_path = "Big_test.mp4"  # Provide the path to your video file
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

prev_frame = None
prev_boxes = []
prev_heights = []

frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_width, resize_height = 416, 416

def process_frame(frame):
    global prev_frame, prev_boxes, prev_heights

    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (resize_width, resize_height), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == person_class_id:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            roi = frame[y:y + h, x:x + w]

            motion_status = "really slow walking"  # Default status

            if prev_frame is not None and roi.size != 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(prev_frame[y:y + h, x:x + w], gray_roi)
                _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(frame_diff)

                if motion_pixels > 1000:  # Significant motion detected
                    if i < len(prev_boxes):
                        prev_y = prev_boxes[i][1]
                        y_movement = abs(y - prev_y)
                        if y_movement > 90:  # Significant vertical movement
                            motion_status = "Jumping"
                        elif y_movement > 5:  # Moderate vertical movement
                            motion_status = "faster Walking"
                    else:
                        motion_status = "kinda moving"
                else:
                    motion_status = "Standing still"  # No significant motion detected

                # Apply pose estimation for more precise crawling detection
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = pose.process(roi_rgb)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # Extract y-coordinates of key points
                    left_knee_y = landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y
                    right_knee_y = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y
                    left_ankle_y = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y
                    right_ankle_y = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y
                    left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
                    right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y

                    # Average y-coordinates of knees and wrists
                    avg_knee_y = (left_knee_y + right_knee_y) / 2
                    avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
                    avg_wrist_y = (left_wrist_y + right_wrist_y) / 2

                    if abs(avg_knee_y - avg_wrist_y) < 0.1 and avg_ankle_y > 0.9:  # Adjust these thresholds as needed
                        motion_status = "Crawling"

            cv2.putText(frame, f"Motion: {motion_status}", (x + w - 250, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_boxes = boxes
    prev_heights = [box[3] for box in boxes]

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    return frame

def capture_and_process():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        cv2.imshow("Image", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture_thread = threading.Thread(target=capture_and_process)
capture_thread.start()
capture_thread.join()

cap.release()
cv2.destroyAllWindows()
