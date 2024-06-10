import cv2
import numpy as np
import threading

net = cv2.dnn.readNet("yolo-coco/yolov4-tiny.weights", "yolo-coco/yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

person_class_id = classes.index("person")

video_path = "Hoomans.mp4" # stavimo bilo koji video tu
cap = cv2.VideoCapture(video_path)

prev_frame = None
frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_width, resize_height = 416, 416

def process_frame(frame):
    global prev_frame

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

            if prev_frame is not None and roi.size != 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                frame_diff = cv2.absdiff(prev_frame[y:y + h, x:x + w], gray_roi)
                _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
                motion_pixels = cv2.countNonZero(frame_diff)
                motion_detected = motion_pixels > 50 # bilo je 3000
                motion_status = "Moving" if motion_detected else "Standing still"
                cv2.putText(frame, f"Motion: {motion_status}", (x + w - 250, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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
