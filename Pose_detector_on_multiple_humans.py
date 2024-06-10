import cv2
import numpy as np

net = cv2.dnn.readNet("yolo-coco/yolov4.weights", "yolo-coco/yolov4.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("yolo-coco/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

person_class_id = classes.index("person")

cap = cv2.VideoCapture(0)  

prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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

            roi = frame[y:y+h, x:x+w]

            if prev_frame is not None and roi.size != 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                frame_diff = cv2.absdiff(prev_frame[y:y+h, x:x+w], gray_roi)
                _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

                motion_pixels = cv2.countNonZero(frame_diff)
                motion_detected = motion_pixels > 80 # bilo je 3000
                motion_status = "Moving" if motion_detected else "Standing still"

                print("Motion status:", motion_status)

                cv2.putText(frame, f"Motion: {motion_status}", (x + w - 250 , y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()