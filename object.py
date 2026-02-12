import cv2
import numpy as np
import time
from playsound import playsound
import threading

# ---------------- LOAD YOLO ----------------
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))

# ---------------- VIDEO SOURCE ----------------
cap = cv2.VideoCapture(0)

frame_skip = 5
frame_id = 0
prev_time = time.time()

# Alert control
alert_playing = False
TARGET_OBJECT = "person"   # Change object here


def play_alert():
    global alert_playing
    alert_playing = True
    playsound("alertmp3.wav")

    alert_playing = False


# ---------------- DETECTION LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    if frame_id % frame_skip != 0:
        continue

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    object_count = {}
    detected_target = False

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label_name = classes[class_ids[i]]
            confidence = confidences[i]

            # Counting
            object_count[label_name] = object_count.get(label_name, 0) + 1

            # Alert detection
            if label_name == TARGET_OBJECT:
                detected_target = True

            color = tuple(map(int, colors[class_ids[i]]))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label_name} {confidence:.2f}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    # Play alert sound
    if detected_target and not alert_playing:
        threading.Thread(target=play_alert).start()

    # Show object count
    y_pos = 60
    for obj, count in object_count.items():
        cv2.putText(frame, f"{obj}: {count}",
                    (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        y_pos += 30

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.imshow("YOLO Alert System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

