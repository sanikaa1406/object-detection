import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
import winsound



# ---------------- ALERT SOUND ----------------
alert_playing = False

def play_alert():
    global alert_playing
    alert_playing = True

    winsound.Beep(1000, 500)   # frequency, duration

    alert_playing = False



# ---------------- LOAD YOLO ----------------
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(1)

running = False

# ---------------- GUI WINDOW ----------------
root = tk.Tk()
root.title("Object Detection Dashboard")
root.geometry("800x600")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Status: Stopped", font=("Arial", 14))
status_label.pack()

# ---------------- DETECTION FUNCTION ----------------
def detect():
    global running, cap

    cap = cv2.VideoCapture(0)  # laptop camera

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape

        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []
        detected_target = False

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
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

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]

                # ALERT if person detected
                if label == "person":
                    detected_target = True

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, label, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # Play alert
        if detected_target and not alert_playing:
            threading.Thread(target=play_alert).start()

        # Convert frame for tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    cap.release()

# ---------------- BUTTON FUNCTIONS ----------------
def start_detection():
    global running
    if not running:
        running = True
        status_label.config(text="Status: Running")
        threading.Thread(target=detect).start()

def stop_detection():
    global running
    running = False
    status_label.config(text="Status: Stopped")

# ---------------- BUTTONS ----------------
Button(root, text="Start Detection", command=start_detection, bg="green", fg="white", width=20).pack(pady=10)
Button(root, text="Stop Detection", command=stop_detection, bg="red", fg="white", width=20).pack(pady=10)

root.mainloop()
