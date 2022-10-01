import cv2
import numpy as np
from gui_buttons import Buttons

# Initialize buttons
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("keyboard", 20, 180)
button.add_button("remote", 20, 260)
button.add_button("scissors", 20, 340)

colors = button.colors

# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255)

# Load classes file
classes = []
with open("dnn_model/classes.txt", "r") as f:
    for class_names in f.readlines():
        class_names = class_names.strip()
        classes.append(class_names)

# Getting the webcam and setting the rosolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 560)

# A function for mouse click affect
def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)


# Create Window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)

while True:
    # Get frames
    ret, frame = cap.read()

    # Get activate buttons list
    active_button_list = button.active_buttons_list()

    # Doing Object detection
    (class_ids, score, bboxes) = model.detect(frame)
    for class_ids, score, bbox in zip(class_ids, score, bboxes):
        (x, y, w, h) = bbox
        class_names = classes[class_ids]

        if class_names in active_button_list:
            cv2.putText(frame, class_names, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 3)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (200, 0, 50), 3)


    # Display buttons
    button.display_buttons(frame)

    # Showing Image
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

