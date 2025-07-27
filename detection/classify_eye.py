import cv2
from ultralytics import YOLO
import os

work_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(work_dir)
model = YOLO(os.path.join(parent_dir,'models/yolov8_eye_open_closed_classifier.pt'))

def classify_eye(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model([image])
    return results[0].probs.top5
