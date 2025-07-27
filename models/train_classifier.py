from ultralytics import YOLO
import os
work_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(work_dir)

def train_model(data_path, epochs=50, img_size=224):
    model = YOLO('yolov8n-cls.pt')  # Use other variants as needed
    model.train(data=data_path, epochs=epochs, imgsz=img_size)
    model.save('yolov8_eye_open_closed_classifier2.pt')

if __name__ == "__main__":
    train_path = os.path.join(parent_dir,'images/')
    train_model(train_path)
