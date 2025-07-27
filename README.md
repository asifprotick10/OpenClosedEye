# Eye State Classifier (Open/Closed) using YOLOv8 + Dlib

This project uses YOLOv8 for classifying whether eyes are open or closed, with real-time detection using Dlib facial landmarks.

## Structure
- `models/train_classifier.py`: Train YOLO classifier
- `main.py`: Real-time webcam detection
- `detection/`: Eye region extraction and YOLO classification

## Downloads
Make sure you download the Dlib shape predictor model:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Download the images dataset from the following link:
https://drive.google.com/drive/folders/1jMjBGfs2GQDoiVwgxizsX6op6001PE_8?usp=sharing
## Installation
```bash
pip install -r requirements.txt

