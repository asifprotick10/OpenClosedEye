import cv2
import dlib
from detection.dlib_utils import get_eye_region
from detection.classify_eye import classify_eye
import os
work_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(work_dir)
predictor_path = os.path.join(parent_dir,'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        l_idx, r_idx = list(range(36, 42)), list(range(42, 48))
        lx1, ly1, lx2, ly2 = get_eye_region(landmarks, l_idx)
        rx1, ry1, rx2, ry2 = get_eye_region(landmarks, r_idx)

        l_eye = frame[ly1-30:ly2+30, lx1-30:lx2+30]
        r_eye = frame[ry1-30:ry2+30, rx1-30:rx2+30]
        l_pred = classify_eye(l_eye) if l_eye.size else None
        r_pred = classify_eye(r_eye) if r_eye.size else None

        if not (l_pred and r_pred): continue
        label = (
            "both eyes open" if l_pred[1] == 0 and r_pred[1] == 0 else
            "left eye closed" if l_pred[1] == 1 and r_pred[1] == 0 else
            "right eye closed" if l_pred[1] == 0 and r_pred[1] == 1 else
            "both eyes closed"
        )
        cv2.rectangle(frame, (lx1-30, ly1-30), (lx2+30, ly2+30), (255, 0, 0), 2)
        cv2.rectangle(frame, (rx1-30, ry1-30), (rx2+30, ry2+30), (255, 0, 0), 2)
        cv2.putText(frame, label, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow('Eye Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
