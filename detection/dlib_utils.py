def get_eye_region(landmarks, eye_indices):
    eye_points = [landmarks.part(i) for i in eye_indices]
    x_coords = [pt.x for pt in eye_points]
    y_coords = [pt.y for pt in eye_points]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)