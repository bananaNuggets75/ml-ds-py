import cv2
import dlib
import os


model_path = os.path.join(os.path.dirname(__file__), "../models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)


def extract_landmarks_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    landmarks = []

    for rect in rects:
        shape = predictor(gray, rect)
        landmarks.append([(p.x, p.y) for p in shape.parts()])

    return landmarks[0] if landmarks else None


def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    landmarks_sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks_from_frame(frame)
        if landmarks:
            landmarks_sequence.append(landmarks)

    cap.release()
    return landmarks_sequence