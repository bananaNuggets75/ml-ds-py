import cv2
import dlib
import numpy as np
from lipnet.preprocessing import img_to_array
  # Ensure this function exists or is implemented in predict.py

# Initialize the dlib face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

# Load the LipNet model
model = load_model('models/trained_sequence_model.h5')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        mouth_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        mouth = frame[min(mouth_points[:, 1]):max(mouth_points[:, 1]), min(mouth_points[:, 0]):max(mouth_points[:, 0])]

        if mouth.size > 0:
            mouth_array = img_to_array(mouth)
            prediction = model.predict(np.expand_dims(mouth_array, axis=0))
            text = decode_prediction(prediction)

            # Draw the mouth rectangle
            cv2.rectangle(frame, (min(mouth_points[:, 0]), min(mouth_points[:, 1])),
                          (max(mouth_points[:, 0]), max(mouth_points[:, 1])), (0, 255, 0), 2)
            cv2.putText(frame, text, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Lip Reading', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
