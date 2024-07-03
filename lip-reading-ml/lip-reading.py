import cv2
import dlib
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from lipnet import LipNet
from lipnet.preprocessing import preprocess

<<<<<<< HEAD
# Path to the shape predictor model file
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Initialize dlib's face detector (HOG-based) and create a facial landmark predictor
=======

# Load the pre-trained LipNet model
model = LipNet()
model.load_weights('path_to_lipnet_weights.h5')

# Initialize dlib's face detector (HOG-based) and the facial landmark predictor
>>>>>>> 98f52aa (lip-subtitle)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
<<<<<<< HEAD
    print("Error: Could not open video capture device.")
=======
    print("Error: Camera could not be opened.")
>>>>>>> 98f52aa (lip-subtitle)
    exit()

print("Camera initialized successfully. Starting capture...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

<<<<<<< HEAD
    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale
=======
>>>>>>> 98f52aa (lip-subtitle)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        mouth_region = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 61)])

<<<<<<< HEAD
        # Extract lip coordinates
        lip_coords = []
        for n in range(48, 68):  # Outer and inner lips
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            lip_coords.append((x, y))
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

        # Print lip coordinates
        print("Lip coordinates:", lip_coords)

    # Display the resulting frame
=======
        # Extract the mouth region from the frame
        mouth_frame = frame[mouth_region[:, 1].min():mouth_region[:, 1].max(),
                      mouth_region[:, 0].min():mouth_region[:, 0].max()]

        # Preprocess the mouth frame for the model
        mouth_frame = cv2.resize(mouth_frame, (100, 50))  # Resize to model input size
        mouth_frame = img_to_array(mouth_frame) / 255.0  # Normalize
        mouth_frame = np.expand_dims(mouth_frame, axis=0)  # Add batch dimension

        # Predict text using the lip reading model
        prediction = model.predict(mouth_frame)
        predicted_text = decode_prediction(prediction)  # Function to decode the prediction

        # Displaying the predicted text
        cv2.putText(frame, predicted_text, (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

>>>>>>> 98f52aa (lip-subtitle)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
