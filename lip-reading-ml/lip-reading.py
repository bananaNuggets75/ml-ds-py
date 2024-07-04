import cv2
import dlib

# Path to the shape predictor model file
predictor_path = 'shape_predictor_68_face_landmarks.dat'

# Initialize dlib's face detector (HOG-based) and create a facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

print("Camera initialized successfully. Starting capture...")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        # Get the landmarks/parts for the face in box
        landmarks = predictor(gray, face)

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
    cv2.imshow('Frame', frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
