import os
import numpy as np
import tensorflow as tf
from .landmark import extract_landmarks_from_video

# Load the pre-trained sequence model
model_path = os.path.join(os.path.dirname(__file__), "../models/trained_sequence_model.h5")
model = tf.keras.models.load_model(model_path)

def predict_sequence(landmarks_sequence):
    landmarks_sequence = np.array(landmarks_sequence).reshape(1, -1, 68, 2)

    # Predict using the model
    prediction = model.predict(landmarks_sequence)
    return prediction

def decode_prediction(prediction):
    # Decode the model prediction to readable text
    # This is a placeholder, implement according to your model's output
    text = ''.join([chr(p) for p in np.argmax(prediction, axis=-1)])
    return text


def main():
    video_path = "path/to/your/video/file.mp4"
    landmarks_sequence = extract_landmarks_from_video(video_path)
    prediction = predict_sequence(landmarks_sequence)
    print(prediction)