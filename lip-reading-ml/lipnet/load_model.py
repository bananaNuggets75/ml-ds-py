import tensorflow as tf
import dlib

def load_lip_reading_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def load_face_detector():
    face_detector = dlib.get_frontal_face_detector()
    return face_detector

def load_shape_predictor(predictor_path):
    shape_predictor = dlib.shape_predictor(predictor_path)
    return shape_predictor
