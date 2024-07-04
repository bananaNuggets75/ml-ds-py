import tensorflow as tf
import os


def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), "trained_sequence_model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path)
    return model
