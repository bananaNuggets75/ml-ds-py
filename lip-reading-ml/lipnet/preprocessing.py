import cv2
import numpy as np

def img_to_array(image):
    resized_image = cv2.resize(image, (64, 64))
    array_image = resized_image.astype('float32') / 255.0  # Normalize the pixel values
    return array_image
