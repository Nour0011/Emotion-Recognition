import cv2
import numpy as np

desired_width = 128
desired_height = 128

def preprocess_image(image, desired_width, desired_height):
    resized_image = cv2.resize(image, (desired_width, desired_height))
    if len(resized_image.shape) == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    preprocessed_image = resized_image[:, :, np.newaxis]
    return preprocessed_image
