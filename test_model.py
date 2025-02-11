import cv2
import numpy as np
from keras.models import load_model
from utils import preprocess_image  # Import the preprocess_image function from utils

# Load the pre-trained model
model_path = r'C:\Users\user\OneDrive\Desktop\face_recognition\e.h5'
model = load_model(model_path)

# Assuming you have an image file path
image_path = r'C:\Users\user\OneDrive\Desktop\ap.jpeg'


# Read the image
frame = cv2.imread(image_path)

# Preprocess the frame using the function from utils
preprocessed_frame = preprocess_image(frame)

def predict_emotion(model, input_data):
    # Make predictions
    predictions = model.predict(np.array([input_data]))

    # Process the predictions as needed for your application
    # (e.g., get the emotion label from the predictions)
    # For example, if your model predicts probabilities for different emotions:
    emotion_label = np.argmax(predictions)

    return emotion_label

# Example usage:
# For each frame:
predicted_emotion = predict_emotion(model, preprocessed_frame)
# Use the result as needed in your application
