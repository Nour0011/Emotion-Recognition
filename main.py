import cv2
import numpy as np
from keras.models import load_model

from test_model import predict_emotion
from utils import preprocess_image

# Load your model (make sure to adjust the path accordingly)
model_path = r'C:\Users\user\OneDrive\Desktop\face_recognition_p\emotion_recognition_model.h5'
model = load_model(model_path)

# Open a connection to the camera (usually camera index 0 is the built-in laptop camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Make predictions on the preprocessed frame
    predictions = model.predict(np.array([preprocessed_frame]))

    # Get the predicted emotion label
    predicted_emotion = predict_emotion(model, np.array([preprocessed_frame]))
    print(f"Predicted Emotion Label: {predicted_emotion}")

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
