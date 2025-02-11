# facial_emotion_recognition.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import preprocess_image

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Open a connection to the camera (0 represents the default camera, change it if needed)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0  # Normalize pixel values

    # Make a prediction
    predictions = model.predict(img)
    emotion_label = np.argmax(predictions)
    emotion = ["happiness", "sadness", "fear", "anger", "contempt", "disgust", "neutrality", "surprise"][emotion_label]

    # Display the emotion on the frame
    cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
