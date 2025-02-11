import cv2
import numpy as np
from keras.models import load_model
from utils import preprocess_image, desired_width, desired_height

# Load the pre-trained emotion detection model
model = load_model("emotion_model_complex.h5")

# Define the list of emotions
emotions = ["happiness", "sadness", "fear", "anger", "contempt", "disgust", "neutrality", "surprise"]

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video from the default camera (adjust the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) which is the face
        face_roi = gray_frame[y:y + h, x:x + w]

        # Preprocess the face image
        preprocessed_face = preprocess_image(face_roi, desired_width, desired_height)

        # Set a confidence threshold (adjust as needed)
        confidence_threshold = 0.5

        # Predict emotion using the loaded model
        emotion_probs = model.predict(np.expand_dims(preprocessed_face, axis=0))

        # Check if the maximum predicted probability is above the threshold
        if np.max(emotion_probs) > confidence_threshold:
            emotion_label = emotions[np.argmax(emotion_probs)]
        else:
            emotion_label = "Not confident"

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the emotion label on the frame
        cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
