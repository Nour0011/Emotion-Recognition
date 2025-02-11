import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

from prepare_dataset import X_train, X_test, y_test, y_train, emotions

# Define the emotion recognition model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))  # 8 classes for 8 emotions

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train.reshape(-1, 64, 64, 1), to_categorical(y_train), epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
y_pred = model.predict(X_test.reshape(-1, 64, 64, 1))
accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))
print("Test Accuracy:", accuracy)

# Save the trained model
model.save("emotion_recognition_model.h5")

# Open a connection to the camera (0 represents the default camera, you can change it if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (64, 64))
    frame = frame / 255.0  # Normalize pixel values to the range [0, 1]
    frame = np.expand_dims(frame, axis=-1)  # Add channel dimension

    # Make predictions using the trained model
    emotion_prediction = model.predict(np.array([frame]))
    predicted_emotion = np.argmax(emotion_prediction)

    # Display the predicted emotion on the frame
    cv2.putText(frame, emotions[predicted_emotion], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Emotion Recognition', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
