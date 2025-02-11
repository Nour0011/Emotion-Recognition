import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Assuming you have the preprocess_image, desired_width, and desired_height functions defined in your 'utils' module
from utils import preprocess_image, desired_width, desired_height

# Learning rate schedule
def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch < 5:
        return initial_lr
    else:
        return initial_lr * np.exp(0.1 * (5 - epoch))

# Load and preprocess images function
def load_and_preprocess_images(emotion_folder, idx, desired_width, desired_height):
    emotion_images = []
    emotion_labels = []

    for image_name in os.listdir(emotion_folder):
        image_path = os.path.join(emotion_folder, image_name)
        print(f"Processing image: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale
        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        preprocessed_image = preprocess_image(image, desired_width, desired_height)

        emotion_images.append(preprocessed_image)
        emotion_labels.append(idx)

    return emotion_images, emotion_labels

# Define the path to the main dataset folder
dataset_path = r"C:\Users\user\OneDrive\Desktop\data_p"
output_model_path = "emotion_model_final.h5"

# Initialize lists to store images and labels
all_images = []
all_labels = []

# Iterate over emotion folders
emotions = ["happiness", "sadness", "fear", "anger", "contempt", "disgust", "neutrality", "surprise"]
for idx, emotion in enumerate(emotions):
    emotion_folder = os.path.join(dataset_path, emotion)

    # Load and preprocess images for the current emotion
    emotion_images, emotion_labels = load_and_preprocess_images(emotion_folder, idx, desired_width, desired_height)

    # Append the images and labels to the overall lists
    all_images.extend(emotion_images)
    all_labels.extend(emotion_labels)

# Convert lists to NumPy arrays
X = np.array(all_images)
y = np.array(all_labels)

# Shuffle the data
X, y = shuffle(X, y, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train_one_hot = to_categorical(y_train, num_classes=len(emotions))
y_test_one_hot = to_categorical(y_test, num_classes=len(emotions))

# Normalize pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Data augmentation
datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                             zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

# Build your more complex model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(desired_width, desired_height, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(emotions), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 20
batch_size = 32

# Learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)

# Checkpoint to save the best model during training
checkpoint = ModelCheckpoint(output_model_path, monitor='val_loss', save_best_only=True, mode='min')

# Train the model using data augmentation and generators
history = model.fit(datagen.flow(X_train, y_train_one_hot, batch_size=batch_size), epochs=num_epochs,
                    validation_data=(X_test, y_test_one_hot), callbacks=[checkpoint, lr_scheduler])

# Save the trained model
model.save(output_model_path)

# Evaluate the model
accuracy = model.evaluate(X_test, y_test_one_hot)[1]
print("Test Accuracy: {:.2f}%".format(accuracy * 100))

# Plotting the learning curves
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels=emotions)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print(classification_report(y_true, y_pred, target_names=emotions))
