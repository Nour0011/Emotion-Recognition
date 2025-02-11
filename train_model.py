from PIL import Image
import numpy as np
from keras.models import load_model

# Load your face recognition model
model = load_model("path_to_your_model.h5")

# Load and preprocess an example image
image_path = "C:\\Users\\user\\OneDrive\\Desktop\\data_p\\surprise\\images - 2020-11-06T203445.294_face.png"
img = Image.open(image_path)

# Resize image to (128, 128) and convert to numpy array
img = img.resize((128, 128))
img_array = np.array(img)

# If the image is grayscale, convert it to RGB
if len(img_array.shape) == 2:
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.concatenate([img_array, img_array, img_array], axis=-1)

# Normalize pixel values to the range [0, 1]
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Make predictions using your model
predictions = model.predict(img_array)

# Display or use the predictions as needed
print(predictions)
