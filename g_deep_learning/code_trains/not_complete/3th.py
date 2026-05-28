# 1. Import the VGG16 model architecture from Keras applications
from tensorflow.keras.applications.vgg16 import VGG16

# 2. Import the image preprocessing utilities (loading, resizing, etc.)
from tensorflow.keras.preprocessing import image

# 3. Import the VGG16‑specific preprocessing function (normalization)
from tensorflow.keras.applications.vgg16 import preprocess_input

# 4. Import NumPy for array operations
import numpy as np

# 5. Load the VGG16 model with ImageNet weights, excluding the final classification layers
#    include_top=False → outputs feature maps (convolutional activations) instead of class probabilities
#    The default input size for VGG16 with include_top=False is (224, 224, 3)
model = VGG16(weights='imagenet', include_top=False)

# 6. Define the path to the input image
img_path = 'elephant.jpg'   # Make sure this file exists in the working directory

# 7. Load the image and resize it to 224x224 pixels (required input size for VGG16)
img = image.load_img(img_path, target_size=(224, 224))

# 8. Convert the PIL image to a NumPy array of shape (224, 224, 3)
x = image.img_to_array(img)

# 9. Add a batch dimension (axis=0) → shape becomes (1, 224, 224, 3)
x = np.expand_dims(x, axis=0)

# 10. Preprocess the image: subtract the mean RGB values (calculated on ImageNet)
#     and possibly scale the pixels (depends on the backend; for VGG16 it does mean subtraction)
x = preprocess_input(x)

# 11. Pass the preprocessed image through the model to extract features
#     The output `features` is a NumPy array of shape (1, height, width, channels)
#     For VGG16 with input 224x224 and include_top=False, the output shape is (1, 7, 7, 512)
features = model.predict(x)   # removed the trailing comma (was a syntax error)

# Optional: print the shape of the extracted features
print("Feature map shape:", features.shape)