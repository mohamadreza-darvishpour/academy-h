# =========================================
# 17 Flower Classification using CNN (Keras)
# =========================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

train_dir = "./project/17flowers/17flowerclasses/train"
test_dir = "./project/17flowers/17flowerclasses/test"


# Parameters

img_size = (150, 150)
batch_size = 32
epochs = 3

# Data Preprocessing + Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# =========================================
# CNN Model
# =========================================

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))

model.add(layers.Dense(17, activation='softmax'))

# =========================================
# Compile Model
# =========================================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# =========================================
# Train Model
# =========================================

history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=test_data
)

# =========================================
# Save Model
# =========================================

model.save("flower_cnn_model.h5")

# =========================================
# Plot Accuracy
# =========================================

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# =========================================
# Prediction Function (Inference)
# =========================================

class_names = list(train_data.class_indices.keys())

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    print("Predicted Class:", class_names[class_index])

    plt.imshow(img)
    plt.axis("off")
    plt.show()

# =========================================
# Example Inference (External Images)
# =========================================

# predict_image("rose.jpg")
# predict_image("sunflower.jpg")
# predict_image("tulip.jpg")