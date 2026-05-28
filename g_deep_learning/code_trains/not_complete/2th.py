# ==========================
# 1. Import TensorFlow's Keras module
# ==========================
from tensorflow import keras

# ==========================
# 2. Load the Xception base model (pretrained on ImageNet)
# ==========================
# weights='imagenet'   : use pretrained ImageNet weights
# input_shape=(150,150,3) : size of input images (height, width, channels)
# include_top=False    : exclude the original classification head (fully connected layers)
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False
)

# ==========================
# 3. Freeze the base model so its weights are not updated during training
# ==========================
base_model.trainable = False

# ==========================
# 4. Define a new input layer
# ==========================
inputs = keras.Input(shape=(150, 150, 3))

# ==========================
# 5. Pass the input through the frozen base model
# ==========================
# training=False : ensures the base model runs in inference mode (e.g., no dropout, batch norm uses moving stats)
# This is good practice even though the model is frozen.
x = base_model(inputs, training=False)

# ==========================
# 6. Apply Global Average Pooling to reduce spatial dimensions to a vector
# ==========================
x = keras.layers.GlobalAveragePooling2D()(x)

# ==========================
# 7. Add a single output neuron (binary classification)
# ==========================
# Dense(1) produces logits (no activation) because we will use from_logits=True in the loss
outputs = keras.layers.Dense(1)(x)

# ==========================
# 8. Create the final model
# ==========================
model = keras.Model(inputs, outputs)

# ==========================
# 9. Define the loss function
# ==========================
# BinaryCrossentropy with from_logits=True expects raw logits (not sigmoid applied)
loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)

# ==========================
# 10. Define the optimizer
# ==========================
optimizer = keras.optimizers.Adam()

# ==========================
# 11. Compile the model (missing in your original code)
# ==========================
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

# ==========================
# 12. (Optional) Display model architecture summary
# ==========================
model.summary()