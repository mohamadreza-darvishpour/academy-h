"""
Correct ResNet50 inference with proper weight loading.
Assumes the correct weights file exists at don_models/resnet50_weights.h5
"""

import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# --------------------------
# 1. Paths and validation
# --------------------------
weights_path = r'C:\Users\beta\Documents\0-academy-hamrah\g_deep_learning\code_trains\resnet50_weights.h5'
img_path = r'C:\Users\beta\Documents\0-academy-hamrah\g_deep_learning\code_trains\44444.png'

if not os.path.exists(weights_path):
    print('\\n\n\nn\n\n\n\n**********   \n\\n\n\n')
    raise FileNotFoundError(
        f"Weights not found at {weights_path}. Please download the correct file "
        "from https://github.com/fchollet/deep-learning-models/releases/download/v0.2/"
        "resnet50_weights_tf_dim_ordering_tf_kernels.h5"
    )

if not os.path.exists(img_path):
    raise FileNotFoundError(f"Image not found: {img_path}")

# --------------------------
# 2. Build model and load weights
# --------------------------
model = ResNet50(weights=None)   # architecture only
try:
    model.load_weights(weights_path)
    print("\n\n\n\n\n✅ Weights loaded successfully.\n\n\n\n\n")
except Exception as e:
    print("\n\n\n\n\n❌ Failed to load weights. Check that the file matches ResNet50 architecture.")
    raise e

# --------------------------
# 3. Preprocess image
# --------------------------
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# --------------------------
# 4. Predict & decode
# --------------------------
preds = model.predict(x)
results = decode_predictions(preds, top=3)[0]

print("\nTop predictions:")
for label_id, label_name, prob in results:
    print(f"  {label_name}: {prob:.4f}")