import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
# -------------------------------
# 1. تنظیم مسیرها و پارامترها
# -------------------------------
data_dir = "/content/17flowerclasses"            # مسیر ریشه دیتاست
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

IMG_SIZE = (128, 128)            # اندازه ورودی تصاویر
BATCH_SIZE = 32
NUM_CLASSES = 17
EPOCHS = 50                      # تعداد دوره‌ها (EarlyStopping قطع می‌کند)

# -------------------------------
# 2. آماده‌سازی داده‌ها با افزایش (Augmentation)
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,               # نرمال‌سازی پیکسل‌ها به [0,1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# برای مجموعه تست فقط نرمال‌سازی می‌کنیم (بدون افزایش)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# بررسی تعداد تصاویر در هر دسته
print(f"تعداد کلاس‌ها: {train_generator.num_classes}")
print(f"نام کلاس‌ها: {train_generator.class_indices}")

# -------------------------------
# 3. ساخت مدل CNN
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# -------------------------------
# 4. کامپایل مدل
# -------------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# 5. تعریف Callback‌ها برای جلوگیری از overfitting و ذخیره بهترین مدل
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_flower_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# -------------------------------
# 6. آموزش مدل
# -------------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,    # به عنوان validation از تست استفاده می‌شود
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# -------------------------------
# 7. ارزیابی نهایی روی مجموعه تست
# -------------------------------
test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print(f"\nدقت روی مجموعه تست: {test_acc:.4f}")

# -------------------------------
# 8. رسم نمودار دقت و تابع ضرر
# -------------------------------
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('مدل دقت')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('مدل تابع ضرر')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -------------------------------
# 9. بارگذاری بهترین مدل ذخیره‌شده (اختیاری)
# -------------------------------
# best_model = tf.keras.models.load_model('best_flower_model.h5')