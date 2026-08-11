import os
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from datasets import load_dataset

# =====================================================
# Parameters
# =====================================================

IMG_SIZE = 64
BATCH_SIZE = 128
LATENT_DIM = 100
EPOCHS = 50

SAVE_DIR = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# =====================================================
# Download CelebA from HuggingFace
# =====================================================

print("Loading CelebA dataset...")

hf_dataset = load_dataset(
    "flwrlabs/celeba",
    split="train"
)

print("Dataset Size:", len(hf_dataset))

# برای تست سریع:
# hf_dataset = hf_dataset.select(range(20000))

# =====================================================
# Preprocessing
# =====================================================

def preprocess(example):

    img = example["image"]

    img = img.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img).astype(np.float32)

    img = (img - 127.5) / 127.5

    return img


def generator_data():

    while True:

        idxs = np.random.permutation(len(hf_dataset))

        for i in range(0, len(idxs), BATCH_SIZE):

            batch_ids = idxs[i:i+BATCH_SIZE]

            batch = []

            for idx in batch_ids:

                try:
                    batch.append(
                        preprocess(hf_dataset[int(idx)])
                    )
                except:
                    pass

            if len(batch) > 0:
                yield np.array(batch, dtype=np.float32)


dataset = tf.data.Dataset.from_generator(
    generator_data,
    output_signature=tf.TensorSpec(
        shape=(None, IMG_SIZE, IMG_SIZE, 3),
        dtype=tf.float32
    )
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

# =====================================================
# Generator
# =====================================================

def build_generator():

    model = Sequential()

    model.add(
        layers.Input(shape=(LATENT_DIM,))
    )

    model.add(
        layers.Dense(
            8 * 8 * 256,
            use_bias=False
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Reshape((8, 8, 256))
    )

    model.add(
        layers.Conv2DTranspose(
            128,
            kernel_size=5,
            strides=2,
            padding="same",
            use_bias=False
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64,
            kernel_size=5,
            strides=2,
            padding="same",
            use_bias=False
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            32,
            kernel_size=5,
            strides=2,
            padding="same",
            use_bias=False
        )
    )

    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2D(
            3,
            kernel_size=5,
            padding="same",
            activation="tanh"
        )
    )

    return model

# =====================================================
# Discriminator
# =====================================================

def build_discriminator():

    model = Sequential()

    model.add(
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    )

    model.add(
        layers.Conv2D(
            64,
            kernel_size=5,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            128,
            kernel_size=5,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(
        layers.Conv2D(
            256,
            kernel_size=5,
            strides=2,
            padding="same"
        )
    )

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    model.add(
        layers.Dense(1)
    )

    return model


generator = build_generator()
discriminator = build_discriminator()

# =====================================================
# Loss
# =====================================================

cross_entropy = tf.keras.losses.BinaryCrossentropy(
    from_logits=True
)

def generator_loss(fake_output):

    return cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )

def discriminator_loss(real_output, fake_output):

    real_loss = cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss

# =====================================================
# Optimizers
# =====================================================

generator_optimizer = Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

discriminator_optimizer = Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

# =====================================================
# Train Step
# =====================================================

@tf.function
def train_step(real_images):

    batch_size = tf.shape(real_images)[0]

    noise = tf.random.normal(
        [batch_size, LATENT_DIM]
    )

    with tf.GradientTape() as gen_tape,\
         tf.GradientTape() as disc_tape:

        fake_images = generator(
            noise,
            training=True
        )

        real_output = discriminator(
            real_images,
            training=True
        )

        fake_output = discriminator(
            fake_images,
            training=True
        )

        gen_loss = generator_loss(
            fake_output
        )

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    gen_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    disc_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            gen_gradients,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            disc_gradients,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss

# =====================================================
# Save Samples
# =====================================================

seed = tf.random.normal(
    [16, LATENT_DIM]
)

def save_generated_images(epoch):

    predictions = generator(
        seed,
        training=False
    )

    predictions = (
        predictions * 127.5 + 127.5
    )

    predictions = tf.clip_by_value(
        predictions,
        0,
        255
    )

    predictions = predictions.numpy().astype(np.uint8)

    canvas = np.zeros(
        (
            IMG_SIZE * 4,
            IMG_SIZE * 4,
            3
        ),
        dtype=np.uint8
    )

    idx = 0

    for i in range(4):
        for j in range(4):

            canvas[
                i*IMG_SIZE:(i+1)*IMG_SIZE,
                j*IMG_SIZE:(j+1)*IMG_SIZE
            ] = predictions[idx]

            idx += 1

    Image.fromarray(canvas).save(
        os.path.join(
            SAVE_DIR,
            f"epoch_{epoch:03d}.png"
        )
    )

# =====================================================
# Training Loop
# =====================================================

steps_per_epoch = len(hf_dataset) // BATCH_SIZE

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    progress = tqdm(
        dataset.take(steps_per_epoch),
        total=steps_per_epoch
    )

    for batch in progress:

        g_loss, d_loss = train_step(batch)

        progress.set_postfix(
            G=float(g_loss),
            D=float(d_loss)
        )

    save_generated_images(epoch + 1)

    print(
        f"G Loss = {float(g_loss):.4f}, "
        f"D Loss = {float(d_loss):.4f}"
    )

# =====================================================
# Save GIF
# =====================================================

print("Creating GIF...")

frames = []

for file in sorted(os.listdir(SAVE_DIR)):

    if file.endswith(".png"):

        frames.append(
            imageio.imread(
                os.path.join(
                    SAVE_DIR,
                    file
                )
            )
        )

imageio.mimsave(
    "celeba_dcgan.gif",
    frames,
    fps=4
)

print("Training Finished.")
print("GIF Saved: celeba_dcgan.gif")