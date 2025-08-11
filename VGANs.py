import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Constants
IMG_ROWS = 64
IMG_COLS = 64
CHANNELS = 3
IMG_SHAPE = (IMG_ROWS, IMG_COLS, CHANNELS)
NOISE_DIM = 100

# Build Generator
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))  # 8x8x256

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  # 16x16x128

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  # 32x32x64

    model.add(layers.Conv2DTranspose(CHANNELS, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')) 
    # Output: 64x64x3

    return model

# Build Discriminator
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=IMG_SHAPE))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

# Loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output) 
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output) 
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Prepare models and optimizers
generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([images.shape[0], NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Generate and save images during training for preview
def generate_and_save_images(model, epoch, test_input=None):
    if test_input is None:
        test_input = tf.random.normal([16, NOISE_DIM])
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        img = (predictions[i] + 1) / 2.0  # scale [-1,1] to [0,1]
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()

# Load and preprocess CelebA dataset
def load_celeba_dataset(batch_size=64):
    import tensorflow_datasets as tfds
    dataset, info = tfds.load('celeb_a', split='train', with_info=True)
    def preprocess(data):
        image = tf.image.resize(data['image'], (64,64))
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5  # normalize to [-1,1]
        return image
    dataset = dataset.map(preprocess).shuffle(10000).batch(batch_size)
    return dataset

# Training loop
def train(dataset, epochs=50, batch_size=64):
    for epoch in range(epochs):
        for image_batch in dataset:
            g_loss, d_loss = train_step(image_batch)

        print(f'Epoch {epoch+1}, Generator loss: {g_loss.numpy():.4f}, Discriminator loss: {d_loss.numpy():.4f}')
        generate_and_save_images(generator, epoch + 1)

    # Save generator model after training
    generator.save('generator_model.h5')
    print("Generator model saved as 'generator_model.h5'")

# Load generator model and generate new images
def load_generator_and_generate(num_images=16, noise_dim=NOISE_DIM):
    from tensorflow.keras.models import load_model

    loaded_generator = load_model('generator_model.h5', compile=False)
    print("Generator model loaded.")

    random_noise = tf.random.normal([num_images, noise_dim])
    generated_images = loaded_generator(random_noise, training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(num_images):
        plt.subplot(4, 4, i+1)
        img = (generated_images[i] + 1) / 2.0  # scale back to [0,1]
        plt.imshow(img)
        plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 50

    dataset = load_celeba_dataset(BATCH_SIZE)
    train(dataset, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # After training, generate some images with the saved model
    load_generator_and_generate(num_images=16)

