import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# Load fashion images dataset (Fashion MNIST)
(train_images, _), (test_images, _) = fashion_mnist.load_data()

# Preprocess images
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Define the encoder (image to latent space)
def build_encoder(input_shape, latent_dim):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Flatten()(x)
    latent_space = Dense(latent_dim)(x)
    return Model(inputs, latent_space, name='encoder')

# Define the decoder (latent space to image)
def build_decoder(latent_dim, output_shape):
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(7*7*64, activation='relu')(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    return Model(latent_inputs, outputs, name='decoder')

# Define the autoencoder (encoder + decoder)
def build_autoencoder(encoder, decoder):
    inputs = Input(shape=input_shape)
    latent_space = encoder(inputs)
    reconstructed = decoder(latent_space)
    return Model(inputs, reconstructed, name='autoencoder')

# Define hyperparameters
latent_dim = 64
input_shape = (28, 28, 1)
batch_size = 128
epochs = 20

# Build and compile the encoder, decoder, and autoencoder
encoder = build_encoder(input_shape, latent_dim)
decoder = build_decoder(latent_dim, input_shape)
autoencoder = build_autoencoder(encoder, decoder)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(train_images, train_images,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(test_images, test_images))

# Generate new fashion designs
latent_samples = np.random.normal(size=(10, latent_dim))
generated_images = decoder.predict(latent_samples)

# Visualize generated images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_images[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
