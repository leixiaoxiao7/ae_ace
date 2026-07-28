import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize pixel values to [0, 1] and flatten images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape to (samples, 784) for flat vector input
x_train = x_train.reshape((len(x_train), 784))
x_test = x_test.reshape((len(x_test), 784))

# Define the bottleneck/latent space dimension
latent_dim = 32  # Size of the latent space representation

# ========== ENCODER ==========
input_img = tf.keras.layers.Input(shape=(784,))
encoded = tf.keras.layers.Dense(128, activation='relu')(input_img)
encoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
# Bottleneck layer - this is the latent space representation
latent = tf.keras.layers.Dense(latent_dim, activation='relu')(encoded)

# Create encoder model
encoder = tf.keras.models.Model(input_img, latent, name='encoder')

# ========== DECODER ==========
# Separate input for the decoder (latent space)
latent_input = tf.keras.layers.Input(shape=(latent_dim,))
decoded = tf.keras.layers.Dense(64, activation='relu')(latent_input)
decoded = tf.keras.layers.Dense(128, activation='relu')(decoded)
decoded = tf.keras.layers.Dense(784, activation='sigmoid')(decoded)

# Create decoder model
decoder = tf.keras.models.Model(latent_input, decoded, name='decoder')

# ========== AUTOENCODER ==========
# Connect encoder output to decoder input
autoencoder_output = decoder(encoder(input_img))
autoencoder = tf.keras.models.Model(input_img, autoencoder_output, name='autoencoder')

# ========== COMPILE AND TRAIN ==========
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
history = autoencoder.fit(
    x_train, x_train,  # Input and target are the same
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# ========== VISUALIZE RESULTS ==========
# Encode and decode some test images
encoded_imgs = encoder.predict(x_test[:10])  # Get latent space representations
decoded_imgs = autoencoder.predict(x_test[:10])  # Get reconstructions

# Plot original, latent space, and reconstructed images
plt.figure(figsize=(20, 6))

for i in range(5):  # Display 5 examples
    # Original images
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.title("Original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Latent space (visualize as bar chart since it's 1D)
    ax = plt.subplot(3, 5, i + 1 + 5)
    plt.bar(range(latent_dim), encoded_imgs[i])
    plt.title(f"Latent Space\n({latent_dim} dimensions)")
    plt.ylim([0, max(encoded_imgs[i])])
    
    # Reconstructed images
    ax = plt.subplot(3, 5, i + 1 + 10)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.title("Reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()

# ========== INTERPOLATE IN LATENT SPACE ==========
print("\nLatent space dimension:", latent_dim)
print(f"Compression ratio: 784/{latent_dim} = {784/latent_dim:.1f}x compression")

# Example: Get latent representation of one image
sample_img = x_test[0:1]
latent_vector = encoder.predict(sample_img)
print(f"\nLatent vector for first test image shape: {latent_vector.shape}")
print(f"Latent vector values (first 5): {latent_vector[0][:5]}")