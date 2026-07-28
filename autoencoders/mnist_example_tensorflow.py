import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

# Load and preprocess MNIST data
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

latent_dim = 32  # Size of the latent space

# ========== ENCODER ==========
encoder_input = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoder_input)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2), padding="same")(x)

# Flatten to vector before bottleneck
x = layers.Flatten()(x)

# LATENT SPACE (BOTTLENECK) - this is the key!
latent = layers.Dense(latent_dim, activation="relu")(x)

# Create encoder model
encoder = keras.Model(encoder_input, latent, name="encoder")

# ========== DECODER ==========
# Input for decoder is the latent space vector
decoder_input = keras.Input(shape=(latent_dim,))

# Reshape to start the convolutional reconstruction
x = layers.Dense(4 * 4 * 128, activation="relu")(decoder_input)  # 4x4x128 = 2048
x = layers.Reshape((4, 4, 128))(x)  # Reshape to match encoder's last conv shape

# Start upsampling/decoding
x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)  # 4x4 -> 8x8
x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)  # 8x8 -> 16x16
x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = layers.UpSampling2D((2, 2))(x)  # 16x16 -> 32x32

# Crop to 28x28 (since we went to 32x32)
x = layers.Cropping2D(((2, 2), (2, 2)))(x)  # Alternative: use padding="valid"

# Final output layer
decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Create decoder model
decoder = keras.Model(decoder_input, decoded, name="decoder")

# ========== AUTOENCODER ==========
# Connect encoder to decoder
autoencoder_output = decoder(encoder(encoder_input))
autoencoder = keras.Model(encoder_input, autoencoder_output, name="autoencoder")

# ========== COMPILE AND TRAIN ==========
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# ========== VISUALIZE RESULTS ==========
# Get reconstructions
decoded_imgs = autoencoder.predict(x_test)

# Get latent space representations
latent_representations = encoder.predict(x_test[:10])

# Plot results
n = 10
plt.figure(figsize=(20, 8))

for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap="gray")
    plt.title("Original")
    plt.axis("off")
    
    # Latent space (visualize as bar chart)
    ax = plt.subplot(3, n, i + 1 + n)
    plt.bar(range(latent_dim), latent_representations[i])
    plt.title(f"Latent ({latent_dim}D)")
    plt.ylim([0, np.max(latent_representations[i])])
    
    # Reconstructed images
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].squeeze(), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()

# ========== LATENT SPACE INFO ==========
print(f"Encoder output shape (latent space): {encoder.output_shape}")
print(f"Latent dimension: {latent_dim}")
print(f"Compression: 28x28x1 = {28*28} values → {latent_dim} values")
print(f"Compression ratio: {28*28/latent_dim:.1f}x")