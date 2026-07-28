import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_2s = x_train[y_train == 2]
x_test_2s = x_test[y_test == 2]

plt.imshow(x_train_2s[0])

# standardize numbers
x_train_2s = x_train_2s.astype('float32') / 255.0
x_test_2s = x_test_2s.astype('float32') / 255.0

# flatten the dataset
x_train_2s = x_train_2s.reshape((len(x_train_2s), 784))
x_test_2s = x_test_2s.reshape((len(x_test_2s), 784))

# create noice
nf = 0.0001
x_train_noisy = x_train_2s * nf
x_test_noisy = x_test_2s * nf
# standardize-&-scale
x_train_noisy += np.random.normal(loc=0, scale=1, size=x_train_2s.shape)
x_test_noisy += np.random.normal(loc=0, scale=1, size=x_test_2s.shape)

# standardize out <0 and >1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


plt.imshow(x_train_noisy[0].reshape(28, 28))

# Build the autoencoder
input_img = layers.Input(shape=(784,))

# Encoder
encoded = layers.Dense(128, activation="relu")(input_img)
encoded = layers.Dense(64, activation="relu")(encoded)
encoded = layers.Dense(32, activation="relu")(encoded)
encoded = layers.Dense(16, activation="relu")(encoded)

# Decoder
decoded = layers.Dense(32, activation="relu")(encoded)
decoded = layers.Dense(64, activation="relu")(decoded)
decoded = layers.Dense(128, activation="relu")(decoded)
decoded = layers.Dense(784, activation="sigmoid")(decoded)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# Train the autoencoder
autoencoder.fit(x_train_noisy, x_train_2s, epochs=100, batch_size=128)

decoded_imgs = autoencoder.predict(x_test_noisy)

n = 10
plt.figure(figsize=(6, 20))
for i in range(n):
    # Display original
    ax = plt.subplot(10, 3, 3*i + 1)
    plt.imshow(x_test_2s[i].reshape(28, 28))
    plt.title("Original")
    plt.axis("off")

    # Display noisy
    ax = plt.subplot(10, 3, 3*i + 2)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.title("Noisy")
    plt.axis("off")

    # Display denoised
    ax = plt.subplot(10, 3, 3*i + 3)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.title("Denoised")
    plt.axis("off")

plt.show()