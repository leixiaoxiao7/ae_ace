import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


# https://www.geeksforgeeks.org/deep-learning/ml-autoencoder-with-tensorflow-2-0/


if __name__ == "__main__":


    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    latent_dim = 32 

    encoder_input = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same")(x)

    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation="relu")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)


    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    autoencoder.fit(
        x_train, x_train,
        epochs=10,
        batch_size=128,
        shuffle=True,
        validation_data=(x_test, x_test)
    )

    encoded_imgs = autoencoder.predict(x_test)
    n = 10  

    plt.figure(figsize=(20, 4))
    for i in range(n):

        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(encoded_imgs[i].reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.show()
