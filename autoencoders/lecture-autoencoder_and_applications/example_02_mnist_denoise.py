"""
@author: XXL
@lecturer: Sreenivas Bhattiprolu
"""

from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import numpy as np


(x_train, _), (x_test, _) = mnist.load_data()  # https://keras.io/datasets/

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

# Add artificial noise.
# Random noise from normal distribution with mean at 0 and std dev of 1.
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip all values to between 0 and 1.
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# Display images with noise.
plt.figure(figsize=(20, 2))
for i in range(1, 10):
    ax = plt.subplot(1, 10, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")
plt.show()


# Model bbilding 

# Sequential(): builds the neural network as a simple stack of layers.
# Each model.add(...) appends one new layer after the previous layer.

model = Sequential()

# Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1))
# Conv2D: learns local image patterns with sliding convolution filters.
# 32: number of filters; output has 32 feature maps.
# (3, 3): each filter looks at a 3-by-3 pixel neighborhood.
# activation='relu': replaces negative values with 0; keeps useful positive features.
# padding='same': pads image borders so height and width stay 28 x 28.
# input_shape=(28, 28, 1): one MNIST grayscale image, with 1 channel.
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))

# MaxPooling2D((2, 2), padding='same')
# MaxPooling2D: downsamples each feature map by keeping the strongest value in each window.
# (2, 2): pooling window size; roughly halves height and width.
# padding='same': pads edges when needed so the downsampled shape is rounded safely.
model.add(MaxPooling2D((2, 2), padding='same'))

# Conv2D(8, (3, 3), activation='relu', padding='same')
# 8 filters: compresses from 32 feature maps to 8 feature maps.
# This forces the model to keep only compact denoising features.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

# Second pooling layer: further compresses spatial size.
# The model removes detail while keeping the strongest learned features.
model.add(MaxPooling2D((2, 2), padding='same'))

# Another 8-filter convolution: refines features at the compressed resolution.
# Same parameters as above: 3x3 filters, ReLU nonlinearity, same spatial size.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 

# Third pooling layer: creates the smallest encoded representation.
# This bottleneck should retain digit structure while discarding some random noise.
model.add(MaxPooling2D((2, 2), padding='same'))
 
# Conv2D(8, ...): learns how to transform bottleneck features before upsampling.
# It keeps 8 feature maps and preserves the current height/width with padding='same'.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

# UpSampling2D((2, 2)): repeats rows and columns to double height and width.
# (2, 2): scale factor for vertical and horizontal dimensions.
model.add(UpSampling2D((2, 2)))

# Conv2D(8, ...): cleans/refines the enlarged feature maps after upsampling.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

# Second upsampling layer: doubles height and width again.
model.add(UpSampling2D((2, 2)))

# Conv2D(32, (3, 3), activation='relu')
# 32 filters: expands back to richer feature maps before the final grayscale output.
# (3, 3): each filter uses a 3-by-3 neighborhood.
# activation='relu': keeps reconstructed features non-negative.
# No padding argument is given here, so Keras uses padding='valid' by default.
# padding='valid': does not pad borders; this reduces height and width by 2 pixels.
model.add(Conv2D(32, (3, 3), activation='relu'))

# Third upsampling layer: doubles height and width again after the valid convolution.
model.add(UpSampling2D((2, 2)))

# Conv2D(1, ...): final reconstructed grayscale image layer.
# 1 filter produces 1 output channel, matching MNIST grayscale format.
# activation='relu' keeps output non-negative; padding='same' keeps the current size unchanged.
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

# optimizer='adam': adaptive gradient optimizer used to update network weights.
# loss='mean_squared_error': penalizes pixel-wise difference between clean image and reconstruction.
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

model.fit(
    x_train_noisy,
    x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    verbose=1,
    validation_data=(x_test_noisy, x_test),
)

model.evaluate(x_test_noisy, x_test)

model.save("./models/denoising_autoencoder.keras")

no_noise_img = model.predict(x_test_noisy)

plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="binary")

    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 + i + 1)
    plt.imshow(no_noise_img[i].reshape(28, 28), cmap="binary")

plt.show()