import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

'''
VERY-bad effects!!!

'''

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_2s = x_train[y_train == 2]
x_test_2s = x_test[y_test == 2]

plt.imshow(x_train_2s[0])

# standardize numbers
x_train_2s = x_train_2s.astype('float32') / 255.0
x_test_2s = x_test_2s.astype('float32') / 255.0

# ✅ 不做任何 reshape，保持 (N, 28, 28)
nf = 0.8
x_train_noisy = x_train_2s * nf
x_test_noisy  = x_test_2s  * nf

x_train_noisy += np.random.normal(loc=0, scale=1, size=x_train_2s.shape)
x_test_noisy  += np.random.normal(loc=0, scale=1, size=x_test_2s.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy,  0., 1.)

plt.imshow(x_train_noisy[0])

# ✅ Input 直接 (28, 28)：28个特征群，每群28个特征
input_img = layers.Input(shape=(28, 28))

# Encoder
# TimeDistributed 对每个特征群独立做 Dense
# 权重 shape: (28_features_in, units_out)  ← 28 在权重里
encoded = layers.TimeDistributed(layers.Dense(16, activation="relu"))(input_img)
encoded = layers.TimeDistributed(layers.Dense(8,  activation="relu"))(encoded)
encoded = layers.TimeDistributed(layers.Dense(4,  activation="relu"))(encoded)

# Decoder
decoded = layers.TimeDistributed(layers.Dense(8,  activation="relu"))(encoded)
decoded = layers.TimeDistributed(layers.Dense(16, activation="relu"))(decoded)
decoded = layers.TimeDistributed(layers.Dense(28, activation="sigmoid"))(decoded)
# ✅ 输出 shape: (N, 28, 28)，无任何 flatten/reshape

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Train
autoencoder.fit(x_train_noisy, x_train_2s, epochs=100, batch_size=128)

decoded_imgs = autoencoder.predict(x_test_noisy)  # shape: (N, 28, 28)

n = 10
plt.figure(figsize=(6, 20))
for i in range(n):
    ax = plt.subplot(10, 3, 3*i + 1)
    plt.imshow(x_test_2s[i])      # ✅ 直接 (28, 28)
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(10, 3, 3*i + 2)
    plt.imshow(x_test_noisy[i])   # ✅ 直接 (28, 28)
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(10, 3, 3*i + 3)
    plt.imshow(decoded_imgs[i])   # ✅ 直接 (28, 28)
    plt.title("Denoised")
    plt.axis("off")

plt.show()