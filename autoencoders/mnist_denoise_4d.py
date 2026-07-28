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

# ✅ 改动1: 不再 flatten，改为添加 channel 维度 (N, 28, 28) -> (N, 28, 28, 1)
x_train_2s = x_train_2s[..., np.newaxis]   # shape: (N, 28, 28, 1)
x_test_2s  = x_test_2s[..., np.newaxis]    # shape: (N, 28, 28, 1)

# create noise
nf = 0.0001
x_train_noisy = x_train_2s * nf
x_test_noisy  = x_test_2s  * nf

# standardize-&-scale
x_train_noisy += np.random.normal(loc=0, scale=1, size=x_train_2s.shape)
x_test_noisy  += np.random.normal(loc=0, scale=1, size=x_test_2s.shape)

# standardize out <0 and >1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy,  0., 1.)

plt.imshow(x_train_noisy[0].reshape(28, 28))

# ✅ 改动2: 输入为3D格式 (28, 28, 1)，使用 Conv2D 构建卷积自编码器
input_img = layers.Input(shape=(28, 28, 1))

# Encoder（下采样）
encoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)   # (14, 14, 32)
encoded = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
encoded = layers.MaxPooling2D((2, 2), padding='same')(encoded)   # (7, 7, 16)

# Decoder（上采样）
decoded = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
decoded = layers.UpSampling2D((2, 2))(decoded)                   # (14, 14, 16)
decoded = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
decoded = layers.UpSampling2D((2, 2))(decoded)                   # (28, 28, 32)

# ✅ 改动3: 输出 1 个 channel，对应灰度图
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)  # (28, 28, 1)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Train the autoencoder（数据现在是3D，直接传入）
autoencoder.fit(x_train_noisy, x_train_2s, epochs=100, batch_size=128)

decoded_imgs = autoencoder.predict(x_test_noisy)

# 可视化（squeeze 掉 channel 维度用于显示）
n = 10
plt.figure(figsize=(6, 20))
for i in range(n):
    # Display original
    ax = plt.subplot(10, 3, 3*i + 1)
    plt.imshow(x_test_2s[i].squeeze())     # (28, 28, 1) -> (28, 28)
    plt.title("Original")
    plt.axis("off")

    # Display noisy
    ax = plt.subplot(10, 3, 3*i + 2)
    plt.imshow(x_test_noisy[i].squeeze())  # (28, 28, 1) -> (28, 28)
    plt.title("Noisy")
    plt.axis("off")

    # Display denoised
    ax = plt.subplot(10, 3, 3*i + 3)
    plt.imshow(decoded_imgs[i].squeeze())  # (28, 28, 1) -> (28, 28)
    plt.title("Denoised")
    plt.axis("off")

plt.show()