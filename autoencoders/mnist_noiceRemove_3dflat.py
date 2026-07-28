import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_2s = x_train[y_train == 2]
x_test_2s  = x_test[y_test == 2]

plt.imshow(x_train_2s[0])
plt.title("Original Clean")
plt.show()

# standardize
x_train_2s = x_train_2s.astype('float32') / 255.0
x_test_2s  = x_test_2s.astype('float32')  / 255.0

# ── 噪声：调低 nf，避免原始信息被淹没 ──────────
nf = 0.3                                            # ✅ 原来 0.8 太强
x_train_noisy = x_train_2s * nf
x_test_noisy  = x_test_2s  * nf

x_train_noisy += np.random.normal(loc=0, scale=0.3, size=x_train_2s.shape)  # ✅ scale 也调低
x_test_noisy  += np.random.normal(loc=0, scale=0.3, size=x_test_2s.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy,  0., 1.)

plt.imshow(x_train_noisy[0])
plt.title("Noisy Input")
plt.show()

# ════════════════════════════════════════════════
# 模型架构：混合设计
#
# 数据流：
# (N,28,28)
#   → TimeDistributed Dense  : 每行内部特征压缩   [行独立，保留行维度]
#   → Flatten                : 打通行间信息        [全局交互]
#   → Dense bottleneck       : 全局最深压缩        [全局语义瓶颈]
#   → Dense expand           : 全局还原
#   → Reshape (28, 16)       : 恢复行结构
#   → TimeDistributed Dense  : 每行内部特征还原
#   → output (N, 28, 28)
# ════════════════════════════════════════════════

input_img = layers.Input(shape=(28, 28))             # (N, 28, 28)

# ── Encoder Part1：行内压缩（保留行维度语义）──────
e = layers.TimeDistributed(layers.Dense(28, activation='relu'))(input_img)  # (N, 28, 28)
e = layers.TimeDistributed(layers.Dense(16, activation='relu'))(e)          # (N, 28, 16)

# ── Encoder Part2：打通行间，全局深度压缩 ──────────
e = layers.Flatten()(e)                              # (N, 28×16=448)
e = layers.Dense(256, activation='relu')(e)          # (N, 256)
e = layers.Dense(128, activation='relu')(e)          # (N, 128)

# ── Bottleneck ─────────────────────────────────────
bottleneck = layers.Dense(64, activation='relu')(e)  # (N, 64)  ← 全局语义瓶颈

# ── Decoder Part1：全局还原 ────────────────────────
d = layers.Dense(128, activation='relu')(bottleneck) # (N, 128)
d = layers.Dense(256, activation='relu')(d)          # (N, 256)
d = layers.Dense(28 * 16, activation='relu')(d)      # (N, 448)

# ── Decoder Part2：恢复行结构，行内还原 ───────────
d = layers.Reshape((28, 16))(d)                      # (N, 28, 16)
d = layers.TimeDistributed(layers.Dense(28, activation='relu'))(d)      # (N, 28, 28)
d = layers.TimeDistributed(layers.Dense(28, activation='sigmoid'))(d)   # (N, 28, 28)

autoencoder = models.Model(input_img, d)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ── 训练 ───────────────────────────────────────────
history = autoencoder.fit(
    x_train_noisy, x_train_2s,
    epochs=100,
    batch_size=128,
    validation_split=0.1
)

# ── Loss 曲线 ──────────────────────────────────────
plt.figure(figsize=(8, 3))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('MSE Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# ── 可视化结果 ─────────────────────────────────────
decoded_imgs = autoencoder.predict(x_test_noisy)     # (N, 28, 28)

n = 10
plt.figure(figsize=(6, 20))
for i in range(n):
    ax = plt.subplot(10, 3, 3*i + 1)
    plt.imshow(x_test_2s[i], cmap='gray')
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(10, 3, 3*i + 2)
    plt.imshow(x_test_noisy[i], cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(10, 3, 3*i + 3)
    plt.imshow(decoded_imgs[i], cmap='gray')
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.show()