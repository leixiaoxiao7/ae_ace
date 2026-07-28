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

# ✅ 保持 (N, 28, 28)，不做任何 reshape
nf = 0.8
x_train_noisy = x_train_2s * nf
x_test_noisy  = x_test_2s * nf

x_train_noisy += np.random.normal(loc=0, scale=1, size=x_train_2s.shape)
x_test_noisy  += np.random.normal(loc=0, scale=1, size=x_test_2s.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy,  0., 1.)

plt.imshow(x_train_noisy[0])


# ✅ Transformer Encoder Block
def transformer_encoder_block(x, num_heads, ff_dim, dropout=0.1):
    # --- Multi-Head Self-Attention ---
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads   # 每个 head 的维度
    )(x, x)                                # Q=K=V=x (self-attention)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)  # residual

    # --- Feed Forward ---
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)    # residual

    return x


# ✅ Transformer Decoder Block
def transformer_decoder_block(x, encoder_output, num_heads, ff_dim, dropout=0.1):
    # --- Masked Multi-Head Self-Attention ---
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads
    )(x, x)
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)  # residual

    # --- Cross-Attention（decoder 关注 encoder 输出）---
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads
    )(x, encoder_output)                  # Q=x, K=V=encoder_output
    attn_output = layers.Dropout(dropout)(attn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)  # residual

    # --- Feed Forward ---
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(x.shape[-1])(ff_output)
    ff_output = layers.Dropout(dropout)(ff_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)    # residual

    return x


# ✅ 建立模型，输入直接 (28, 28)
input_img = layers.Input(shape=(28, 28))   # (N, 28, 28)

# --- Encoder ---
# 先升维，让 Attention 有更多维度可以学习
enc = layers.Dense(64)(input_img)          # (N, 28, 64)
enc = transformer_encoder_block(enc, num_heads=4, ff_dim=128)   # (N, 28, 64)
enc = transformer_encoder_block(enc, num_heads=4, ff_dim=128)   # (N, 28, 64)

# Bottleneck：压缩特征维度
bottleneck = layers.Dense(16, activation="relu")(enc)           # (N, 28, 16)

# --- Decoder ---
dec = layers.Dense(64)(bottleneck)         # (N, 28, 64)
dec = transformer_decoder_block(dec, enc, num_heads=4, ff_dim=128)  # (N, 28, 64)
dec = transformer_decoder_block(dec, enc, num_heads=4, ff_dim=128)  # (N, 28, 64)

# ✅ 输出还原回 (N, 28, 28)，无任何 flatten
output = layers.Dense(28, activation="sigmoid")(dec)            # (N, 28, 28)

autoencoder = models.Model(input_img, output)
autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.summary()

# Train
autoencoder.fit(x_train_noisy, x_train_2s, epochs=100, batch_size=128)

decoded_imgs = autoencoder.predict(x_test_noisy)  # (N, 28, 28)

n = 10
plt.figure(figsize=(6, 20))
for i in range(n):
    ax = plt.subplot(10, 3, 3*i + 1)
    plt.imshow(x_test_2s[i])
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(10, 3, 3*i + 2)
    plt.imshow(x_test_noisy[i])
    plt.title("Noisy")
    plt.axis("off")

    ax = plt.subplot(10, 3, 3*i + 3)
    plt.imshow(decoded_imgs[i])
    plt.title("Denoised")
    plt.axis("off")

plt.show()