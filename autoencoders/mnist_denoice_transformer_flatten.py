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

# ── 调低噪声强度 ───────────────────────────────────
nf = 0.1                                              # ✅ 原来 0.8 太强
x_train_noisy = x_train_2s * nf
x_test_noisy  = x_test_2s  * nf

x_train_noisy += np.random.normal(loc=0, scale=0.3, size=x_train_2s.shape)
x_test_noisy  += np.random.normal(loc=0, scale=0.3, size=x_test_2s.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy  = np.clip(x_test_noisy,  0., 1.)

plt.imshow(x_train_noisy[0], cmap='gray')
plt.title("Noisy Input")
plt.show()


# ════════════════════════════════════════════════
# Transformer Block（行间 Self-Attention）
# ════════════════════════════════════════════════
def transformer_encoder_block(x, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads
    )(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def transformer_decoder_block(x, encoder_output, num_heads, ff_dim, dropout=0.1):
    # Self-Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads
    )(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Cross-Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=x.shape[-1] // num_heads
    )(x, encoder_output)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed Forward
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(x.shape[-1])(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


# ════════════════════════════════════════════════
# 模型架构：Transformer + Flatten 全局打通
#
# 数据流：
# (N, 28, 28)
#   → Dense 升维           : 行内特征扩展      (N, 28, 64)
#   → Transformer x2       : 行间全局交互      (N, 28, 64)
#   → Flatten              : ✅ 打通所有信息    (N, 28×64=1792)
#   → Dense bottleneck     : 全局语义压缩      (N, 128)
#   → Dense expand         : 全局还原          (N, 1792)
#   → Reshape              : 恢复行结构        (N, 28, 64)
#   → Transformer x2       : 行间全局还原      (N, 28, 64)
#   → Dense 降维           : 行内还原          (N, 28, 28)
# ════════════════════════════════════════════════

input_img = layers.Input(shape=(28, 28))              # (N, 28, 28)

# ── Encoder Part1：升维 + Transformer 行间交互 ────
enc = layers.Dense(64)(input_img)                     # (N, 28, 64)
enc = transformer_encoder_block(enc, num_heads=4, ff_dim=128)  # (N, 28, 64)
enc = transformer_encoder_block(enc, num_heads=4, ff_dim=128)  # (N, 28, 64)

# ── Encoder Part2：Flatten 打通全局 ───────────────
enc_flat = layers.Flatten()(enc)                      # (N, 1792)
enc_flat = layers.Dense(512, activation='relu')(enc_flat)      # (N, 512)
enc_flat = layers.Dense(256, activation='relu')(enc_flat)      # (N, 256)

# ── Bottleneck：全局语义瓶颈 ───────────────────────
bottleneck = layers.Dense(128, activation='relu')(enc_flat)    # (N, 128)

# ── Decoder Part1：全局还原 ────────────────────────
dec_flat = layers.Dense(256, activation='relu')(bottleneck)    # (N, 256)
dec_flat = layers.Dense(512, activation='relu')(dec_flat)      # (N, 512)
dec_flat = layers.Dense(28 * 64, activation='relu')(dec_flat)  # (N, 1792)

# ── Decoder Part2：恢复行结构 + Transformer 还原 ──
dec = layers.Reshape((28, 64))(dec_flat)              # (N, 28, 64)
dec = transformer_decoder_block(dec, enc, num_heads=4, ff_dim=128)  # (N, 28, 64)
dec = transformer_decoder_block(dec, enc, num_heads=4, ff_dim=128)  # (N, 28, 64)

# ── 输出：还原回 (N, 28, 28) ───────────────────────
output = layers.Dense(28, activation='sigmoid')(dec)  # (N, 28, 28)

autoencoder = models.Model(input_img, output)
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

# ── 可视化 ─────────────────────────────────────────
decoded_imgs = autoencoder.predict(x_test_noisy)      # (N, 28, 28)

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