import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# ════════════════════════════════════════════════
# 生成有规律的模拟数据
# 真正有信号的脑区：前30个
# 其余95个：纯噪声
# ════════════════════════════════════════════════
np.random.seed(42)
N           = 300
N_REGIONS   = 125
N_FEATURES  = 5
TRUE_SIGNAL = 30    # 真正有信号的脑区数量（用于验证）
TARGET      = 20    # 目标降维脑区数

# 生成基础信号：前30个脑区有结构性信号
signal = np.random.randn(N, TRUE_SIGNAL, N_FEATURES).astype('float32') * 2.0

# 生成纯噪声：后95个脑区
noise_regions = np.random.randn(N, N_REGIONS - TRUE_SIGNAL, N_FEATURES).astype('float32') * 0.3

# 拼合完整数据集
x_data = np.concatenate([signal, noise_regions], axis=1)   # (300, 125, 5)

# 加观测噪声
x_noisy = x_data + np.random.normal(0, 0.3, x_data.shape).astype('float32')
x_noisy = x_noisy.astype('float32')

print(f"x_data  shape : {x_data.shape}")   # (300, 125, 5)
print(f"x_noisy shape : {x_noisy.shape}")  # (300, 125, 5)
print(f"真正有信号的脑区索引: 0 ~ {TRUE_SIGNAL - 1}")


# ════════════════════════════════════════════════
# Transformer Block
# ════════════════════════════════════════════════
def transformer_encoder_block(x, num_heads, ff_dim, dropout=0.1):
    d_model = x.shape[-1]
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def transformer_decoder_block(x, encoder_output, num_heads, ff_dim, dropout=0.1):
    d_model = x.shape[-1]

    # Self-Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Cross-Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, encoder_output)
    attn = layers.Dropout(dropout)(attn)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed Forward
    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


# ════════════════════════════════════════════════
# 模型架构：Transformer + Flatten 全局打通
#
# (N, 125, 5)
#   → Dense 升维                (N, 125, 32)
#   → Transformer x2            (N, 125, 32)   脑区间全局交互
#   → Flatten                   (N, 125×32=4000)  ✅ 打通全局
#   → Dense bottleneck          (N, 256) → (N, 128)
#   → Dense expand              (N, 256) → (N, 4000)
#   → Reshape                   (N, 125, 32)   恢复脑区结构
#   → Transformer decoder x2    (N, 125, 32)
#   → Dense 降维                (N, 125, 5)    还原特征
# ════════════════════════════════════════════════
D_MODEL   = 32
NUM_HEADS = 4
FF_DIM    = 64
FLAT_DIM  = N_REGIONS * D_MODEL    # 125 × 32 = 4000

input_data = layers.Input(shape=(N_REGIONS, N_FEATURES))   # (N, 125, 5)

# ── Encoder Part1：升维 + Transformer ─────────────────
enc = layers.Dense(D_MODEL, name='proj_up')(input_data)              # (N, 125, 32)
enc = transformer_encoder_block(enc, num_heads=NUM_HEADS, ff_dim=FF_DIM)  # (N, 125, 32)
enc = transformer_encoder_block(enc, num_heads=NUM_HEADS, ff_dim=FF_DIM)  # (N, 125, 32)

# ── Encoder Part2：Flatten 打通全局 ───────────────────
enc_flat = layers.Flatten(name='flatten')(enc)                        # (N, 4000)
enc_flat = layers.Dense(512, activation='relu')(enc_flat)             # (N, 512)
enc_flat = layers.Dense(256, activation='relu')(enc_flat)             # (N, 256)

# ── Bottleneck ────────────────────────────────────────
# 125 这个数字在 Flatten 权重里：
# Flatten层权重 shape = (4000, 512) = (125×32, 512)  ← 125 在这里
bottleneck = layers.Dense(128, activation='relu', name='bottleneck')(enc_flat)  # (N, 128)

# ── Decoder Part1：全局还原 ───────────────────────────
dec_flat = layers.Dense(256, activation='relu')(bottleneck)           # (N, 256)
dec_flat = layers.Dense(512, activation='relu')(dec_flat)             # (N, 512)
dec_flat = layers.Dense(FLAT_DIM, activation='relu')(dec_flat)        # (N, 4000)

# ── Decoder Part2：恢复脑区结构 + Transformer ─────────
dec = layers.Reshape((N_REGIONS, D_MODEL), name='reshape')(dec_flat) # (N, 125, 32)
dec = transformer_decoder_block(dec, enc, num_heads=NUM_HEADS, ff_dim=FF_DIM)  # (N, 125, 32)
dec = transformer_decoder_block(dec, enc, num_heads=NUM_HEADS, ff_dim=FF_DIM)  # (N, 125, 32)

# ── 输出：还原回 (N, 125, 5) ──────────────────────────
output = layers.Dense(N_FEATURES, activation='linear', name='recon_output')(dec)  # (N, 125, 5)

autoencoder = models.Model(input_data, output)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# ════════════════════════════════════════════════
# 训练
# ════════════════════════════════════════════════
history = autoencoder.fit(
    x_noisy, x_data,
    epochs=27,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

plt.figure(figsize=(8, 3))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Reconstruction MSE Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()


# ════════════════════════════════════════════════
# 通过权重筛选 Top-20 脑区
#
# 关键权重：proj_up 层 (第一个Dense)
# 权重 shape: (5, 32)
# 但是我们需要 125 这个维度
#
# 方法：用训练好的 encoder 输出每个脑区的表示
# 再计算每个脑区表示的 SSQ（能量）
# SSQ 越高 → 这个脑区携带的信息越多 → 越重要
# ════════════════════════════════════════════════

# Step1：建立只到 Transformer 输出的 encoder 子模型
encoder_model = models.Model(
    inputs=input_data,
    outputs=enc,          # Transformer 输出，shape: (N, 125, 32)
    name='encoder_only'
)

# Step2：获取每个脑区的表示
region_repr = encoder_model.predict(x_data)          # (N, 125, 32)

# Step3：计算每个脑区的 SSQ（跨样本、跨特征维度）
# 每个脑区的重要性 = 它在所有样本中表示向量的平均能量
region_ssq = np.sum(region_repr ** 2, axis=(0, 2))   # (125,)
# axis=0: 跨样本求和, axis=2: 跨D_MODEL维度求和

# Step4：归一化重要性分数
region_importance = region_ssq / region_ssq.sum()    # (125,)

# Step5：选出 Top-20
top20_indices        = np.argsort(region_importance)[::-1][:TARGET]
top20_indices_sorted = np.sort(top20_indices)

print(f"\n{'='*50}")
print(f"Top-{TARGET} 脑区索引（按重要性排序）: {top20_indices}")
print(f"Top-{TARGET} 脑区索引（排序后）      : {top20_indices_sorted}")
print(f"\n重要性分数 Top-{TARGET}: {region_importance[top20_indices].round(4)}")

# Step6：验证：检查筛选出的脑区有多少来自真实信号区（0~29）
true_signal_set    = set(range(TRUE_SIGNAL))
selected_set       = set(top20_indices_sorted.tolist())
correctly_found    = true_signal_set & selected_set
print(f"\n{'='*50}")
print(f"真实信号脑区 (0~{TRUE_SIGNAL-1}): {TRUE_SIGNAL} 个")
print(f"筛选目标数量               : {TARGET} 个")
print(f"正确找到的信号脑区          : {len(correctly_found)} 个")
print(f"正确找到的脑区索引          : {sorted(correctly_found)}")
print(f"筛选准确率                  : {len(correctly_found)/TARGET*100:.1f}%")

# Step7：最终降维
x_selected = x_data[:, top20_indices_sorted, :]     # (N, 20, 5)
print(f"\n最终降维结果 shape: {x_selected.shape}")   # (300, 20, 5)


# ════════════════════════════════════════════════
# 可视化：脑区重要性分布
# ════════════════════════════════════════════════
plt.figure(figsize=(14, 5))

plt.bar(range(N_REGIONS), region_importance,
        color='steelblue', alpha=0.5, label='All Regions')

plt.bar(top20_indices, region_importance[top20_indices],
        color='crimson', alpha=0.9, label=f'Top-{TARGET} Selected')

plt.axvline(x=TRUE_SIGNAL - 0.5, color='green',
            linestyle='--', linewidth=2, label=f'True Signal Boundary (0~{TRUE_SIGNAL-1})')

plt.xlabel('Brain Region Index')
plt.ylabel('SSQ Importance Score (normalized)')
plt.title(f'Brain Region Importance\n'
          f'Correctly identified: {len(correctly_found)}/{TARGET} signal regions')
plt.legend()
plt.tight_layout()
plt.show()