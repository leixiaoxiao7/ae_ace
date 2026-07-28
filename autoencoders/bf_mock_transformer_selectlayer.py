import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt

# =====================================================
# 1️⃣ 生成模拟数据
# =====================================================
np.random.seed(42)
tf.random.set_seed(42)

N           = 300
N_REGIONS   = 125
N_FEATURES  = 5
TRUE_SIGNAL = 30
TARGET      = 20

# 前30个脑区有信号
signal = np.random.randn(N, TRUE_SIGNAL, N_FEATURES).astype('float32') * 2.0

# 后95个脑区纯噪声
noise_regions = np.random.randn(
    N, N_REGIONS - TRUE_SIGNAL, N_FEATURES
).astype('float32') * 0.3

x_data = np.concatenate([signal, noise_regions], axis=1)

# 加观测噪声
x_noisy = x_data + np.random.normal(0, 0.3, x_data.shape).astype('float32')

print("x_data shape:", x_data.shape)
print("x_noisy shape:", x_noisy.shape)


# =====================================================
# 2️⃣ Transformer Block
# =====================================================
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


# =====================================================
# 3️⃣ 模型结构
# =====================================================
D_MODEL   = 32
NUM_HEADS = 4
FF_DIM    = 64
FLAT_DIM  = N_REGIONS * D_MODEL  # 125 × 32 = 4000

input_data = layers.Input(shape=(N_REGIONS, N_FEATURES))

# =====================================================
# ✅ 关键层：Region Mixing Layer
# 权重 shape = (125, 125)
# =====================================================
x = layers.Permute((2, 1))(input_data)  # (N, 5, 125)

x = layers.Dense(
    N_REGIONS,
    use_bias=False,
    kernel_regularizer=regularizers.l1(1e-4),  # 可去掉
    name="region_mixing"
)(x)  # (N, 5, 125)

x = layers.Permute((2, 1))(x)  # (N, 125, 5)

# =====================================================
# Encoder
# =====================================================
x = layers.Dense(D_MODEL, name='proj_up')(x)

x = transformer_encoder_block(x, NUM_HEADS, FF_DIM)
x = transformer_encoder_block(x, NUM_HEADS, FF_DIM)

encoder_output = x

# =====================================================
# Flatten 全局打通
# =====================================================
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)

bottleneck = layers.Dense(128, activation='relu', name='bottleneck')(x)

# =====================================================
# Decoder
# =====================================================
x = layers.Dense(256, activation='relu')(bottleneck)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(FLAT_DIM, activation='relu')(x)

x = layers.Reshape((N_REGIONS, D_MODEL))(x)

x = transformer_encoder_block(x, NUM_HEADS, FF_DIM)
x = transformer_encoder_block(x, NUM_HEADS, FF_DIM)

output = layers.Dense(N_FEATURES, activation='linear')(x)

# =====================================================
# 构建模型
# =====================================================
autoencoder = models.Model(input_data, output)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

# =====================================================
# 4️⃣ 训练
# =====================================================
history = autoencoder.fit(
    x_noisy,
    x_data,
    epochs=25,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =====================================================
# 5️⃣ 查看权重（确认包含 125 维）
# =====================================================
print("\n==== 权重形状 ====")
for w in autoencoder.weights:
    print(w.name, w.shape)

# ✅ 获取 Region Mixing 权重
W_region = autoencoder.get_layer("region_mixing").get_weights()[0]
print("\nRegion Mixing Weight shape:", W_region.shape)
# 应为 (125, 125)

# =====================================================
# 6️⃣ 根据权重筛选 Top‑20 脑区
# =====================================================
region_importance = np.sum(W_region ** 2, axis=1)  # L2 norm
region_importance /= region_importance.sum()

top20_indices = np.argsort(region_importance)[::-1][:TARGET]
top20_sorted  = np.sort(top20_indices)

print("\nTop-20 脑区索引:", top20_sorted)

# =====================================================
# 7️⃣ 验证筛选准确率
# =====================================================
true_signal_set = set(range(TRUE_SIGNAL))
selected_set = set(top20_sorted.tolist())

correct = true_signal_set & selected_set

print("\n真实信号脑区数量:", TRUE_SIGNAL)
print("正确找到数量:", len(correct))
print("正确找到索引:", sorted(correct))
print("筛选准确率: {:.1f}%".format(len(correct)/TARGET*100))

# =====================================================
# 8️⃣ 最终降维
# =====================================================
x_selected = x_data[:, top20_sorted, :]
print("\n最终降维 shape:", x_selected.shape)

# =====================================================
# 9️⃣ 可视化脑区重要性
# =====================================================
plt.figure(figsize=(14,5))
plt.bar(range(N_REGIONS), region_importance, alpha=0.5)
plt.bar(top20_sorted, region_importance[top20_sorted], color='red')
plt.axvline(x=TRUE_SIGNAL - 0.5, color='green', linestyle='--')
plt.title("Brain Region Importance")
plt.xlabel("Region Index")
plt.ylabel("Importance")
plt.show()