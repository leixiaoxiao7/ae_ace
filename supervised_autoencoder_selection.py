import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt


import sys
sys.path.append('./funcs')
sys.path.append('./data')
from funcs.tools import *
from funcs.brainCook import *
from funcs.nns import *
from funcs.dataCook import *
from funcs.graphPlot import *

np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# 1️⃣ 数据生成
# =====================================================
# N           = 400
# N_REGIONS   = 125
# N_FEATURES  = 5
TRUE_SIGNAL = 30
TARGET      = 30

outSavePkl='./data/gmwm3D'
data = loadPkl_file(outSavePkl)

gmwm3D = data['gmwm3D']
gmwm3D_extend=data['gmwm3D_extend']
gmwmDF = data['df']
gmwmDim = data['gmwmDimensions']
uniqueHeaders = data['uniqueHeaders']
otherFeaturesH = data['otherFeatureHeaders']
siteList = data['siteList']

# specify X and y
X=gmwm3D
yy=gmwmDF['ace']
y=np.array(yy)

(N,N_REGIONS,N_FEATURES) = X.shape
"""
# 信号脑区
signal = np.random.randn(N, TRUE_SIGNAL, N_FEATURES).astype('float32') * 2.0

# 噪声脑区
noise = np.random.randn(N, N_REGIONS - TRUE_SIGNAL, N_FEATURES).astype('float32') * 0.3

x_data = np.concatenate([signal, noise], axis=1)

# 分类标签（前一半病人=1，后一半=0）
y = np.zeros((N, 1))
y[:N//2] = 1

# 让病人组信号更强（模拟真实差异）
x_data[:N//2, :TRUE_SIGNAL] += 1.5

x_noisy = x_data + np.random.normal(0, 0.3, x_data.shape).astype('float32')
"""
x_data=X 
x_noisy = X

print("X shape:", x_data.shape)
print("y shape:", y.shape)


# =====================================================
# 2️⃣ Transformer Block
# =====================================================
def transformer_block(x, num_heads, ff_dim):
    d_model = x.shape[-1]
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads
    )(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

    ff = layers.Dense(ff_dim, activation='relu')(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


# =====================================================
# 3️⃣ 模型结构
# =====================================================
D_MODEL = 32
NUM_HEADS = 4
FF_DIM = 64
FLAT_DIM = N_REGIONS * D_MODEL

inputs = layers.Input(shape=(N_REGIONS, N_FEATURES))

# ✅ Region Mixing（含125维）
x = layers.Permute((2,1))(inputs)
x = layers.Dense(
    N_REGIONS,
    use_bias=False,
    kernel_regularizer=regularizers.l1(1e-4),
    name="region_mixing"
)(x)
x = layers.Permute((2,1))(x)

# Encoder
x = layers.Dense(D_MODEL)(x)
x = transformer_block(x, NUM_HEADS, FF_DIM)
x = transformer_block(x, NUM_HEADS, FF_DIM)

# Flatten
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)

# ✅ Bottleneck
bottleneck = layers.Dense(128, activation='relu', name="bottleneck")(x)

# =====================================================
# 🔵 分类分支
# =====================================================
cls = layers.Dense(64, activation='relu')(bottleneck)
cls_output = layers.Dense(1, activation='sigmoid', name="cls_output")(cls)

# =====================================================
# 🔵 重建分支
# =====================================================
x = layers.Dense(FLAT_DIM, activation='relu')(bottleneck)
x = layers.Reshape((N_REGIONS, D_MODEL))(x)
x = transformer_block(x, NUM_HEADS, FF_DIM)
recon_output = layers.Dense(N_FEATURES, name="recon_output")(x)

# =====================================================
# 构建模型
# =====================================================
model = models.Model(inputs, [recon_output, cls_output])

model.compile(
    optimizer='adam',
    loss={
        "recon_output": "mse",
        "cls_output": "binary_crossentropy"
    },
    loss_weights={
        "recon_output": 1.0,
        "cls_output": 0.5   # λ 控制分类重要性
    },
    metrics={"cls_output": "accuracy"}
)

model.summary()

# =====================================================
# 4️⃣ 训练
# =====================================================
history = model.fit(
    x_noisy,
    {"recon_output": x_data, "cls_output": y},
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# =====================================================
# 5️⃣ 用权重筛选脑区
# =====================================================
W_region = model.get_layer("region_mixing").get_weights()[0]

region_importance = np.sum(W_region**2, axis=1)
region_importance /= region_importance.sum()

top20 = np.argsort(region_importance)[::-1][:TARGET]
top20_sorted = np.sort(top20)

print("\nTop20:", top20_sorted)

# 验证筛选效果
true_signal = set(range(TRUE_SIGNAL))
selected = set(top20_sorted.tolist())
correct = true_signal & selected

print("正确找到:", len(correct))
print("准确率: {:.1f}%".format(len(correct)/TARGET*100))


# //////////////////////////////////////////////////////////////////////////

# =====================================================
# 🔎 6️⃣ 训练过程全面可视化
# =====================================================

plt.figure(figsize=(14,5))

# 总 loss
plt.subplot(1,3,1)
plt.plot(history.history['loss'], label='Train Total')
plt.plot(history.history['val_loss'], label='Val Total')
plt.title("Total Loss")
plt.legend()

# 重建 loss
plt.subplot(1,3,2)
plt.plot(history.history['recon_output_loss'], label='Train Recon')
plt.plot(history.history['val_recon_output_loss'], label='Val Recon')
plt.title("Reconstruction Loss")
plt.legend()

# 分类 loss
plt.subplot(1,3,3)
plt.plot(history.history['cls_output_loss'], label='Train Cls')
plt.plot(history.history['val_cls_output_loss'], label='Val Cls')
plt.title("Classification Loss")
plt.legend()

plt.tight_layout()
plt.show()



# =====================================================
# 🔎 7️⃣ 分类能力分析（真实脑数据非常重要）
# =====================================================

_, y_pred = model.predict(x_data)

plt.figure(figsize=(6,5))
plt.hist(y_pred[y.flatten()==0], bins=25, alpha=0.6, label='Healthy')
plt.hist(y_pred[y.flatten()==1], bins=25, alpha=0.6, label='Disease')
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Classification Probability Distribution")
plt.legend()
plt.show()

print("Mean prediction (Healthy):", y_pred[y.flatten()==0].mean())
print("Mean prediction (Disease):", y_pred[y.flatten()==1].mean())



# =====================================================
# 🔎 8️⃣ 重建误差可视化（判断模型是否只学分类）
# =====================================================

recon_pred, _ = model.predict(x_data)

recon_error = np.mean((x_data - recon_pred)**2, axis=(1,2))

plt.figure(figsize=(6,5))
plt.hist(recon_error[y.flatten()==0], bins=25, alpha=0.6, label='Healthy')
plt.hist(recon_error[y.flatten()==1], bins=25, alpha=0.6, label='Disease')
plt.title("Reconstruction Error Distribution")
plt.legend()
plt.show()



# =====================================================
# 🔎 9️⃣ Region Mixing 权重矩阵热图
# =====================================================

plt.figure(figsize=(6,5))
plt.imshow(W_region, cmap='bwr')
plt.colorbar()
plt.title("Region Mixing Weight Matrix (125x125)")
plt.xlabel("Input Region")
plt.ylabel("Output Region")
plt.show()



# =====================================================
# 🔎 🔟 脑区重要性完整排序图
# =====================================================

plt.figure(figsize=(14,5))
plt.bar(range(N_REGIONS), region_importance, alpha=0.6)
plt.bar(top20_sorted, region_importance[top20_sorted], color='red')
plt.axvline(x=TRUE_SIGNAL - 0.5, color='green', linestyle='--')
plt.title("Brain Region Importance")
plt.xlabel("Region Index")
plt.ylabel("Normalized Importance")
plt.show()



# =====================================================
# 🔎 11️⃣ Top20 vs 真实信号可视化对比
# =====================================================

signal_mask = np.zeros(N_REGIONS)
signal_mask[:TRUE_SIGNAL] = 1

selection_mask = np.zeros(N_REGIONS)
selection_mask[top20_sorted] = 1

plt.figure(figsize=(14,3))
plt.plot(signal_mask, label='True Signal (0-29)')
plt.plot(selection_mask, label='Selected Top20')
plt.legend()
plt.title("True Signal vs Selected Regions")
plt.show()



# =====================================================
# 🔎 12️⃣ Bottleneck 空间可视化 (PCA)
# =====================================================

from sklearn.decomposition import PCA

# 提取 bottleneck 特征
bottleneck_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("bottleneck").output
)

latent = bottleneck_model.predict(x_data)

pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent)

plt.figure(figsize=(6,5))
plt.scatter(
    latent_2d[y.flatten()==0,0],
    latent_2d[y.flatten()==0,1],
    alpha=0.6,
    label='Healthy'
)
plt.scatter(
    latent_2d[y.flatten()==1,0],
    latent_2d[y.flatten()==1,1],
    alpha=0.6,
    label='Disease'
)
plt.legend()
plt.title("Bottleneck PCA Projection")
plt.show()



# =====================================================
# 🔎 13️⃣ Transformer 输出激活强度（判断是否过平滑）
# =====================================================

encoder_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("bottleneck").input  # Flatten前
)

encoder_output = encoder_model.predict(x_data)

activation_strength = np.mean(np.abs(encoder_output), axis=0)

plt.figure(figsize=(10,4))
plt.plot(activation_strength)
plt.title("Mean Encoder Activation Strength")
plt.xlabel("Feature Index")
plt.show()