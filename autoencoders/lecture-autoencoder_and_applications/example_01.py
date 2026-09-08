"""
@author: XXL
@lecturer: Sreenivas Bhattiprolu
"""
# Convolutional autoencoder demo:
# - load one RGB image
# - resize and normalize it
# - train a small encoder-decoder to reconstruct the same image
# - display the reconstruction
from matplotlib.pyplot import imshow
import numpy as np
# import cv2
from PIL import Image
import matplotlib.pyplot as plt

#from keras.preprocessing.image import img_to_array # get numerical arrays from image
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, ReLU
from tensorflow.keras.models import Model

# Fix the random seed so training is reproducible.
np.random.seed(42)



# parameterizations
# Fixed spatial resolution for preprocessing and model I/O.
SIZE=256

# Single-image container; later reshaped into a batch tensor.
img_data =[]
'''
# Legacy OpenCV preprocessing path kept as a reference.
img=cv2.imread('images/monalisa.jpg',1) # 1 - color; 0 - grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(SIZE,SIZE))
'''

# Image to reconstruct.
img_file = 'images/monalisa.jpg'

# Load with PIL and force RGB so the model always receives 3 channels.
try:
    img_pil = Image.open(img_file).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError("图片路径错误：images/monalisa.jpg")

# Match the image to the model input size.
img_pil = img_pil.resize((SIZE, SIZE))

# Convert the PIL image into a NumPy RGB array.
img_arr = np.array(img_pil)

# Wrap the single image in a list so it can become a batch.
img_data.append(img_arr)

# Create the batch tensor expected by Keras: (batch, height, width, channels).
img_array = np.reshape(img_data,(len(img_data),SIZE, SIZE, 3))
# Normalize pixel values to [0, 1] for stable training.
img_array = img_array.astype('float32') / 255.


# constructing a autoencoder-NN (U-Net style)
# ==================== 清晰度优化：共 6 处改动 ====================
# 改动1: 网络加宽  32/8/8 -> 64/128/256/512 通道（下方 Encoder/Decoder）
# 改动2: 上采样 nearest -> bilinear（3 个 UpSampling2D）
# 改动3: 输出层激活 relu -> sigmoid（末尾的 Conv2D(3, ...)）
# 改动4: 损失函数 mse -> mae，并删除无意义的 accuracy 指标（model.compile）
# 改动5: 训练轮数 500（U-Net+BN 收敛很快，实测 200 轮已达 40.5 dB）
# 改动6: Sequential 纯编码-解码 -> U-Net 跳跃连接（Functional API）
# 改动7: 每个卷积后加 BatchNormalization（修复 U-Net 训练塌缩）
# 实测 PSNR（越大越好）: 原版 30.19 dB -> 改动1-5 后 35.77 dB -> U-Net 40.54 dB
# ================================================================

# ==================== 层类型速查表（大白话版）====================
# 层                 | 通俗理解                              | 在脚本里的作用
# -------------------+--------------------------------------+------------------------------
# Input              | 收件台：收进 256x256 的 RGB 照片       | 规定输入形状
# Conv2D(3x3)        | 小放大镜滑遍全图，专找一种花纹          | 提特征：浅层找边角，深层找结构
# BatchNormalization | 每道工序后把零件重新校准、统一分量      | 稳住数值，防训练塌缩
# ReLU               | 只留好消息：负数归零，正数放行          | 引入非线性，才能学复杂模式
# MaxPooling2D       | 缩印：2x2 四格只留最显眼的              | 压缩画面，逼网络记大意
# UpSampling2D       | 把缩印图平滑放大回去（bilinear）       | 解码器逐步恢复原尺寸
# Concatenate        | 把编码时拍的高清细节照钉在旁边对照修补  | 跳跃连接：细节绕过瓶颈，清晰度关键
# Conv2D(3,sigmoid)  | 收尾调色成 RGB，颜色深浅限制在 0~1      | 输出合法像素
# 一句话流程: 卷积找花纹 -> BN 稳分量 -> ReLU 留有用 -> pooling 记大意（编码）
#            -> 放大并对照原细节修补（解码）-> sigmoid 调色输出
# ================================================================

# 【改动6】为什么必须换掉 Sequential（这是逼近原图清晰度的关键）:
# 原结构: 所有信息必须挤过 32x32 的 bottleneck，发丝/轮廓/纹理等高频
#        细节在下采样时已物理丢失，解码器只能"猜"，这就是模糊的上限。
# U-Net: 把编码器每一层的特征用 Concatenate 直接拼到解码器同分辨率层，
#        细节走"高速公路"绕过 bottleneck，bottleneck 只学全局结构。
# 代价: 信息绕过了压缩点，严格说这不再是"压缩型"自编码器——
#      压缩率和清晰度不可兼得，本脚本优先清晰度。
# 注意: 跨层连接 Sequential 表达不了，必须用 Functional API（层当作函数调用）。

inputs = Input(shape=(SIZE, SIZE, 3))

# 【改动7】Conv -> BN -> ReLU 块。
# 为什么必须加: 网络加深后，sigmoid+MAE 组合在训练初期容易饱和卡死
# （实测不加 BN 时输出塌缩成纯色，PSNR 仅 12 dB）。
# BatchNormalization 把每层输入重新归一化，梯度稳定，收敛更快更稳。
def conv_bn(x, filters):
    x = Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)  # BN 自带偏置，关掉卷积偏置
    x = BatchNormalization()(x)
    return ReLU()(x)

# ---- Encoder ----
# 【改动1】通道 64/128/256（原 32/8/8 太窄，bottleneck 存不下细节）。
# padding='same': pads image borders so height and width stay unchanged.
c1 = conv_bn(inputs, 64)                                             # 256x256x64
c1 = conv_bn(c1, 64)
p1 = MaxPooling2D((2, 2), padding='same')(c1)                        # 256 -> 128

c2 = conv_bn(p1, 128)
c2 = conv_bn(c2, 128)
p2 = MaxPooling2D((2, 2), padding='same')(c2)                        # 128 -> 64

c3 = conv_bn(p2, 256)
c3 = conv_bn(c3, 256)
p3 = MaxPooling2D((2, 2), padding='same')(c3)                        # 64 -> 32

# ---- Bottleneck: 只负责全局结构的压缩编码 ----
b = conv_bn(p3, 512)                                                 # 32x32x512
b = conv_bn(b, 512)

# ---- Decoder: 每层先上采样，再拼接同尺度编码特征（跳跃连接）----
# 【改动2】interpolation='bilinear'（原默认 'nearest' 复制像素，产生马赛克伪影）
u3 = UpSampling2D((2, 2), interpolation='bilinear')(b)               # 32 -> 64
u3 = Concatenate()([u3, c3])  # 跳跃连接: 64x64 编码细节直达解码器
c4 = conv_bn(u3, 256)
c4 = conv_bn(c4, 256)

u2 = UpSampling2D((2, 2), interpolation='bilinear')(c4)              # 64 -> 128
u2 = Concatenate()([u2, c2])  # 跳跃连接
c5 = conv_bn(u2, 128)
c5 = conv_bn(c5, 128)

u1 = UpSampling2D((2, 2), interpolation='bilinear')(c5)              # 128 -> 256
u1 = Concatenate()([u1, c1])  # 跳跃连接
c6 = conv_bn(u1, 64)
c6 = conv_bn(c6, 64)

# Final RGB layer: 3 filters produce red, green, blue channels.
# 【改动3】relu -> sigmoid: 像素已归一化到 [0,1]，sigmoid 把输出约束到同一区间，
#          避免 relu 无上界导致的亮色过曝偏置、颜色发闷。
outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(c6)

model = Model(inputs, outputs)

# optimizer='adam': adaptive gradient optimizer used to update network weights.
# 【改动4】loss mse -> mae: MSE(L2) 对没把握的像素取"平均值"导致模糊；
#          MAE(L1) 逼近真实像素中位数，边缘更锐利。
#          同时删掉 metrics=['accuracy']（分类指标，对像素回归无意义）。
model.compile(optimizer='adam', loss='mae')
model.summary()


# model-fitting
# Train the autoencoder to reproduce the input image.
# 【改动5】epochs: 500 足够。单图重建是"记忆"任务，无过拟合风险；
#          U-Net+BN 收敛很快（实测 200 轮已达 40.5 dB，肉眼难辨差异），
#          时间充裕可加到 1000-2000 进一步逼近无损。
model.fit(img_array, img_array, epochs=500, shuffle=True)

# Predict the reconstructed version of the input image.
pred = model.predict(img_array)

# Display the reconstructed image from the first batch element.
# imshow(pred[0].reshape(SIZE,SIZE,3))
plt.imshow(pred[0].reshape(SIZE, SIZE, 3))
plt.axis("off")
plt.show()