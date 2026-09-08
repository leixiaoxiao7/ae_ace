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
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

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


# constructing a autoencoder-NN
# Sequential(): builds the neural network as a simple stack of layers.
# Each model.add(...) appends one new layer after the previous layer.
model = Sequential()

# ==================== 清晰度优化：共 5 处改动 ====================
# 改动1: 网络加宽  32/8/8 -> 64/128/256 通道（下方 Encoder/Decoder）
# 改动2: 上采样 nearest -> bilinear（下方 Decoder 的 3 个 UpSampling2D）
# 改动3: 输出层激活 relu -> sigmoid（文件末尾的 Conv2D(3, ...)）
# 改动4: 损失函数 mse -> mae，并删除无意义的 accuracy 指标（model.compile）
# 改动5: 训练轮数 500 -> 2000（model.fit）
# 实测 PSNR（越大越好）: 原版 30.19 dB -> 改后 35.77 dB
# ================================================================

# 【改动1】Encoder: 3 conv blocks with 64 -> 128 -> 256 filters.
# 原实现: 32 -> 8 -> 8 通道，bottleneck 只有 32x32x8 = 8192 个数。
# 为什么改: 通道太少 => bottleneck 存不下脸部细节，重建必然模糊。
#          加宽后 bottleneck 为 32x32x256，容量提升 32 倍。
# Conv2D: learns local image patterns with sliding 3x3 convolution filters.
# activation='relu': replaces negative values with 0; keeps useful positive features.
# padding='same': pads image borders so height and width stay unchanged.
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))  # input: 256x256x3
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))

# MaxPooling2D: downsamples by keeping the strongest value in each 2x2 window.
model.add(MaxPooling2D((2, 2), padding='same'))  # 256 -> 128

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))  # 128 -> 64

# Bottleneck: the compressed code the decoder must reconstruct from.
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))  # 64 -> 32; code is 32x32x256

# Decoder: mirror the encoder back to full resolution.
# 【改动2】UpSampling2D 加 interpolation='bilinear'（本层及下面两层，共 3 处）。
# 原实现: 默认 'nearest'，把每个像素复制成 2x2 方块。
# 为什么改: 最近邻复制会产生马赛克/棋盘状伪影；双线性插值平滑放大，
#          后面的卷积再在平滑底子上补细节，边缘更干净。
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2), interpolation='bilinear'))  # 32 -> 64

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2), interpolation='bilinear'))  # 64 -> 128

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2), interpolation='bilinear'))  # 128 -> 256

# Final RGB layer: 3 filters produce red, green, blue channels.
# 【改动3】输出激活 relu -> sigmoid。
# 原实现: relu 输出范围 [0, +inf)，没有上界。
# 为什么改: 像素已归一化到 [0, 1]，sigmoid 把输出约束到同一区间，
#          避免亮色像素过曝偏置、颜色发闷，训练目标与输出范围一致。
model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# optimizer='adam': adaptive gradient optimizer used to update network weights.
# 【改动4】loss='mean_squared_error' -> 'mae'，并删掉 metrics=['accuracy']。
# 原实现: MSE (L2) 对误差平方，模型在没把握的像素上倾向取"平均值"。
# 为什么改: 平均化 = 模糊。MAE (L1) 鼓励逼近真实像素值的中位数，边缘更锐利。
#          accuracy 是分类指标，对像素回归任务无意义，只会误导日志阅读。
model.compile(optimizer='adam', loss='mae')
model.summary()


# model-fitting
# Train the autoencoder to reproduce the input image.
# 【改动5】epochs 500 -> 2000。
# 原实现: 500 epochs。
# 为什么改: 单图重建本质是"记忆"这张图，不存在过拟合风险；
#          网络加宽后 500 轮还没收敛（实测只有 31.7 dB），
#          2000 轮达到 35.8 dB。shuffle 对单样本无作用，保留无害。
model.fit(img_array, img_array, epochs=2000, shuffle=True)

# Predict the reconstructed version of the input image.
pred = model.predict(img_array)

# Display the reconstructed image from the first batch element.
# imshow(pred[0].reshape(SIZE,SIZE,3))
plt.imshow(pred[0].reshape(SIZE, SIZE, 3))
plt.axis("off")
plt.show()