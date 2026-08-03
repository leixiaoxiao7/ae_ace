"""
@author: XXL
@lecturer: Sreenivas Bhattiprolu
"""
from matplotlib.pyplot import imshow
import numpy as np
# import cv2
from PIL import Image

#from keras.preprocessing.image import img_to_array # get numerical arrays from image
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential

np.random.seed(42)



# parameterizations
SIZE=256

img_data =[]
'''
img=cv2.imread('images/monalisa.jpg',1) # 1 - color; 0 - grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(SIZE,SIZE))
'''

img_file = 'images/monalisa.jpg'

try:
    img_pil = Image.open(img_file).convert("RGB")
except FileNotFoundError:
    raise FileNotFoundError("图片路径错误：images/monalisa.jpg")

# img = cv2.resize(img,(SIZE,SIZE))
img_pil = img_pil.resize((SIZE, SIZE))

# 转numpy数组（等价 cv2读取后的矩阵）
img_arr = np.array(img_pil)

# 不再需要 cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_data.append(img_arr)

img_array = np.reshape(img_data,(len(img_data),SIZE, SIZE, 3))
img_array = img_array.astype('float32') / 255.


# constructing a autoencoder-NN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))

model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()


# model-fitting
model.fit(img_array, img_array, epochs=5, shuffle=True) # 500

pred = model.predict(img_array)

imshow(pred[0].reshape(SIZE,SIZE,3))