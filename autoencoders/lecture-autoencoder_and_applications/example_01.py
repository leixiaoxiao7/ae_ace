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

# Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3))
# Conv2D: learns local image patterns with sliding convolution filters.
# 32: number of filters; output has 32 feature maps.
# (3, 3): each filter looks at a 3-by-3 pixel neighborhood.
# activation='relu': replaces negative values with 0; keeps useful positive features.
# padding='same': pads image borders so height and width stay SIZE x SIZE.
# input_shape=(SIZE, SIZE, 3): one RGB image, with 3 color channels.
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 3)))  # First local feature extractor.

# MaxPooling2D((2, 2), padding='same')
# MaxPooling2D: downsamples each feature map by keeping the strongest value in each window.
# (2, 2): pooling window size; roughly halves height and width.
# padding='same': pads edges when needed so the downsampled shape is rounded safely.
model.add(MaxPooling2D((2, 2), padding='same'))  # Reduce spatial resolution by 2x.

# Conv2D(8, (3, 3), activation='relu', padding='same')
# 8 filters: compresses from 32 feature maps to 8 feature maps.
# This keeps fewer learned features, forcing the autoencoder to store compact information.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Narrower feature maps.

# Second pooling layer: further compresses spatial size.
# After repeated pooling, the network stores a smaller representation of the image.
model.add(MaxPooling2D((2, 2), padding='same'))  # Further downsample.

# Another 8-filter convolution: refines features at the compressed resolution.
# Same parameters as above: 3x3 filters, ReLU nonlinearity, same spatial size.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Bottleneck refinement.

# Bottleneck compression.
# Third pooling layer: creates the smallest encoded representation.
# This bottleneck is the compressed code the decoder must use to reconstruct the image.
model.add(MaxPooling2D((2, 2), padding='same'))

# Decoder: expand the compressed representation back to image space.
# Conv2D(8, ...): learns how to transform bottleneck features before upsampling.
# It keeps 8 feature maps and preserves the current height/width with padding='same'.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Prepare for upsampling.

# UpSampling2D((2, 2)): repeats rows and columns to double height and width.
# (2, 2): scale factor for vertical and horizontal dimensions.
model.add(UpSampling2D((2, 2)))  # Restore spatial size.

# Conv2D(8, ...): cleans/refines the enlarged feature maps after upsampling.
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))  # Refine reconstructed features.

# Second upsampling layer: doubles height and width again.
model.add(UpSampling2D((2, 2)))  # Restore spatial size again.

# Conv2D(32, ...): expands back to richer feature maps before final RGB output.
# 32 filters gives the decoder more channels to reconstruct image details.
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # Recover richer detail.

# Third upsampling layer: returns the feature maps to the original image resolution.
model.add(UpSampling2D((2, 2)))  # Return to original resolution.

# Conv2D(3, ...): final image layer.
# 3 filters produce 3 output channels: red, green, blue.
# activation='relu' keeps output non-negative; padding='same' keeps image size unchanged.
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))  # Final RGB reconstruction.

# Mean squared error measures reconstruction quality; accuracy is logged for reference.
# optimizer='adam': adaptive gradient optimizer used to update network weights.
# loss='mean_squared_error': penalizes pixel-wise reconstruction difference.
# metrics=['accuracy']: logs an extra metric during training; the loss is the main objective here.
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.summary()


# model-fitting
# Train the autoencoder to reproduce the input image.
model.fit(img_array, img_array, epochs=500, shuffle=True) # 500

# Predict the reconstructed version of the input image.
pred = model.predict(img_array)

# Display the reconstructed image from the first batch element.
# imshow(pred[0].reshape(SIZE,SIZE,3))
plt.imshow(pred[0].reshape(SIZE, SIZE, 3))
plt.axis("off")
plt.show()