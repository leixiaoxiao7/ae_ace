import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

data = np.array([
    [80,120, 27, 200],
    [81,121, 28, 201],
    [70,110, 20, 190],
    [71,111, 21, 191],
    [86,125, 21, 240]])
mean = np.mean(data, axis=0)
std_dev = np.std(data, axis=0, ddof=1)
data_scaled = (data - mean) / std_dev

random.seed(600)

input_layer = layers.Input(shape=(4,)) # Input layer
encoded = layers.Dense(2, activation='linear', use_bias=False)(input_layer)
decoded = layers.Dense(4, activation='linear', use_bias=False)(encoded)
autoencoder = models.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

history = autoencoder.fit(data_scaled, data_scaled, epochs=2000, batch_size=5)
predicted_data = autoencoder.predict(data_scaled)

np.sum((data_scaled - predicted_data) ** 2) / 20

plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Epochs')  # Added xlabel
plt.ylabel('MSE')    # Added ylabel
plt.show()

encoder = models.Model(input_layer, encoded)
encoded_data = encoder.predict(data_scaled)

plt.figure()
plt.scatter(encoded_data[:, 0], encoded_data[:, 1])
# Add identity numbers to the scatter plot
for i in range(len(encoded_data)):
    plt.text(encoded_data[i, 0], encoded_data[i, 1], str(i + 1))

plt.xlabel('Encoded Dimension 1')
plt.ylabel('Encoded Dimension 2')
plt.show()

weights = autoencoder.get_weights()
hidden = np.dot(np.transpose(weights[0]), np.transpose(data_scaled))
np.dot(np.transpose(weights[1]), hidden)