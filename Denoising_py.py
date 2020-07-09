# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 02:10:30 2020

@author: RUSHI
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, BatchNormalization, Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from PIL import Image

# Loading CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizing inputs
x_train = x_train/255
x_test = x_test/255

# Noise (normal distribution) addition function with mean=0.0 and std=0.1
def noise_addition(image):
    noise = np.random.normal(loc=0.0, scale=0.1, size=image.shape)
    image = image + noise
    image = np.clip(image, 0., 1.)
    return image    

noisy_x_train = noise_addition(x_train)
noisy_x_test = noise_addition(x_test)

# Displaying original and noisy image from the dataset
index = 0
plt.subplot(1, 2, 1)
plt.imshow(x_train[index])
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(noisy_x_train[index])
plt.title('Noisy')
plt.show()

# Defining model
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(BatchNormalization())

model.add(Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same', activation='sigmoid'))

model.compile(optimizer='adam', metrics=['accuracy'], loss='mse')
model.summary()

# Training

checkpoint = ModelCheckpoint('best_model.h5', verbose=1, save_best_only=True, save_weights_only=True)

model.fit(noisy_x_train, x_train, validation_data=(noisy_x_test, x_test), epochs=40, batch_size=128, callbacks=[checkpoint])