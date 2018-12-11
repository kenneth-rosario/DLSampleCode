# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Activation, Flatten, Dropout, BatchNormalization

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

# Load train images and test images

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# Investigate your data set
print("the train images are 28x28 and there are 60000 samples for training: ", train_images.shape)
print("the test images are 28x28 and there are 10000 samples for testing: ", test_images.shape)

# Pre-process data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)

plt.figure(figsize=(10,10))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


number_of_classes = 10
train_images = train_images/255.0
test_images = test_images/255.0
print(train_images[0].shape)
print(train_images.shape)

for i in range(25):
    plt.subplot(5, 5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

# model = Sequential()
# model.add(Conv2D(filters=64, kernel_size=2, strides=1, padding="same", activation="relu"))
# model.add(Dense(512, activation="relu"))
# model.add(Dense(10, activation='softmax'))
#
#
# # Train
# model.compile(optimizer=tf.train.AdamOptimizer(),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_images,  train_labels, batch_size=64,  epochs=5)
