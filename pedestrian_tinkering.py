import numpy as np
import matplotlib.pyplot as plt
from image_loader import pedestrian_loader, pedestrian_box_border_information
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Activation, Flatten, Dropout, BatchNormalization

images = pedestrian_loader()/255.0
train_labels = pedestrian_box_border_information()

# plt.figure()
# plt.imshow(images[0])
# plt.show()

print(images)
model = Sequential()
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(170, activation='relu'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images,  train_labels, batch_size=64,  epochs=5)
