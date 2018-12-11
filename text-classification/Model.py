import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from preprocess import  load_data


train_tweets = []
train_labels = []
test_tweets = []
test_labels = []


[(train_tweets, train_labels), (test_tweets, test_labels)] = load_data()


# model = Sequential()
#
# model.add(Dense(5, input_shape=(11001, )))
# model.add(Dense(3, activation='softmax'))
#
#
# model.compile(optimizer=tf.train.AdamOptimizer(),
#                loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_tweets,  train_labels, batch_size=64,  epochs=5)