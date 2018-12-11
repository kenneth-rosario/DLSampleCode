import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, GlobalAveragePooling1D, MaxPool1D, Embedding
from preprocess import load_data
from tensorflow.python.keras.layers import MaxPool2D

train_tweets = []
train_labels = []
test_tweets = []
test_labels = []


[(train_tweets, train_labels), (test_tweets, test_labels)] = load_data()

print(train_tweets.shape)


model = Sequential()

model.add(Embedding(100000, 16))
model.add(GlobalAveragePooling1D())
model.add(Dense(512, activation=tf.nn.relu))

model.add(Dense(16, activation=tf.nn.relu))
model.add(Dense(3, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(),
               loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_tweets,  train_labels, batch_size=512,  epochs=40)