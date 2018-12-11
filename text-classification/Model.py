import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  Dense, GlobalAveragePooling1D, MaxPool1D, Embedding, Dropout, Conv1D
from preprocess import load_data
from tensorflow.python.keras.layers import MaxPool2D


(train_tweets, train_labels), (test_tweets, test_labels) = load_data()

print(train_tweets.shape)


model = Sequential()

# model.add(Embedding(100000, 16))
# model.add(Conv1D(kernel_size=1, strides=1, padding="same", filters=64))
# model.add(GlobalAveragePooling1D())
model.add(Dense(16, input_shape=(37,)))
model.add(Dense(512, activation=tf.nn.relu))
model.add(Dropout(0.3))
model.add(Dense(512,activation=tf.nn.relu ))
model.add(Dropout(0.3))
model.add(Dense(3, activation=tf.nn.softmax))
model.compile(optimizer=tf.train.AdamOptimizer(),
               loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_tweets,  train_labels, batch_size=50,  epochs=25)
result = model.evaluate(test_tweets, test_labels)
print(result)