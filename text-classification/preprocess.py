import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import numpy as np


def convert(arg, dtype):
    arg = tf.convert_to_tensor(
        value=arg,
        dtype=dtype
    )
    return arg


def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("cleantextlabels7.csv", names=colnames)

    labels = data.label.tolist()
    tweets = data.tweets.tolist()

    train_tweets = tweets[0: 11001]
    train_labels = labels[0:11001]

    test_tweets = tweets[11001:]
    test_label = labels[11001:]

    t = Tokenizer()
    t.fit_on_texts(train_tweets)
    train_tweets = t.texts_to_matrix(tweets, mode='count')
    t.fit_on_texts(test_tweets)
    test_tweets = t.texts_to_matrix(tweets, mode='count')

    return [(convert(train_tweets, tf.float32), convert(train_labels, tf.int64))
        ,(convert(test_tweets, tf.float32), convert(test_label, tf.int64))]

if __name__ == '__main__':
    load_data()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("Succesfully Loaded Data into training dataset and testing")