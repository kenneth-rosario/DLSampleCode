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

def pad_str_280(arg):
    difference = 280 - len(arg)
    return "{:<"+str(difference)+"}".format(arg)


def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("cleantextlabels7.csv", names=colnames)

    labels = data.label.tolist()
    tweets = data.tweets.tolist()

    train_tweets = tweets[0: 11001]
    train_labels = labels[0:11001]

    test_tweets = tweets[11001:]
    test_label = labels[11001:]

    train_tweets = [text_to_word_sequence(x) for x in train_tweets]
    test_tweets = [text_to_word_sequence(x) for x in test_tweets]
    t = Tokenizer(tweets)
    t.fit_on_texts(tweets)
    print(t.word_counts)
    print(t.document_count)
    print(t.word_index)
    print(t.word_docs)
    return [(convert(train_tweets, object), convert(train_labels, tf.int64))
        ,(convert(test_tweets, object), convert(test_label, tf.int64))]

if __name__ == '__main__':
    load_data()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("Succesfully Loaded Data into training dataset and testing")