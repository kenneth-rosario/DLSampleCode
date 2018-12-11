import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot

import numpy as np


def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("cleantextlabels7.csv", names=colnames)

    labels = data.label.tolist()
    tweets = data.tweets.tolist()

    train_tweets = tweets[0: 11001]
    train_labels = labels[0: 11001]

    test_tweets = tweets[11001:]
    test_label = labels[11001:]


    t = Tokenizer(num_words=100000)
    t.fit_on_texts(train_tweets)
    train_tweets = t.texts_to_sequences(train_tweets)
    t.fit_on_texts(test_tweets)
    test_tweets = t.texts_to_sequences(test_tweets)


    # train_tweets = t.texts_to_matrix(tweets, mode='')
    # t.fit_on_texts(test_tweets)
    # test_tweets = t.texts_to_matrix(tweets, mode='count')



    train_tweets =  keras.preprocessing.sequence.pad_sequences(train_tweets, value= 0, padding='post', maxlen=280)
    test_tweets = keras.preprocessing.sequence.pad_sequences(test_tweets, value=0, padding='post', maxlen=280)
    print(train_tweets)

    return [(train_tweets, train_labels), (test_tweets, test_label)]

if __name__ == '__main__':
    load_data()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("Succesfully Loaded Data into training dataset and testing")