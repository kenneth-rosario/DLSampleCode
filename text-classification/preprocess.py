import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot


def get_max_words(arr):
    ds = set()
    for i in arr:
        for j in i:
            ds.add(j)
    return len(ds)

def get_max_len(arr):
    max = 0
    for i in arr:
        if len(i) > max:
            max = len(arr)
    return max

def load_data():
    colnames = ["tweets", "label"]
    data = pd.read_csv("cleantextlabels7.csv", names=colnames)

    labels = data.label.tolist()
    tweets = data.tweets.tolist()

    train_tweets = tweets[0: 11001]
    train_labels = labels[0: 11001]

    test_tweets = tweets[11001:]
    test_label = labels[11001:]


    t = Tokenizer(num_words=get_max_words(train_tweets))
    t.fit_on_texts(train_tweets)
    train_tweets = t.texts_to_sequences(train_tweets)
    t.fit_on_texts(test_tweets)
    test_tweets = t.texts_to_sequences(test_tweets)


    train_tweets =  keras.preprocessing.sequence.pad_sequences(train_tweets, value= 0, padding='post', maxlen=get_max_words(train_tweets))
    test_tweets = keras.preprocessing.sequence.pad_sequences(test_tweets, value=0, padding='post', maxlen=get_max_words(test_tweets))



    return [(train_tweets, train_labels), (test_tweets, test_label)]

if __name__ == '__main__':
    load_data()
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")
    print("Succesfully Loaded Data into training dataset and testing")