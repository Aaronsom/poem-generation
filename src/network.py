from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon
import numpy as np
import data_prepocessing
import embedding as embedding_loader
from collections import OrderedDict

def create_model(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=50, input_length=n, weights=[embedding]),
        Flatten(),
        Dropout(0.5),
        Dense(n*500),
        Activation("relu"),
        Dropout(0.5),
        Dense(vocab_len),
        Activation("softmax")
    ])
    model.compile(optimizer=optimizer.Adam(lr=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    #SGD(lr=0.001, momentum=0.5, nesterov=True)
    return model

import os
if __name__ == "__main__":
    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    set_floatx("float16")
    set_epsilon(1e-04)
    n = 7
    epochs = 10
    batch_size = 1024
    max_limit = 40000

    poems = data_prepocessing.tokenize_poems("DATA/train_poems.txt")
    words = OrderedDict([(token, 0) for poem in poems for token in poem]).keys()

    #Save embedding for generator
    embedding, dictionary = embedding_loader.get_embedding(words, limit=max_limit, save=True)
    model = create_model(n, embedding, len(dictionary))

    # tuplelize train and dev set, convert to indices and convert list of tuples to two lists
    # data points with label <oov/> are removed because they are noise for the training
    train_data, train_labels = zip(*embedding_loader.tuple_to_indices(
        data_prepocessing.ngram_tuplelizer(poems[:3000], n), dictionary, True))
    dev_data, dev_labels = zip(*embedding_loader.tuple_to_indices(
        data_prepocessing.ngram_tuplelizer(poems[3000:3500], n), dictionary, True))
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    dev_data = np.array(dev_data)
    dev_labels = np.array(dev_labels)

    callbacks = [ModelCheckpoint("models/model.hdf5", save_best_only=True)]
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs,
              validation_data=(dev_data, dev_labels), callbacks=callbacks)
