from keras.models import Sequential
from keras.layers import Activation, Dense, Embedding, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon
import numpy as np
import src.data_prepocessing as dp
import src.embedding as embedding_loader
from src.global_constants import TRAINING_DATA, EMBEDDING_DIMENSION, EMBEDDING_BINARY, MODELS_DICT

def create_model(n, embedding, vocab_len):
    model = Sequential([
        Embedding(input_dim=vocab_len, output_dim=EMBEDDING_DIMENSION, input_length=n, weights=[embedding]),
        Flatten(),
        Dropout(0.5),
        Dense(n*500),
        Activation("relu"),
        Dropout(0.5),
        Dense(vocab_len),
        Activation("softmax")
    ])
    model.compile(optimizer=optimizer.Adam(lr=5e-4),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

if __name__ == "__main__":
    set_floatx("float16")
    set_epsilon(1e-04)
    n = 7
    epochs = 15
    batch_size = 512
    max_limit = 40000

    poems = dp.tokenize_poems(TRAINING_DATA)
    words = set([token for poem in poems for token in poem])

    #Save embedding for generator
    embedding, dictionary = embedding_loader.get_embedding(words, binary=EMBEDDING_BINARY, limit=max_limit, save=True)
    model = create_model(n, embedding, len(dictionary))

    # tuplelize train and dev set, convert to indices and convert list of tuples to two lists
    # data points with label <oov/> are removed because they are noise for the training
    train_data, train_labels = zip(*embedding_loader.tuple_to_indices(
        dp.ngram_tuplelizer(poems, n), dictionary, True))
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    callbacks = [ModelCheckpoint(MODELS_DICT+"//model.hdf5", save_best_only=True)]
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=0.15)
