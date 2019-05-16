from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle

import data_prepocessing as dp

EMBEDDING_PATH = r"C:\Users\Gregor\Documents\Programming\embedding\glove.6B.50d.txt" #"DATA/glove.6B.50d.txt""

EMBEDDING_FILE = r"DATA/embedding.pkl"
DICTIONARY_FILE = r"DATA/dict.pkl"

EMBEDDING_DIM = 50

def get_embedding(words, save=False, load=False, limit=400000):
    if load:
        embedding = pickle.load(open(EMBEDDING_FILE, "rb"))
        dictionary = pickle.load(open(DICTIONARY_FILE, "rb"))
        return embedding, dictionary
    entire_embedding = KeyedVectors.load_word2vec_format(EMBEDDING_PATH, limit=limit)
    # init with special tokens
    embedding = [np.random.normal(size=EMBEDDING_DIM), np.random.normal(size=EMBEDDING_DIM),
                 np.random.normal(size=EMBEDDING_DIM), np.random.normal(size=EMBEDDING_DIM)]
    dictionary = {"<oov/>": 0, "<eos/>": 1, "<eol/>": 2, "<sos/>": 3}
    for word in words:
        if word in dictionary:
            continue
        try:
            word_embedding = entire_embedding[word]
        except KeyError:
            continue # keep vocab small
        dictionary[word] = len(dictionary)
        embedding.append(word_embedding)
    embedding = np.array(embedding)
    if save:
        pickle.dump(embedding, open(EMBEDDING_FILE, "wb"))
        pickle.dump(dictionary, open(DICTIONARY_FILE, "wb"))
    return embedding, dictionary

def tuple_to_indices(ngram_tuples, dictionary, remove_oov_labels=False):
    index_ngram_tuples = []
    for ngram, label in ngram_tuples:
        index_ngram = []
        for word in ngram:
            try:
                index_ngram.append(dictionary[word])
            except KeyError:
                index_ngram.append(dictionary["<oov/>"])
        try:
            index_label = dictionary[label]
        except KeyError:
            if remove_oov_labels:
                continue
            else:
                index_label = dictionary["<oov/>"]
        index_ngram_tuples.append((index_ngram, index_label))
    return index_ngram_tuples

if __name__ == "__main__":
    poems = dp.tokenize_poems("DATA/train_poems.txt")
    words = set([token for poem in poems for token in poem])
    embedding, dictionary = get_embedding(words)

    ngram_tuples = dp.ngram_tuplelizer([poems[0]], 7)
    converted_tuple = tuple_to_indices(ngram_tuples, dictionary)
    print(converted_tuple)


