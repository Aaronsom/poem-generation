from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.optimizers as optimizer
from keras.backend import set_floatx, set_epsilon
from poem_generator.dataGenerator import TupleDataGenerator
import poem_generator.data_prepocessing as dp
import poem_generator.embedding as embedding_loader
from poem_generator.global_constants import TRAINING_DATA, EMBEDDING_DIMENSION, EMBEDDING_BINARY, MODELS_DICT
from poem_generator.transformer import transformer

if __name__ == "__main__":
    set_floatx("float16")
    set_epsilon(1e-04)
    ns = [25]
    epochs = 20
    batch_size = 512
    validation_split = 0.9

    poems = dp.characterize_poems(TRAINING_DATA)
    words = sorted(list(set([token for poem in poems for token in poem])))

    #Save embedding for generator
    embedding, dictionary = embedding_loader.get_char_embedding(words, save=True)

    #model = load_model(MODELS_DICT+"/5model.hdf5", custom_objects={"PositionalEncoding": PositionalEncoding, "Attention": Attention})
    model = transformer(100, embedding, len(dictionary), True, train_embedding=True)

    model.compile(optimizer=optimizer.Adam(decay=1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    generator = TupleDataGenerator(poems[:int(validation_split*len(poems))], ns, dictionary, 0.1, batch_size, single=True)
    validation_generator = TupleDataGenerator(poems[int(validation_split*len(poems)):], ns, dictionary, 0, batch_size, single=True)
    callbacks = [ModelCheckpoint(MODELS_DICT+"/char-model.hdf5", save_best_only=True),
                 CSVLogger(MODELS_DICT+"/char-log.csv", append=True, separator=';')]
    model.fit_generator(
        generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, workers=4)
