from tensorflow.keras.callbacks import Callback
from poem_generator.word_generator import generate_poem


class PoemCallback(Callback):

    def __init__(self, poems, seed_length, dictionary, single=True):
        super(PoemCallback, self).__init__()
        self.poems = poems
        self.dictionary = dictionary
        self.reverse_dictionary = {dictionary[key]: key for key in dictionary.keys()}
        self.seed_length = seed_length
        self.single = single

    def on_epoch_end(self, epoch, logs=None):
        for i in range(self.poems):
            print(f"Poem {i+1}/{self.poems}")
            generate_poem(self.model, self.reverse_dictionary, self.dictionary, self.seed_length, single=self.single)