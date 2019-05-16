import embedding as embedding_loader
import numpy as np
from keras.models import load_model
from keras.backend import set_epsilon, set_floatx

def generate_poem(model, reverse_dictionary, dictionary, seed_length):
    poem = ""
    last_output = ""
    iterations = 0
    seed = np.array([dictionary["<sos/>"]]*seed_length)
    already_eol = False  # Sometimes too many eols are generated, this breaks the format
    while iterations < 60 and last_output != "<eos/>":
        last_output_dist = model.predict(np.array([seed])).squeeze()
        last_output_idx = np.random.choice(len(dictionary), 1, p=last_output_dist).item()
        last_output = reverse_dictionary[last_output_idx]

        seed = np.append(seed[1:], last_output_idx)
        iterations += 1

        if last_output == "<eos/>" or iterations == 60:
            if already_eol:
                poem += "\n"
            else:
                poem += "\n\n"
        elif last_output == "<oov/>" or last_output == "<sos/>":
            pass
        elif last_output == "<eol/>":
            if  iterations>1 and not already_eol:
                already_eol = True
                poem += "\n"

        else:
            already_eol = False
            poem += last_output + " "
    print(poem)
    return poem

def generate_poems(num_of_poems, seed_length, output_filename, model_file):
    _, dictionary = embedding_loader.get_embedding(None, load=True)
    reverse_dictionary = {dictionary[key]: key for key in dictionary.keys()}
    model = load_model(model_file)
    generated_poems = ""
    for i in range(num_of_poems):
        print(f"{i+1}/{num_of_poems}")
        poem = generate_poem(model, reverse_dictionary, dictionary, seed_length)
        generated_poems += poem
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(generated_poems)

if __name__ == "__main__":
    set_floatx("float16")
    set_epsilon(1e-04)
    generate_poems(1000, 7, "generated/poems.txt", "models/model.hdf5")