from poem_generator.global_constants import END_OF_LINE_TOKEN, END_OF_SEQUENCE_TOKEN, START_OF_SEQUENCE_TOKEN

def tokenize_poems(file):
    poems = []
    with open(file, encoding="utf-8") as f:
        poem = []
        for line in f.readlines():
            if line == "\n":
                poem[-1] = END_OF_SEQUENCE_TOKEN
                poems.append(poem)
                poem = []
            else:
                tokens = line.rstrip("\n").lower().split(" ") + [END_OF_LINE_TOKEN]
                poem.extend(tokens)
    return poems

def characterize_poems(file):
    poems = []
    with open(file, encoding="utf-8") as f:
        poem = []
        for line in f.readlines():
            if line == "\n":
                poem[-1] = END_OF_SEQUENCE_TOKEN
                poems.append(poem)
                poem = []
            else:
                tokens = list(line.rstrip("\n").lower()) + [END_OF_LINE_TOKEN]
                poem.extend(tokens)
    return poems

def multi_ngram_tupelizer(poems, ns, sos_pad=True, first_only=False, single=False):
    tuples = []
    for n in ns:
        tuples.append(ngram_tuplelizer(poems, n, sos_pad, first_only, single))
    return tuples

def ngram_tuplelizer(poems, n, sos_pad=True, first_only=False, single=False):
    tuples = []
    pad = n if sos_pad else 1
    poems = [[START_OF_SEQUENCE_TOKEN]*pad + poem for poem in poems]
    for poem in poems:
        i = 0
        while i+n < len(poem):
            if single:
                tuples.append((poem[i:i+n], poem[i+n]))
            tuples.append((poem[i:i+n], poem[i+1:i+n+1]))
            i += 1
            if first_only:
                break
    return tuples


if __name__ == "__main__":
    poems = tokenize_poems("../data/train_poems.txt")
    tuples = multi_ngram_tupelizer(poems, range(2, 70), sos_pad=False, first_only=True)
    print(tuples)