def tokenize_poems(file):
    poems = []
    with open(file, encoding="utf-8") as f:
        poem = []
        for line in f.readlines():
            if line == "\n":
                poem[-1] = "<eos/>"
                poems.append(poem)
                poem = []
            else:
                tokens = line.rstrip("\n").lower().split(" ") + ["<eol/>"]
                poem.extend(tokens)
    return poems

def ngram_tuplelizer(poems, n):
    tuples = []
    poems = [["<sos/>"]*n + poem for poem in poems]
    for poem in poems:
        i = 0
        while i+n < len(poem):
            tuples.append((poem[i:i+n], poem[i+n]))
            i += 1
    return tuples


if __name__ == "__main__":
    poems = tokenize_poems("DATA/train_poems.txt")
    print(ngram_tuplelizer([poems[0]], 7))