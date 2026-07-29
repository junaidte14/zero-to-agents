from collections import defaultdict

corpus = {"low": 5, "lower": 2, "lowest": 4, "newer": 6, "wider": 3, "new": 2}

def word_to_symbols(word):
    return tuple(word) + ("</w>",)

vocab = {word_to_symbols(w): freq for w, freq in corpus.items()}

def get_pair_freqs(vocab):
    pairs = defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab

merges = []
for step in range(10):
    pairs = get_pair_freqs(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merges.append(best)
    print(f"Merge {step + 1}: {best} (freq={pairs[best]})")

def encode(word, merges):
    symbols = list(word_to_symbols(word))
    for pair in merges:
        i, new_symbols = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols

for w in ["lowering", "newest", "wow"]:
    print(f"{w!r} -> {encode(w, merges)}")