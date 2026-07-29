import numpy as np
from collections import defaultdict

corpus = [
    "king rules kingdom with power", "queen rules kingdom with grace",
    "man eats apple happily", "woman eats orange happily",
    "king eats apple slowly", "queen eats orange slowly",
    "man drinks water daily", "woman drinks water daily",
    "king wears crown proudly", "queen wears crown proudly",
    "kings ruled many kingdoms", "the kingdom obeys the king",
]

def char_ngrams(word, n=3):
    padded = "<" + word + ">"
    return set(padded[i:i + n] for i in range(len(padded) - n + 1))

window = 2
ngram_cooc = defaultdict(lambda: defaultdict(int))
all_words = set(w for s in corpus for w in s.split())

for sentence in corpus:
    tokens = sentence.split()
    for i, tok in enumerate(tokens):
        start, end = max(0, i - window), min(len(tokens), i + window + 1)
        context_words = [tokens[j] for j in range(start, end) if j != i]
        for g in char_ngrams(tok):
            for ctx in context_words:
                ngram_cooc[g][ctx] += 1

context_vocab = sorted(all_words)
ngram_vocab = sorted(ngram_cooc.keys())
M = np.array([[ngram_cooc[g][c] for c in context_vocab] for g in ngram_vocab], dtype=float)

U, S, Vt = np.linalg.svd(M, full_matrices=False)
d = 5
ngram_embeddings = {g: (U[i, :d] * S[:d]) for i, g in enumerate(ngram_vocab)}

def word_vector(word):
    vecs = [ngram_embeddings[g] for g in char_ngrams(word) if g in ngram_embeddings]
    return np.sum(vecs, axis=0) if vecs else None

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

king_v, queen_v, apple_v = word_vector("king"), word_vector("queen"), word_vector("apple")
oov_v = word_vector("kingly")   # never appears anywhere in the corpus, in any form

print("cos(king, queen)        =", cosine(king_v, queen_v))
print("cos(king, apple)        =", cosine(king_v, apple_v))
print("cos(kingly[OOV], king)  =", cosine(oov_v, king_v))
print("cos(kingly[OOV], queen) =", cosine(oov_v, queen_v))
print("cos(kingly[OOV], apple) =", cosine(oov_v, apple_v))