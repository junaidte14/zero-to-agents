from collections import defaultdict
import math

corpus = [
    "king rules kingdom",
    "queen rules kingdom",
    "man eats apple",
    "woman eats orange",
    "king eats apple",
    "queen eats orange",
    "man drinks water",
    "woman drinks water",
]

vocabulary = ["king", "queen", "man", "woman", "apple", "orange"]
vocab_size = len(vocabulary)
word_to_index = {w: i for i, w in enumerate(vocabulary)}
print(word_to_index)

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


window = 2
cooc = defaultdict(lambda: defaultdict(int))
for sentence in corpus:
    tokens = sentence.split()
    for i, tok in enumerate(tokens):
        if tok not in word_to_index:
            continue
        start, end = max(0, i - window), min(len(tokens), i + window + 1)
        for j in range(start, end):
            if j != i:
                cooc[tok][tokens[j]] += 1

print(cooc)
context_vocab = sorted(set(w for s in corpus for w in s.split()))
print(context_vocab)

def cooc_vector(word):
    out = [cooc[word][ctx] for ctx in context_vocab]
    print(word)
    print(out)
    return out

king_v, queen_v, apple_v = cooc_vector("king"), cooc_vector("queen"), cooc_vector("apple")
print("cos(king, queen) =", cosine_similarity(king_v, queen_v))  # 0.75
print("cos(king, apple) =", cosine_similarity(king_v, apple_v))  # 0.41