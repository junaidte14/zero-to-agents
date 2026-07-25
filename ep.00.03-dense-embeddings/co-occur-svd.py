import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD

# 1. Setup vocabulary and corpus
vocabulary = ["king", "queen", "man", "woman", "apple", "orange"]
word_to_index = {w: i for i, w in enumerate(vocabulary)}

corpus = [
    "king rules kingdom with power",
    "queen rules kingdom with grace",
    "man eats apple happily",
    "woman eats orange happily",
    "king eats apple slowly",
    "queen eats orange slowly",
    "man drinks water daily",
    "woman drinks water daily",
    "king wears crown proudly",
    "queen wears crown proudly",
]

# 2. Build co-occurrence dictionary
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

# 3. Create dense co-occurrence matrix M
context_vocab = sorted(set(w for s in corpus for w in s.split()))

def cooc_vector(word):
    return [cooc[word][ctx] for ctx in context_vocab]

M = np.array([cooc_vector(w) for w in vocabulary], dtype=float)
print("Co-occurrence matrix shape:", M.shape)

# 4. Method A: Manual NumPy SVD
U, S, Vt = np.linalg.svd(M, full_matrices=False)
d = 2
dense_embeddings = U[:, :d] * S[:d]

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

idx = word_to_index
king, queen, man, apple = (dense_embeddings[idx[w]] for w in ["king", "queen", "man", "apple"])

print("\n--- NumPy SVD Cosine Similarities ---")
print("cos(king, queen) =", cosine(king, queen))
print("cos(king, man)   =", cosine(king, man))
print("cos(king, apple) =", cosine(king, apple))

# 5. Method B: Scikit-Learn TruncatedSVD
svd = TruncatedSVD(n_components=2, random_state=42)
sklearn_embeddings = svd.fit_transform(M)

king_sk, queen_sk, man_sk, apple_sk = (sklearn_embeddings[idx[w]] for w in ["king", "queen", "man", "apple"])

print("\n--- TruncatedSVD Cosine Similarities ---")
print("cos(king, queen) =", cosine(king_sk, queen_sk))
print("cos(king, man)   =", cosine(king_sk, man_sk))
print("cos(king, apple) =", cosine(king_sk, apple_sk))