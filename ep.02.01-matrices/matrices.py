import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
x = np.array([1, 0, -1])

def matvec_by_hand(A, x):
    m, n = A.shape
    return np.array([sum(A[i, j] * x[j] for j in range(n)) for i in range(m)])

print("By hand:", matvec_by_hand(A, x))
print("NumPy:  ", A @ x)
print("Row 0 . x =", np.dot(A[0], x), "<- matches output[0], confirming §2.1")


vocab = ["king", "queen", "man", "woman"]
word_to_idx = {w: i for i, w in enumerate(vocab)}
EmbeddingMatrix = np.array([
    [-1.4, 0.7, 2.1, 0.3], [-1.3, 0.9, 2.0, 0.4],
    [ 0.2,-0.5, 0.1, 1.2], [ 0.3,-0.4, 0.0, 1.3],
])

def one_hot(word):
    v = np.zeros(len(vocab)); v[word_to_idx[word]] = 1; return v

lookup_via_matmul = one_hot("queen") @ EmbeddingMatrix
direct_row = EmbeddingMatrix[word_to_idx["queen"]]
print(np.allclose(lookup_via_matmul, direct_row))  # True


static_embed = {
    "sat":   np.array([0.1,  0.2,  0.3, 0.4]),
    "bank":  np.array([0.3, -0.4,  0.0, 1.3]),
    "river": np.array([0.4, -0.3,  0.1, 1.5]),
    "money": np.array([0.2, -0.5,  0.1, 1.2]),
}

def full_self_attention(sentence_tokens, static_embed):
    X = np.array([static_embed[t] for t in sentence_tokens])  # (n, d) -- Q, K, V all reuse X here
    d_k = X.shape[1]
    scores = (X @ X.T) / np.sqrt(d_k)          # (n, n) -- every pairwise dot product, ONE matmul
    np.fill_diagonal(scores, -np.inf)           # exclude self, matching Episode 01.02's setup
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights = weights / weights.sum(axis=1, keepdims=True)   # row-wise softmax
    return weights, weights @ X                 # (n,n) @ (n,d) -> (n,d): every word's new vector, at once

sent_river = "sat bank river".split()
weights, output = full_self_attention(sent_river, static_embed)

print("Weight matrix (row = query word, column = key word):")
print("        ", "  ".join(f"{t:>8s}" for t in sent_river))
for i, t in enumerate(sent_river):
    print(f"{t:8s}", "  ".join(f"{w:8.3f}" for w in weights[i]))

def cosine(v1, v2): return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
bank_idx = sent_river.index("bank")
print(round(cosine(output[bank_idx], static_embed["money"]), 3))
print(round(cosine(output[bank_idx], static_embed["river"]), 3))