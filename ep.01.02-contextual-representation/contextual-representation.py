import numpy as np
from collections import defaultdict

corpus = [
    "deposited money bank", "bank approved loan", "opened account bank",
    "withdrew cash bank", "bank charged fee", "bank transferred funds",
    "loan from bank", "money in bank",
    "sat bank river", "fished along bank water", "boat drifted toward bank",
    "river bank covered mud", "birds nested bank stream", "swam near bank river",
    "bank overgrown reeds", "fish swim near bank",
]

window = 3
cooc = defaultdict(lambda: defaultdict(int))
all_words = set(w for s in corpus for w in s.split())
for sentence in corpus:
    tokens = sentence.split()
    for i, tok in enumerate(tokens):
        start, end = max(0, i - window), min(len(tokens), i + window + 1)
        for j in range(start, end):
            if j != i:
                cooc[tok][tokens[j]] += 1

words_sorted = sorted(all_words)
static_embed = {w: np.array([cooc[w][c] for c in words_sorted], dtype=float) for w in words_sorted}

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

print("Static 'bank' vs 'money':", round(cosine(static_embed["bank"], static_embed["money"]), 3))
print("Static 'bank' vs 'river':", round(cosine(static_embed["bank"], static_embed["river"]), 3))

def contextual_vector(word, sentence_tokens, static_embed):
    idx = sentence_tokens.index(word)
    neighbors = [static_embed[t] for j, t in enumerate(sentence_tokens) if j != idx]
    return np.mean(neighbors, axis=0)

finance_ctx = contextual_vector("bank", "deposited money bank".split(), static_embed)
river_ctx = contextual_vector("bank", "sat bank river".split(), static_embed)

print("\nContextual 'bank' (finance sentence) vs 'money':", round(cosine(finance_ctx, static_embed["money"]), 3))
print("Contextual 'bank' (finance sentence) vs 'river':", round(cosine(finance_ctx, static_embed["river"]), 3))
print("Contextual 'bank' (river sentence)   vs 'money':", round(cosine(river_ctx, static_embed["money"]), 3))
print("Contextual 'bank' (river sentence)   vs 'river':", round(cosine(river_ctx, static_embed["river"]), 3))


#using library

import torch
import torch.nn.functional as F

def to_tensor(sentence_tokens):
    return torch.tensor(np.array([static_embed[t] for t in sentence_tokens]), dtype=torch.float32)

def attend(word, sentence_tokens):
    X = to_tensor(sentence_tokens)
    idx = sentence_tokens.index(word)
    query = X[idx:idx + 1]
    mask = [i for i, t in enumerate(sentence_tokens) if t != word]
    keys = values = X[mask]
    d_k = query.shape[-1]
    scores = (query @ keys.T) / np.sqrt(d_k)
    weights = F.softmax(scores, dim=-1)
    for tok, w in zip([sentence_tokens[i] for i in mask], weights[0].tolist()):
        print(f"  {tok:10s} attention weight = {w:.3f}")
    return weights @ values

print("River sentence, attention weights for 'bank':")
river_attn = attend("bank", "sat bank river".split())