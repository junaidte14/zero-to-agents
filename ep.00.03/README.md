# From Zero to Agents
## Module 00 — Introduction
### Episode 00.03: Dense Embeddings — Compressing Meaning Into Fewer Dimensions

---

## 0. Where we're starting from

Episode 00.02 left two problems open. We're solving the first one today: co-occurrence vectors gave us real, graded similarity between words, but they're the same size as the vocabulary — mostly zeros, hugely redundant. The question posed at the end: how do you compress that down to a small, dense vector while keeping the relational structure that made "king" and "queen" come out similar?

There turn out to be two genuinely different families of answer, and it's worth knowing both exist before picking one, because you'll meet both again — the first resurfaces when we hit dimensionality reduction and PCA properly in Module 02; the second, word2vec, is the direct conceptual ancestor of the embedding layer sitting at the bottom of every transformer we build starting in Module 04.

## 1. Theory: two roads to the same destination

**1.1 Road one — count, then compress (matrix factorization).**
We already have a full co-occurrence matrix — every word's relationship to every context word, all spelled out, huge and sparse. The compression idea here is: don't throw that matrix away and start over — **squeeze it**. Find a much smaller matrix that, when you "unsqueeze" it, reconstructs something close to the original. The mathematical tool for exactly this — take a big matrix, find the best possible low-rank approximation of it — is the **Singular Value Decomposition (SVD)**, which we'll use today as a working tool and formalize properly in Module 02. This general family (co-occurrence + SVD) is sometimes called **Latent Semantic Analysis (LSA)** in the older NLP literature.

**1.2 Road two — don't count, predict (word2vec).**
The second idea, introduced by Mikolov et al. at Google in 2013, throws away the explicit counting step entirely. Instead: train a tiny neural network to do one job — given a word, predict which words are likely to appear near it (or the reverse: given surrounding words, predict the missing center word). You never explicitly build a co-occurrence matrix at all. The network has a hidden layer far smaller than the vocabulary, and *that hidden layer's learned weights, once training is done, are the word embeddings.* The network was never asked to produce embeddings directly — they fall out as a side effect of getting good at the prediction task. This is called **skip-gram** (predict context from center word) or **CBOW** — Continuous Bag of Words (predict center word from context); we'll use skip-gram as the running example.

**1.3 Why this works at all — the surprising part.**
Neither approach is told anything about grammar, gender, royalty, or fruit. Both only ever see raw co-occurrence statistics. And yet word2pec-trained embeddings famously support **vector arithmetic that tracks real-world relationships**: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$. Nobody hand-designed a "royalty" or "gender" direction in the vector space — it emerged because the training signal (context prediction) happened to be rich enough that the geometry organizing itself around directions like "royal-ness" and "gender" was simply the *most efficient* way to solve the prediction task. This is the first real example in this course of a pattern that will repeat constantly from here on: **structure that looks designed actually emerged from optimization pressure alone.** Hold onto that idea; it's most of what "deep learning" turns out to mean in practice.

**1.4 What "dense" actually buys you.**
A one-hot vector is $V$-dimensional (vocabulary size — tens of thousands or more) and interpretable at a glance (exactly one 1 tells you exactly which word). A dense embedding is typically 50–300 dimensions, and **no single dimension means anything on its own** — meaning lives in the *combination* and *direction* of many dimensions together. This is called a **distributed representation**, and it's a genuine trade: you lose per-dimension interpretability, but you gain a representation dense enough for a neural network to do useful arithmetic on, and small enough to actually train and store at scale.

## 2. Math: formalizing both roads

**2.1 SVD, as a tool (full derivation deferred to Module 02).**
Given a co-occurrence matrix $M \in \mathbb{R}^{V \times V}$ (or $V \times C$ for a separate context vocabulary), SVD factors it as:

$$M = U \Sigma V^T$$

where $U$ and $V$ are matrices with orthonormal columns, and $\Sigma$ is diagonal, holding the **singular values** in decreasing order of importance — think of them as "how much of the matrix's structure this direction explains," largest first. The compression move: keep only the top $d$ singular values (and the corresponding columns of $U$), throwing away the rest:

$$\text{Embedding of word } i = (U_{:,1:d} \, \Sigma_{1:d,1:d})_{i,:} \in \mathbb{R}^d$$

This is provably the **best possible rank-$d$ approximation** of $M$ in a precise mathematical sense (this is the Eckart–Young theorem — filed away for Module 02, not proven here). Practically: it keeps the directions in the data that explain the most variation, and discards the rest as noise/redundancy.

**2.2 The skip-gram objective (full training mechanics deferred to Module 03).**
Given a corpus of $T$ words $w_1, \ldots, w_T$ and a context window of size $c$, skip-gram tries to find embeddings that maximize:

$$\frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-c \le j \le c \\ j \ne 0}} \log p(w_{t+j} \mid w_t)$$

Read it outward-in: for every word $w_t$ in the corpus, look at every nearby word $w_{t+j}$ within the window, and add up how well the model currently predicts each nearby word given the center word — averaged over the whole corpus. Training pushes the embeddings toward whatever configuration makes those predictions as accurate as possible.

The probability itself is a **softmax** over the whole vocabulary:

$$p(w_O \mid w_I) = \frac{\exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}'_w \cdot \mathbf{v}_{w_I})}$$

Here $\mathbf{v}_{w_I}$ is the "input" embedding of the center word, $\mathbf{v}'_{w_O}$ is the "output" embedding of a candidate context word — the model actually keeps *two* embedding tables during training, one per role, and conventionally throws away the output table afterward, keeping only $\mathbf{v}$. The numerator scores how well this specific center–context pair fits together; the denominator normalizes across every word in the vocabulary so the whole thing is a valid probability distribution. That denominator — summing over the entire vocabulary for every single training step — is expensive, which is exactly why real implementations use **negative sampling** (approximate the sum using a small random subset instead of the full vocabulary) rather than computing it exactly. We won't implement gradient descent by hand for this yet — that machinery (backpropagation, softmax, loss functions) is Module 03's job, done properly and reused everywhere after. Today we use the *idea* and a library that has already implemented the mechanics.

## 3. Code: both roads, from the co-occurrence matrix up

**3.1 From scratch — SVD compression of the co-occurrence matrix**

```python
import numpy as np
from collections import defaultdict

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

context_vocab = sorted(set(w for s in corpus for w in s.split()))
def cooc_vector(word):
    return [cooc[word][ctx] for ctx in context_vocab]

M = np.array([cooc_vector(w) for w in vocabulary], dtype=float)
print("Sparse co-occurrence matrix shape:", M.shape)   # (6, 20) -- 20-dim, mostly zeros

U, S, Vt = np.linalg.svd(M, full_matrices=False)

d = 2
dense_embeddings = U[:, :d] * S[:d]   # scale the top-d directions by their singular values
print("Dense embedding shape:", dense_embeddings.shape)  # (6, 2)

def cosine(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

idx = word_to_index
king, queen, man, apple = (dense_embeddings[idx[w]] for w in ["king", "queen", "man", "apple"])
print("cos(king, queen) =", cosine(king, queen))  # 1.0
print("cos(king, man)   =", cosine(king, man))    # 0.76
print("cos(king, apple) =", cosine(king, apple))  # 0.30
```

Compressed from 20 dimensions down to 2, the relational ordering survives and even sharpens: king–queen (royal pair) are most similar, king–man (both people) are moderately similar, king–apple (unrelated) are least similar. Note king and queen land on the *exact* same 2-D point here — a toy-corpus artifact (our eight sentences treat king/queen as perfectly interchangeable), not something you'd see at real scale, where no two words share *identical* usage statistics.

**3.2 Using a library — the same SVD idea, production-grade**

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=2, random_state=42)
sklearn_embeddings = svd.fit_transform(M)   # same M as above
print(sklearn_embeddings)
```

Same mechanism as 3.1, handling numerical edge cases and scaling to matrices far larger than we'd want to run raw NumPy SVD on by hand.

**3.3 Predict, don't count — word2vec via `gensim`**

```python
from gensim.models import Word2Vec
import random

random.seed(42)
royals = ["king", "queen"]; royal_actions = ["rules", "wears", "commands"]; royal_objects = ["kingdom", "crown", "throne", "army"]
commoners = ["man", "woman"]; commoner_actions = ["eats", "drinks", "buys"]; commoner_objects = ["apple", "orange", "water", "bread"]

sentences = []
for _ in range(500):
    if random.random() < 0.5:
        s, a, o = random.choice(royals), random.choice(royal_actions), random.choice(royal_objects)
    else:
        s, a, o = random.choice(commoners), random.choice(commoner_actions), random.choice(commoner_objects)
    sentences.append([s, a, o])

model = Word2Vec(sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100, seed=42)

print(model.wv.most_similar("king", topn=3))
# [('queen', 0.994), ('kingdom', 0.986), ('crown', 0.985)]
print(model.wv.similarity("king", "apple"))
# 0.74 -- still positive (small synthetic vocabulary), but clearly the weakest relationship
```

No co-occurrence matrix was ever built here. The network was only ever trained to predict nearby words from a center word, 500 tiny three-word sentences, 100 passes over the data — and "queen" comes out as the single most similar word to "king," ahead of every other word in the vocabulary, purely as a side effect of getting good at that prediction task.

## 4. Where this leaves us

Both roads solve the compression problem from Episode 00.02: co-occurrence-plus-SVD gives you a mathematically optimal low-rank summary of statistics you've already gathered; word2vec skips gathering those statistics explicitly and lets a small prediction task discover a good embedding as a byproduct of training. In practice, predictive approaches like word2vec (and its successors — GloVe, fastText, and eventually the embeddings learned inside a transformer) won out, mostly because they scale better to enormous corpora and because the embeddings they produce tend to capture sharper relational structure — that king–man+woman≈queen arithmetic doesn't come nearly as cleanly out of raw SVD.

Two problems are still deliberately unsolved:
- Every embedding technique so far assigns **one fixed vector per word**, regardless of context. "Bank" gets one embedding whether it means a riverbank or a financial institution. That's a real limitation we're carrying forward on purpose — it's exactly the problem attention mechanisms (Module 04+) exist to solve.
- We still haven't touched **tokenization** properly — real vocabularies handle misspellings, made-up words, and non-space-delimited languages using subword units (byte-pair encoding and friends), not whole-word splitting.

## 5. Before Episode 00.04

We're closing out Module 00 soon and moving into Module 01 proper, where these threads get formalized one at a time. Before that: reread the word2vec output in §3.3 — `similarity("king", "apple") = 0.74` is still fairly high, even though apple is clearly the least related word in that toy vocabulary.

> Why would a word2vec-style model, trained correctly, still assign a *fairly high* similarity score to two genuinely unrelated words like "king" and "apple," rather than something close to zero? Think about what the vector space actually represents, and what "zero similarity" would even require geometrically once every word is squeezed into the same small, dense, densely-populated space — no need for a precise answer, just a working hunch.

That question is where Module 01 picks up properly, starting with tokenization — the piece we've been quietly skipping this entire module.

---

**Previous:** Episode 00.02 — From Words to Numbers: The Representation Problem
**Next:** Episode 00.04 — Module 00 Wrap-Up, and What Module 01 Covers