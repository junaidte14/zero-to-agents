# From Zero to Agents
## Module 02 — Mathematical Foundations
### Episode 02.01: Matrices — Organized, Simultaneous Dot Products
 
---
 
## 0. Closing the open question
 
Episode 02.00 ended by asking: attention computes a similarity score between one query and *every* key in a sentence at once, not one dot product at a time in a loop — what kind of object lets you express that as a single computation? The answer is a **matrix**, and by the end of this episode we'll have used one to fully decode the actual attention equation from Vaswani et al.'s 2017 paper *"Attention Is All You Need"* — the paper the entire second half of this course is building toward.
 
## 1. Theory: three views of the same object, all of them things you already have
 
**1.1 A matrix is a rectangular table of numbers.**
Nothing more mysterious than that at first: an $m \times n$ matrix has $m$ rows and $n$ columns. But you've already been using matrices constantly without the formal name — the co-occurrence table from Episode 00.02 was a matrix (rows = words, columns = context words). The embedding table from every episode since Module 00 is a matrix too — an embedding table with $V$ words and $d$ dimensions per embedding is literally a $V \times d$ matrix, one row per word.
 
**1.2 A matrix as a collection of vectors, stacked.**
The most direct connection to Episode 02.00: a matrix's rows (or columns) are themselves vectors. An embedding table isn't a new kind of object at all — it's every word's embedding vector, stacked into rows. This reframing matters because it means "look up a word's embedding" and "select a row of a matrix" are the exact same operation, which §2 makes precise (and genuinely surprising) in a moment.
 
**1.3 A matrix as a linear transformation — a function that moves vectors.**
The third view, and the one that unlocks everything downstream: a matrix can be understood as a *function* that takes a vector in and produces a different vector out — rotating it, scaling it, projecting it, or some combination. A $2\times 2$ rotation matrix takes any 2D vector and spins it around the origin by a fixed angle, no matter which vector you feed it. This is the view that makes "training a neural network" meaningful later — a network layer is, at its core, a learned matrix that transforms input vectors into more useful ones.
 
## 2. Math: matrix-vector and matrix-matrix multiplication, stated precisely
 
**2.1 Matrix-vector multiplication — literally "many dot products, done at once."**
Given an $m \times n$ matrix $A$ and an $n$-dimensional vector $\mathbf{x}$, the product $A\mathbf{x}$ is an $m$-dimensional vector defined by:
 
$$(A\mathbf{x})_i = \sum_{j=1}^{n} A_{ij} x_j = \text{row}_i(A) \cdot \mathbf{x}$$
 
Read this precisely: **the $i$-th entry of the output is just the dot product of the $i$-th row of $A$ with $\mathbf{x}$.** This directly answers Episode 02.00's closing question: matrix-vector multiplication *is* "compute a dot product against every row simultaneously," expressed as one clean operation rather than a loop you write yourself.
 
**2.2 The genuinely surprising consequence — embedding lookup is matrix multiplication.**
Take the one-hot vector for a word (Episode 00.02) and multiply it against the embedding matrix (rows = word embeddings). Per §2.1, entry $i$ of the result is $\text{row}_i(\text{EmbeddingMatrix}) \cdot \text{one-hot vector}$. Since the one-hot vector is all zeros except a single 1, this dot product is zero everywhere except at the matching row, where it equals exactly that row. **The result is precisely that word's embedding row — reproduced exactly, by ordinary matrix multiplication, with no special "lookup" operation required at all.** This is not a coincidence or a trick — it's the literal mechanism real embedding layers use internally: "looking up an embedding" and "multiplying a one-hot vector by the embedding matrix" are mathematically the same operation, and frameworks like PyTorch's `nn.Embedding` are simply an optimized shortcut that skips computing all those zero-multiplications explicitly.
 
**2.3 Matrix-matrix multiplication — many matrix-vector products, stacked.**
For $A$ ($m \times n$) and $B$ ($n \times p$), the product $AB$ ($m \times p$) is defined entry-by-entry as:
 
$$(AB)_{ik} = \sum_{j=1}^{n} A_{ij} B_{jk}$$
 
The cleanest way to *think* about this: each column of $B$ is itself a vector, and multiplying $A$ by that single column is exactly the matrix-vector operation from §2.1. Matrix-matrix multiplication is just doing that once per column of $B$, and stacking the results side by side. This is the exact mechanism behind $QK^T$ in the attention formula — about to become concrete.
 
**2.4 Decoding the real equation — "Attention Is All You Need," in full.**
The complete scaled dot-product attention formula from Vaswani et al. (2017) is:
 
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
 
Every symbol, decoded with exact shapes, now that matrices are formal:
 
- $Q$ (queries) is an $n \times d_k$ matrix — $n$ words in the sentence, each represented as a $d_k$-dimensional query vector, all stacked into rows. This is exactly the "matrix as a collection of vectors" view from §1.2.
- $K$ (keys) is likewise $n \times d_k$ — one key vector per word.
- $K^T$ (the **transpose** of $K$ — flip rows and columns, turning it into $d_k \times n$) sets up the shapes so $Q K^T$ multiplies correctly: $(n \times d_k)(d_k \times n) = n \times n$.
- $QK^T$ is therefore an $n \times n$ matrix where **entry $(i,j)$ is exactly $\text{row}_i(Q) \cdot \text{row}_j(K)$** — the dot product between word $i$'s query and word $j$'s key, per §2.1. Every pairwise similarity score between every word and every other word, computed in one matrix multiplication — precisely the "organized, simultaneous dot products" Episode 02.00 was building toward, and exactly what Episode 01.02's Python loop was doing one pair at a time, by hand.
- Dividing by $\sqrt{d_k}$ is the scaling term already introduced conceptually in Episode 01.02 — now you can see precisely what it's scaling: every entry of that freshly-computed $n \times n$ score matrix.
- $\text{softmax}(\cdot)$ is applied row-by-row, turning each row of raw scores into proper probability weights that sum to 1 — exactly the normalization step from Episode 01.02's attention weights, now applied to every word's row simultaneously instead of one at a time.
- $V$ (values) is another $n \times d_v$ matrix of per-word vectors (often, but not necessarily, the same dimensionality as $K$).
- Multiplying the $n \times n$ weight matrix by $V$ ($n \times d_v$) gives an $n \times d_v$ output — per §2.3, this is $n$ separate weighted sums, one per row, each one a specific word's brand-new, fully-contextualized vector.
This single equation, once every symbol has a shape and a meaning attached, is now something you could pick out of any transformer paper and read correctly on sight — which was the entire point of this module.
 
## 3. Code: matrix operations verified, then the full equation implemented properly
 
**3.1 Matrix-vector multiplication, and embedding lookup as multiplication**
 
```python
import numpy as np
 
A = np.array([[1, 2, 3], [4, 5, 6]])
x = np.array([1, 0, -1])
 
def matvec_by_hand(A, x):
    m, n = A.shape
    return np.array([sum(A[i, j] * x[j] for j in range(n)) for i in range(m)])
 
print("By hand:", matvec_by_hand(A, x))
print("NumPy:  ", A @ x)
print("Row 0 . x =", np.dot(A[0], x), "<- matches output[0], confirming §2.1")
```
```
By hand: [-2 -2]
NumPy:   [-2 -2]
Row 0 . x = -2 <- matches output[0], confirming §2.1
```
 
```python
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
```
```
True
```
 
§2.2 confirmed directly: a one-hot vector times the embedding matrix produces exactly the same result as indexing the row directly. This is the actual mechanism, not a metaphor.
 
**3.2 Full self-attention over an entire sentence — one matrix operation, not a loop**
 
Reusing the "bank" co-occurrence embeddings from Episode 01.02, but this time computing every word's contextual vector simultaneously, exactly the way §2.4 describes:
 
```python
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
```
```
Weight matrix (row = query word, column = key word):
              sat      bank     river
sat         0.000     0.500     0.500
bank        0.374     0.000     0.626
river       0.374     0.626     0.000
```
 
```python
def cosine(v1, v2): return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
bank_idx = sent_river.index("bank")
print(round(cosine(output[bank_idx], static_embed["money"]), 3))  # 0.687
print(round(cosine(output[bank_idx], static_embed["river"]), 3))  # 0.987
```
```
0.687
0.987
```
 
Identical results to Episode 01.02's single-word loop (0.687 vs. 0.987) — but this time, `sat`, `bank`, *and* `river` all got fully contextualized in one pass, from a single matrix multiplication, exactly matching the real $QK^T$ mechanism from §2.4 rather than approximating it one word at a time. Look at the weight matrix itself: `bank`'s row shows it attends 0.626 to `river` and 0.374 to `sat`, exactly as before — but now `sat` and `river` also received their own fully-computed contextual vectors in the same operation, something Episode 01.02 explicitly flagged as still missing.
 
## 4. Where this leaves us
 
The real attention formula — the exact one from the paper, not an approximation of it — is now fully decoded and reproduced with working code, using only tools built across two episodes of linear algebra. What's still missing, precisely two things:
 
- $Q$, $K$, and $V$ in our implementation are all just the *same* static embeddings reused three times. Real transformers pass the input through three separate **learned linear transformations** (three trainable weight matrices) to produce genuinely different $Q$, $K$, and $V$ matrices — meaning the model *learns* what to treat as a query, a key, and a value, rather than being handed one fixed representation for all three roles.
- "Learned" requires a mechanism for *how* a matrix's values get adjusted based on how wrong its output was — which requires calculus (derivatives, gradients) that Module 02 hasn't built yet.
## 5. Before Episode 02.02
 
> Section 1.3 called a matrix "a function that transforms vectors" — rotating, scaling, projecting. If training a neural network means *adjusting* a matrix's entries so its transformation gets progressively better at some task, what mathematical tool would tell you *which direction* to adjust each entry in, and *by how much*, to make the output slightly less wrong?
 
That's the on-ramp into Episode 02.02 — derivatives and gradients, the actual mechanism behind "learning."
 
---
 
**Previous:** Episode 02.00 — Vectors, Norms, and Dot Products
**Next:** Episode 02.02 — Derivatives and Gradients: The Mechanism Behind "Learning"