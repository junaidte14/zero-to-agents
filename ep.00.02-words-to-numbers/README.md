# From Zero to Agents
## Module 00 — Introduction
### Episode 00.02: From Words to Numbers — The Representation Problem
 
---
 
## 0. Where we're starting from
 
Episode 00.01 landed on a working definition of intelligence built around selecting good actions under uncertainty, and generalizing efficiently. Every example we used to demonstrate that — the lookup table, the nearest-neighbor classifier — quietly only ever operated on **numbers**: coordinates, labels, distances.
 
But almost every requirement you actually care about — "summarize this document," "make the customer happy," "book me a flight" — arrives as **language**. Before we can build anything that reasons about that, we need to answer a much more basic question: a neural network is, underneath everything, a machine that does arithmetic on numbers. So how does a word become a number — and what's lost, or gained, in that translation? This is the actual on-ramp into Module 01, and everything downstream (embeddings, attention, transformers) is a refinement of the answer we build here.
 
## 1. Theory: three attempts, in the order the field actually tried them
 
**1.1 Attempt one — just encode the characters (this doesn't work, and seeing why matters).**
Every piece of text is already stored as numbers at the hardware level — ASCII assigns `'A'` the number 65, Unicode extends this to the rest of the world's scripts. Problem: this encodes *symbols*, not *meaning*. The number 65 has no relationship to the number for `'B'` (66) that reflects anything linguistic — it's an arbitrary lookup, a historical accident of how the standard was designed. Two completely unrelated words can differ by one character and be numerically adjacent; two words with nearly identical meaning ("happy" / "glad") can have wildly different character codes. Character-level numeric encoding solves *storage*, not *understanding*. We need representations where numerical closeness reflects meaning closeness — that's the actual goal from here forward.
 
**1.2 Attempt two — one word, one slot (one-hot encoding).**
The next natural idea: build a vocabulary of every word you care about, give each one a unique index, and represent a word as a vector of all zeros except a single 1 at that word's index. This is a real representation neural networks can consume — it's numeric, it's fixed-size, it's well-defined for any word in the vocabulary.
 
But it has a specific, provable flaw, which section 2 will make precise: **every pair of distinct words is equally "far apart"** under this scheme. "King" and "queen" are exactly as dissimilar, numerically, as "king" and "banana." One-hot encoding throws away every bit of relational meaning between words — which is precisely the information a system needs if it's going to generalize (recall Episode 00.01: generalization is the whole game).
 
**1.3 Attempt three — let context define meaning (the distributional hypothesis).**
The idea that breaks the deadlock is a linguistics idea older than modern AI: the **distributional hypothesis**, most famously phrased by linguist J.R. Firth in 1957 — *"you shall know a word by the company it keeps."* The claim: a word's meaning is well approximated by the contexts it tends to appear in. "King" and "queen" show up near words like "rules," "kingdom," "throne." "Apple" and "orange" show up near "eats," "fruit," "juice." If two words share similar surrounding contexts across a large body of text, they probably mean similar things — even if a machine has no access to "meaning" in any human sense.
 
This single idea — infer meaning from co-occurrence patterns, not from a dictionary definition — is the seed of every word embedding technique that follows: co-occurrence matrices, word2vec, GloVe, and eventually the learned embeddings inside a transformer. Section 3 builds the simplest possible version of it from scratch, so you can watch the idea produce a numerically meaningful result with your own eyes before we make it fancier.
 
## 2. Math: proving one-hot encoding can't represent similarity
 
Let $V$ be the vocabulary size (the number of distinct words we're representing). A one-hot vector for the $i$-th word in the vocabulary is $\mathbf{e}_i \in \mathbb{R}^V$, defined as:
 
$$(\mathbf{e}_i)_k = \begin{cases} 1 & k = i \\ 0 & k \neq i \end{cases}$$
 
Now measure similarity between two *distinct* words $i \neq j$ using **cosine similarity** — the standard way to measure how "aligned" two vectors are, defined as:
 
$$\text{cos}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert \mathbf{u} \rVert \, \lVert \mathbf{v} \rVert}$$
 
The numerator $\mathbf{u} \cdot \mathbf{v}$ is the dot product — multiply corresponding entries and sum them. For any two distinct one-hot vectors $\mathbf{e}_i$ and $\mathbf{e}_j$ ($i \neq j$), every position where one vector is nonzero, the other is zero by construction — so:
 
$$\mathbf{e}_i \cdot \mathbf{e}_j = \sum_{k=1}^{V} (\mathbf{e}_i)_k (\mathbf{e}_j)_k = 0$$
 
which makes $\text{cos}(\mathbf{e}_i, \mathbf{e}_j) = 0$ **for every distinct pair, with no exceptions.** This isn't an approximation or a tendency — it's a direct algebraic consequence of the definition. One-hot vectors are, by construction, mutually orthogonal: geometrically, every word sits on its own perpendicular axis, equally "unrelated" to every other word. There is no vocabulary, no clever indexing scheme, that fixes this — the flaw is structural, not incidental.
 
Now contrast with a **co-occurrence vector**. Fix a context window of size $w$. For word $i$, define its co-occurrence vector $\mathbf{c}_i \in \mathbb{R}^V$ where:
 
$$(\mathbf{c}_i)_k = \text{number of times word } k \text{ appears within } w \text{ tokens of word } i \text{, across the corpus}$$
 
Unlike one-hot vectors, two different words $i$ and $j$ that tend to appear near the same surrounding words will have co-occurrence vectors with **overlapping nonzero entries**, so $\mathbf{c}_i \cdot \mathbf{c}_j > 0$, and $\text{cos}(\mathbf{c}_i, \mathbf{c}_j)$ becomes a genuine, graded, computable measure of how similarly two words are used — the exact thing one-hot encoding was structurally incapable of producing. Section 3 computes this number directly and shows it behaves the way intuition demands.
 
## 3. Code: watching the two representations actually diverge
 
**3.1 From scratch — one-hot vectors and their forced-zero similarity**
 
```python
import math
 
vocabulary = ["king", "queen", "man", "woman", "apple", "orange"]
vocab_size = len(vocabulary)
word_to_index = {w: i for i, w in enumerate(vocabulary)}
 
def one_hot(word):
    vec = [0] * vocab_size
    vec[word_to_index[word]] = 1
    return vec
 
def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
 
king, queen, apple = one_hot("king"), one_hot("queen"), one_hot("apple")
print("cos(king, queen) =", cosine_similarity(king, queen))  # 0.0
print("cos(king, apple) =", cosine_similarity(king, apple))  # 0.0 -- identical to the line above!
```
 
Run it, and confirm the proof from §2 with your own eyes: **both similarities come out exactly 0.0**, despite "king"/"queen" being obviously more related than "king"/"apple." One-hot encoding cannot express that distinction, structurally, no matter how the vocabulary is built.
 
**3.2 From scratch — a tiny co-occurrence embedding, built from the distributional hypothesis**
 
```python
from collections import defaultdict
 
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
 
king_v, queen_v, apple_v = cooc_vector("king"), cooc_vector("queen"), cooc_vector("apple")
print("cos(king, queen) =", cosine_similarity(king_v, queen_v))  # 0.75
print("cos(king, apple) =", cosine_similarity(king_v, apple_v))  # 0.41
```
 
From a corpus of eight short sentences — nothing more — "king" and "queen" come out **0.75 similar**, while "king" and "apple" come out **0.41 similar**. Nobody told the system that kings and queens are both royalty. It inferred a graded, meaningful distinction purely from which words tend to appear near which other words — exactly Firth's hypothesis from §1.3, made computational. This is a genuinely primitive embedding — real ones (word2vec onward, coming later in Module 01) are far more refined — but the core mechanism, "meaning from context," is already fully present here.
 
**3.3 Using a library — the same co-occurrence idea, vectorized**
 
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)  # bag-of-words counts per sentence, not per word --
                                       # a related but distinct representation, useful to compare
 
print(vectorizer.get_feature_names_out())
print(X.toarray())
```
 
Worth noting explicitly: `CountVectorizer` here builds a **document**-level representation (one vector per sentence), not a **word**-level co-occurrence embedding like we hand-built in 3.2. That distinction — word-level vs. document-level representations — is exactly the kind of thing that's easy to blur when you jump straight to a library without building the raw version first, which is why we built 3.2 by hand before reaching for `sklearn`.
 
## 4. Where this leaves us
 
Character encoding solves storage but throws away meaning. One-hot encoding is numerically valid but structurally incapable of expressing similarity — every pair of words is forced to be equally, maximally unrelated, and no vocabulary design fixes that. Co-occurrence vectors, built from nothing more than the distributional hypothesis, produce the first genuinely graded, meaningful numerical relationships between words we've seen in this course.
 
Two problems remain, both deliberately left open:
- Co-occurrence vectors have dimension $V$ (vocabulary size) — for a real vocabulary of tens of thousands of words, that's enormous and mostly sparse (zeros). We want something **dense** and **low-dimensional** that keeps the same relational structure.
- We split text into whole words by calling `.split()` — real text has punctuation, misspellings, made-up words, and languages where "words" aren't separated by spaces at all. We quietly dodged the **tokenization** problem entirely.
## 5. Before Episode 00.03
 
Look back at the `context_vocab` list printed in §3.2. Notice it's exactly as long as the number of distinct words across all eight sentences — meaning our "embedding" vectors are still nearly as large and sparse as one-hot vectors were, just with more useful numbers in them.
 
> If real vocabularies have hundreds of thousands of words, and co-occurrence vectors are that length, we've solved the *similarity* problem but not the *size* problem. Sit with this: how might you compress a huge, mostly-zero co-occurrence vector down to a small, dense one — say, 100 or 300 numbers — while keeping (most of) the relational structure that made "king" and "queen" come out similar? You don't need the exact algorithm, just a hunch about where the "extra," redundant information in a sparse vector might be hiding.
 
That question is the direct on-ramp into dense word embeddings — word2vec and friends — in Episode 00.03.
 
---
 
**Previous:** Episode 00.01 — What Is Intelligence?
**Next:** Episode 00.03 — Dense Embeddings: Compressing Meaning Into Fewer Dimensions