# From Zero to Agents
## Module 01 — Representing Language as Numbers
### Episode 01.02: Toward Contextual Representations
 
---
 
## 0. Closing the open question
 
Episode 01.01 ended by asking what it would take for a word's vector to be computed fresh, for a specific sentence, using the words actually nearby right now — rather than one fixed vector baked in ahead of time from corpus-wide averages. Today we build the simplest possible version of exactly that, and in doing so, land within one short step of the mechanism — attention — that the rest of this course is ultimately building toward.
 
## 1. Theory: the problem static embeddings can never solve
 
**1.1 Polysemy — one spelling, several meanings.**
Every embedding technique built so far — one-hot, co-occurrence, SVD, word2vec, fastText — has one unbreakable assumption baked in: **one token gets one vector**, full stop, regardless of where it appears. This is fine for words with one dominant meaning. It quietly breaks for words like "bank," which means something entirely different in "I deposited money at the bank" versus "I sat by the river bank." A static embedding for "bank" has no choice but to average over *every* use of the word across the whole training corpus — becoming a blurry compromise that isn't quite the financial sense and isn't quite the river sense, but some corpus-weighted blend of both, applied identically no matter which sentence it's actually sitting in.
 
**1.2 Early fixes — give ambiguous words more than one vector.**
The first attempts at this problem, sometimes called **multi-prototype embeddings** (Reisinger & Mooney, 2010; Huang et al., 2012), tried clustering a word's different usage contexts and learning a *separate* static vector per cluster — one "bank (finance)" vector, one "bank (river)" vector. This helps, but it requires deciding the number of senses per word in advance, doesn't handle sentence-by-sentence nuance within a single "sense," and doesn't generalize to genuinely novel combinations of meaning. It's a patch, not a structural fix.
 
**1.3 The structural fix — make the vector a function of the sentence, not just the word.**
The real fix, formalized properly with **ELMo** (Peters et al., 2018 — "Deep contextualized word representations," using bidirectional LSTMs to read the whole sentence before producing any single word's vector) and refined dramatically by the attention mechanism (arriving in full in Module 04), is to stop treating "get a word's vector" as a table lookup entirely. Instead: **feed the whole sentence in, and let the model compute each word's vector by mixing in information from the other words actually present.** The word "bank" no longer has one canonical vector anywhere — it only ever has a vector *in the context of a specific sentence*, computed on the spot.
 
**1.4 The simplest possible version of "mixing in context."**
Before reaching for a trained model, notice we can approximate this idea with something almost embarrassingly simple, using only tools we already have: take a word's static embedding, and **blend it with the average of its neighboring words' static embeddings, for this specific sentence.** No training required — just arithmetic on vectors we've already built. This is a genuine, working (if crude) contextualization mechanism, and — this is the important part — it is a **direct conceptual ancestor of attention**. Attention, when we get to it properly, is this same idea with one upgrade: instead of averaging every neighbor *equally*, it learns to weight each neighbor by how *relevant* it actually is to the word being contextualized. Uniform averaging is attention's degenerate special case — attention where every weight happens to be equal. Section 3 builds both, back to back, so you can see the relationship directly.
 
## 2. Math: from fixed lookup to context-dependent function
 
**2.1 Formalizing the shift.**
A static embedding is a function of the word alone: $\mathbf{e}: V \rightarrow \mathbb{R}^d$, taking a vocabulary index and returning a fixed vector — this is everything built in Episodes 00.02 through 01.01. A **contextual** embedding is a function of the word *and its surrounding sequence*:
 
$$f(w_i, \, w_1, \ldots, w_n) \rightarrow \mathbb{R}^d$$
 
Notice this genuinely generalizes the static case — if $f$ simply ignores every argument except $w_i$, you're back to a plain lookup table. Everything from here forward is about designing $f$ to actually use the sequence, not throw it away.
 
**2.2 Simplest possible $f$ — neighbor averaging.**
Given a fixed self-weight $\alpha \in [0,1]$ and a context window, define:
 
$$f(w_i, \text{sentence}) = \alpha \cdot \mathbf{e}(w_i) + (1-\alpha) \cdot \frac{1}{|C_i|}\sum_{w_j \in C_i} \mathbf{e}(w_j)$$
 
where $C_i$ is the set of neighboring words in this specific sentence. Every word in $C_i$ contributes **equally** — this uniform contribution is exactly the limitation the next formula fixes.
 
**2.3 The attention-flavored version — weighted by relevance.**
Instead of a plain average, weight each neighbor by how relevant it is to the word being contextualized, measured as a dot-product similarity between the word's own vector (the **query**) and each neighbor's vector (the **key**), turned into proper weights via softmax:
 
$$\text{weight}(w_i, w_j) = \frac{\exp(\mathbf{e}(w_i) \cdot \mathbf{e}(w_j) / \sqrt{d})}{\sum_{w_k \in C_i} \exp(\mathbf{e}(w_i) \cdot \mathbf{e}(w_k) / \sqrt{d})}, \qquad f(w_i, \text{sentence}) = \sum_{w_j \in C_i} \text{weight}(w_i, w_j) \cdot \mathbf{e}(w_j)$$
 
The $\sqrt{d}$ in the denominator is a scaling term to keep the dot products from growing too large as dimensionality increases (recall Episode 00.04 — larger $d$ changes the typical *magnitude* of dot products between vectors; this scaling compensates for that). **This is the scaled dot-product attention formula**, in full, months before we've written a single line of PyTorch's actual attention module. We're not implementing it from raw derivation yet — that's Module 04's job, once gradients and learned projections are on the table — but the *shape* of the computation is already exactly right, and worth recognizing on sight from here on.
 
## 3. Code: uniform mixing, then relevance-weighted mixing, side by side
 
**3.1 From scratch — static vs. context-averaged "bank"**
 
```python
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
```
 
```
Static 'bank' vs 'money': 0.115
Static 'bank' vs 'river': 0.227
 
Contextual 'bank' (finance sentence) vs 'money': 0.943
Contextual 'bank' (finance sentence) vs 'river': 0.694
Contextual 'bank' (river sentence)   vs 'money': 0.696
Contextual 'bank' (river sentence)   vs 'river': 0.969
```
 
The static vector barely favors either sense (0.115 vs. 0.227 — genuinely ambiguous, as expected of an average over every use of the word). The moment we mix in the *actual* neighbors of a specific sentence, the same word "bank" produces two clearly different vectors — 0.943 toward "money" in one sentence, 0.969 toward "river" in the other — with zero training, using only arithmetic on embeddings we'd already built.
 
**3.2 Using a library — the same idea, but relevance-weighted (real scaled dot-product attention)**
 
```python
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
```
 
```
River sentence, attention weights for 'bank':
  sat        attention weight = 0.374
  river      attention weight = 0.626
```
 
This is the qualitative difference attention adds over plain averaging: uniform mixing (§3.1) would have given "sat" and "river" equal 50/50 weight regardless of relevance. Real (scaled dot-product) attention instead computes that "river" is more relevant to "bank" than "sat" is — via their raw dot-product similarity — and weights it higher, automatically, with no training involved yet; the weighting comes purely from the geometry of embeddings we already had. Run the same code on the finance sentence `"deposited money bank"` and you'll find the two context words tie at 0.5/0.5 — a useful edge case, not a bug: **when two context words are genuinely equally relevant, attention correctly reduces to uniform averaging.** Uniform averaging isn't a different mechanism from attention — it's attention's output in the special case where nothing distinguishes the candidates.
 
## 4. Where this leaves us
 
We've now built, from first principles and without training a single parameter, a working (if primitive) contextual embedding mechanism — and shown that it's structurally identical to the attention formula real transformers use, just with fixed, unlearned "weights" (the raw static embeddings themselves standing in for what will later be learned query/key/value projections). What's still missing, deliberately, going into Module 02 and beyond:
 
- The embeddings we used as queries and keys today are the *same* static embeddings from Episode 01.01 — real attention learns separate, trainable projections for queries, keys, and values, rather than reusing one fixed embedding for all three roles. That requires the gradient-descent machinery of Module 03.
- We only contextualized one word ("bank") in isolation. A real transformer layer contextualizes *every* word in a sentence simultaneously, each one attending to all the others — including to each other's freshly-contextualized representations, stacked in multiple layers.
- We hand-picked a window and averaged neighbors uniformly as a baseline; real models learn how far to look and what to weight, entirely from data.
## 5. Module 01 wrap, and what comes next
 
Module 01 built the full representation pipeline end to end: tokenization solves the symbol-level OOV problem (01.00); subword composition solves the meaning-level OOV problem (01.01); and today closes the module by solving — in primitive form — the context-sensitivity problem static embeddings could never solve on their own, landing us one recognizable step from real attention.
 
Module 02 — **Mathematical Foundations** — is next, and it's not a detour: we're about to need vectors, matrices, and calculus formalized properly (not just used informally, as we have been) to build gradient descent, backpropagation, and eventually the learned query/key/value projections that turn today's hand-built approximation into the real thing.
 
---
 
**Previous:** Episode 01.01 — Subword Embeddings in Practice
**Next:** Module 02, Episode 02.00 — Vectors, Spaces, and What "Dimension" Really Means