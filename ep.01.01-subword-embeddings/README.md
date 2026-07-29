# From Zero to Agents
## Module 01 — Representing Language as Numbers
### Episode 01.01: Subword Embeddings in Practice
 
---
 
## 0. Closing the open question
 
Episode 01.00 ended by asking: does Module 00's word-embedding machinery just apply directly to subword tokens like `lowe` + `r` + `i` + `n` + `g`, or does something break?
 
The direct answer to the mechanical half: **yes, it applies directly, unchanged.** Every subword token produced by BPE gets its own row in an embedding table, exactly the way whole words did in Module 00 — nothing about "looking up a vector by ID" cares whether that ID represents a whole word or a fragment. That part isn't the interesting part.
 
The interesting part is what happens to **meaning** once "context" is redefined at the subword level — and it turns out to both cost something and unlock something genuinely new.
 
## 1. Theory: what changes when the unit is a fragment, not a word
 
**1.1 The cost — frequent fragments become semantically generic.**
Recall the distributional hypothesis from Episode 00.02: a token's meaning is inferred from the contexts it appears in. A whole word like "kingdom" appears in a fairly specific set of contexts. But a common suffix fragment like `ing` appears at the end of "running," "eating," "kingdom," "morning," "spring" — an enormous, wildly heterogeneous set of contexts, most of which share almost nothing semantically except the fragment itself. The distributional hypothesis still applies mechanically, but the "meaning" it converges on for a piece like `ing` is necessarily a blurry average over hundreds of unrelated use-cases — closer to a grammatical signal (this is probably a present participle or gerund ending) than a semantic one. This isn't a flaw exactly — it's the fragment doing the only job it realistically can — but it does mean subword embeddings for common fragments carry less specific meaning than Module 00's whole-word embeddings did.
 
**1.2 The unlock — composition enables genuine generalization to unseen words.**
This is the part worth the whole episode. If a word's meaning is represented as **whole**, an unseen word is a total loss — no vector, no fallback, nothing. But if a word's meaning can be built by **composing** the meanings of its known pieces, then a genuinely unseen word can still get a sensible vector, assembled from parts the model has actually seen before, even though the whole was never encountered. This is precisely the recovery of what tokenization (Episode 01.00) already bought us at the *symbol* level — no OOV tokens — now happening a level up, at the *meaning* level: no OOV embeddings either.
 
**1.3 fastText — making composition the actual training objective.**
This idea has a name and a real, widely-used implementation: **fastText**, introduced by Bojanowski, Grave, Joulin, and Mikolov in 2017 (the same Mikolov behind word2vec, extending his own earlier work). The mechanism: represent every word not as a single atomic vector, but as the **sum of the embeddings of its character n-grams** (overlapping fragments of length $n$, plus the whole word itself as one more "n-gram"). Training still uses a skip-gram-style predictive objective — but critically, the thing being trained is the *n-gram* embedding table, and a word's vector is only ever the sum of the n-grams that happen to compose it, recomputed on demand. This means a word fastText never saw during training still gets a real vector at inference time, for free, as long as it shares character n-grams with words that *were* seen — no special OOV-handling code required; generalization is simply what the architecture does by construction.
 
## 2. Math: n-grams, and vectors as sums
 
**2.1 Extracting character n-grams.**
For a word $w$, pad it with boundary markers (so `king` becomes `<king>`, letting the model distinguish a fragment at a word boundary from the same fragment mid-word), then take every contiguous substring of length $n$:
 
$$\text{ngrams}_n(w) = \{\, s \mid s \text{ is a contiguous substring of length } n \text{ of } \langle w \rangle \,\}$$
 
For `king` with $n=3$: `<ki`, `kin`, `ing`, `ng>`. fastText typically extracts n-grams across a *range* of lengths (e.g., $n = 3$ to $6$) and unions them together, plus the whole padded word itself as one final "n-gram," so both fine-grained fragments and the full word contribute.
 
**2.2 Composing the word vector.**
Given a learned n-gram embedding table $\mathbf{z}_g \in \mathbb{R}^d$ for every n-gram $g$ in the training vocabulary, the vector for word $w$ is simply:
 
$$\mathbf{v}_w = \sum_{g \,\in\, \text{ngrams}(w)} \mathbf{z}_g$$
 
Compare this directly to Episode 00.03's skip-gram score, $p(w_O \mid w_I) \propto \exp(\mathbf{v}'_{w_O} \cdot \mathbf{v}_{w_I})$ — fastText uses the exact same predictive skip-gram machinery, with one substitution: everywhere the original formulation used a single atomic vector $\mathbf{v}_{w_I}$ for the center word, fastText substitutes the *sum* $\sum_{g} \mathbf{z}_g$ instead. Training still adjusts vectors to make context prediction more accurate — but because a word's vector is now a sum over shared, reusable pieces, gradient updates for one word (Module 03 will make "gradient update" precise) also nudge the n-gram vectors that *other, unrelated words* happen to share — which is exactly the mechanism that lets never-seen words inherit sensible vectors afterward.
 
## 3. Code: building composition from scratch, then the real thing
 
**3.1 From scratch — n-gram co-occurrence embeddings, composed by summation**
 
We don't have gradient descent yet (that's Module 03), so instead of fastText's actual predictive training, we reuse the count-based machinery already built in Episodes 00.02–00.03 — co-occurrence counts, then SVD — applied one level down, at the **character n-gram** level instead of the whole-word level. Same tools, finer granularity.
 
```python
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
```
 
```
cos(king, queen)        = 0.702
cos(king, apple)        = 0.249
cos(kingly[OOV], king)  = 0.974
cos(kingly[OOV], queen) = 0.820
cos(kingly[OOV], apple) = 0.389
```
 
`"kingly"` never appears anywhere in the training corpus — not as a whole word, not as any substring of another training word. Its vector is built entirely by summing the embeddings of the fragments it shares with `<king>` (namely `<ki`, `kin`, `ing`) and whatever other n-grams it contains. And the result lands almost on top of `king` (0.974), meaningfully closer to `queen` than to `apple` — real compositional generalization, produced by nothing more exotic than "sum the pieces you recognize," with no training example of the word itself ever required.
 
**3.2 Using a library — fastText via `gensim`, doing this properly with real training**
 
```python
from gensim.models import FastText, Word2Vec
import random
 
# Reusing the same synthetic royals/commoners corpus generator from Episode 00.03
random.seed(42)
royals, royal_actions, royal_objects = ["king", "queen"], ["rules", "wears", "commands"], ["kingdom", "crown", "throne", "army"]
commoners, commoner_actions, commoner_objects = ["man", "woman"], ["eats", "drinks", "buys"], ["apple", "orange", "water", "bread"]
sentences = []
for _ in range(500):
    if random.random() < 0.5:
        s, a, o = random.choice(royals), random.choice(royal_actions), random.choice(royal_objects)
    else:
        s, a, o = random.choice(commoners), random.choice(commoner_actions), random.choice(commoner_objects)
    sentences.append([s, a, o])
 
ft_model = FastText(sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100, seed=42, min_n=2, max_n=4)
w2v_model = Word2Vec(sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100, seed=42)
 
oov_word = "kingly"
print("'kingly' in FastText's known-word list?", oov_word in ft_model.wv.key_to_index)  # False
print(ft_model.wv.most_similar(oov_word, topn=3))
```
 
```
'kingly' in FastText's known-word list? False
[('kingdom', 0.9995), ('king', 0.9994), ('queen', 0.9994)]
```
 
```python
w2v_model.wv[oov_word]  # raises KeyError: "Key 'kingly' not present"
```
 
This is the cleanest possible demonstration of the whole episode: fastText was never trained on the word "kingly" — its `key_to_index` confirms that word literally isn't in its known-word list — and yet asking it for the word's nearest neighbors returns "king" and "queen" at the top, computed live from n-gram composition. The plain word2vec model, trained on the identical corpus, has no such fallback and simply raises a `KeyError`. Same corpus, same training regime, one architectural difference — compositional vs. atomic word vectors — and the practical difference is total failure versus graceful generalization.
 
## 4. Where this leaves us
 
Subword tokenization (01.00) guarantees no symbol is ever truly unrepresentable. Subword-composed embeddings (today) push that same guarantee up to the level of meaning: no word need be truly unrepresentable either, as long as it shares recognizable pieces with something the model has seen. Both are instances of the same underlying idea, showing up twice at two different layers of the pipeline — decompose the unfamiliar into familiar, recombinable parts.
 
One limitation is still standing, and it's the same one flagged at the end of Episode 00.03, now sharper: **every embedding we've built across this entire course so far — word-level or subword-level — is still exactly one fixed vector per token, regardless of the sentence it's actually sitting in.** "Bank" gets the same vector next to "river" as it does next to "loan." Composition solved *unseen words*. It did nothing for *ambiguous* ones.
 
## 5. Before Episode 01.02
 
> Every technique so far — one-hot, co-occurrence, SVD, word2vec, fastText — assigns a token's vector by looking at how it's used *on average, across the whole training corpus*. But a specific sentence isn't the average case — it's one particular context. Sit with this: what would it take for a word's vector to be computed *fresh, for this specific sentence*, using the actual neighboring words present right now, rather than a single fixed vector baked in ahead of time from corpus-wide statistics? What piece of machinery would need to exist to let nearby words in a sentence directly influence each other's representation, on the fly?
 
That question is the direct on-ramp into contextual embeddings, and from there, attention.
 
---
 
**Previous:** Episode 01.00 — Tokenization: Why "Words" Aren't the Right Unit
**Next:** Episode 01.02 — Toward Contextual Representations