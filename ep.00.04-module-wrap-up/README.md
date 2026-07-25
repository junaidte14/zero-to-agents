# From Zero to Agents
## Module 00 — Introduction
### Episode 00.04: Module Wrap-Up — Why Nothing Is Ever Truly "Unrelated" in Embedding Space

---

## 0. Closing the open question

Episode 00.03 ended with `similarity("king", "apple") = 0.74` — high, even though apple is the clearly unrelated word in that toy vocabulary — and asked why. There are two real reasons stacked on top of each other, and separating them is the last piece needed to close out Module 00 cleanly.

## 1. Theory: two different reasons, easy to conflate

**1.1 Reason one — dimensionality itself, nothing to do with meaning at all.**
Go back to Episode 00.02 §2: one-hot vectors in a $V$-dimensional space are *forced* to be exactly orthogonal (cosine similarity of exactly 0), because $V$ is enormous and each word occupies its own dedicated axis. Dense embeddings deliberately abandon that — we compressed down to 2 or 20 dimensions specifically because we wanted words to *share* structure. But sharing structure is a double-edged sword: in a low-dimensional space, there simply aren't enough independent directions for every pair of unrelated words to end up perpendicular. Two *completely random, meaningless* vectors in 2 or 20 dimensions will, on average, already have a fairly high cosine similarity just from geometry — before any actual semantic relationship is involved at all. High dimensionality is precisely what makes near-orthogonality possible; we gave that up on purpose when we compressed, and this is the cost.

**1.2 Reason two — real embeddings aren't randomly distributed; they're anisotropic.**
Even accounting for reason one, real trained embeddings (word2vec, GloVe, and — as later research found — the internal representations inside transformers too) don't spread out evenly through their available space. They tend to bunch into a narrow cone, all pointing in roughly similar directions, rather than filling the space uniformly. This phenomenon is called **anisotropy**, and it's well documented in the literature — see Mu & Viswanath's *"All-but-the-Top"* (2018) and Ethayarajh's *"How Contextual are Contextualized Word Representations?"* (2019). One major driver: every word, no matter its meaning, tends to co-occur with a handful of extremely frequent function words ("the," "a," "of," "eats," in our toy example) — and that *shared* frequency signal pushes every embedding a little bit in the same direction, on top of whatever direction its actual meaning points toward. The result: even two genuinely unrelated words share a "floor" of similarity coming from that common component.

**1.3 Why this matters practically, not just theoretically.**
This isn't a curiosity — it's the reason raw cosine similarity between embeddings is often a noisier signal than people assume, and why real systems frequently apply corrections (removing the top principal component, or other normalization tricks) before trusting similarity scores. Filing this away now saves confusion later: when you eventually query embedding similarity in a production RAG or agentic system at AIVerse and get a "surprisingly high" score for two unrelated pieces of text, this is very often why — not a bug, a structural property of the geometry itself.

## 2. Math: quantifying reason one directly

Take two vectors drawn independently and uniformly at random from the unit sphere in $\mathbb{R}^d$ (i.e., completely meaningless vectors — no shared structure, no anisotropy, nothing but dimensionality). The expected value of their *absolute* cosine similarity has a known closed form:

$$\mathbb{E}[\,|\cos(\mathbf{u}, \mathbf{v})|\,] \approx \sqrt{\frac{2}{\pi d}}$$

Read the shape of this, not just the formula: as $d$ (the number of dimensions) grows, the expected similarity shrinks proportionally to $\frac{1}{\sqrt{d}}$. Small $d$ → high baseline similarity, purely from geometry. Large $d$ → vectors are almost forced to be nearly perpendicular, purely from geometry, which is exactly the "curse of dimensionality" running in *reverse* — usually framed as a curse for search and distance-based algorithms, here it's actually a **blessing** for representation: high dimensionality is what makes it *possible* for a large vocabulary to have every pair be genuinely, independently positioned relative to each other, the way one-hot's $V$ dimensions guaranteed (at the cost of zero shared structure) back in Episode 00.02.

## 3. Code: watching the formula hold up

```python
import numpy as np

def avg_abs_cosine(d, n_trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    v1 = rng.normal(size=(n_trials, d))
    v2 = rng.normal(size=(n_trials, d))
    v1 /= np.linalg.norm(v1, axis=1, keepdims=True)
    v2 /= np.linalg.norm(v2, axis=1, keepdims=True)
    cos = np.sum(v1 * v2, axis=1)
    return np.mean(np.abs(cos))

for d in [2, 5, 20, 50, 100, 300, 1000]:
    empirical = avg_abs_cosine(d)
    theoretical = np.sqrt(2 / (np.pi * d))
    print(f"d={d:5d}  empirical={empirical:.4f}   theory={theoretical:.4f}")
```

```
d=    2  empirical=0.6344   theory=0.5642
d=    5  empirical=0.3747   theory=0.3568
d=   20  empirical=0.1828   theory=0.1784
d=   50  empirical=0.1139   theory=0.1128
d=  100  empirical=0.0808   theory=0.0798
d=  300  empirical=0.0459   theory=0.0461
d= 1000  empirical=0.0255   theory=0.0252
```

At $d=2$ — the exact dimensionality we used for the SVD embeddings in Episode 00.03 — two **completely random, meaningless** vectors already average an absolute cosine similarity of ~0.63. Against that baseline, king–apple's 0.74 was doing a lot less semantic work than it looked like. At $d=300$ (a realistic real-world embedding size), that same random baseline drops to ~0.05 — meaning any similarity meaningfully above that is much more likely to be capturing genuine relational structure rather than a geometric artifact. This is exactly why production embedding models use hundreds of dimensions rather than the toy 2–20 we used for teaching clarity this module.

## 4. Module 00 wrap-up

Four episodes, one throughline: we started by defining intelligence as *effective, efficient generalization under uncertainty* (00.01) — and everything since has been building the substrate that generalization actually runs on top of. Text can't be reasoned over numerically until it's numbers (00.02); the naive numeric encoding (one-hot) is structurally incapable of expressing relatedness; the distributional hypothesis fixes that by inferring meaning from context; and today's episode showed that the fix has its own cost, one worth understanding precisely rather than being surprised by later.

Concepts now in your toolkit, load-bearing for everything ahead:
- The rational-agent formalism (§2 of 00.01) — the skeleton every agentic system in this course will eventually be built on.
- The distributional hypothesis — "you shall know a word by the company it keeps" — the seed of every representation technique from here to the transformer.
- One-hot vs. co-occurrence vs. dense embeddings, and the concrete mathematical reason each one behaves the way it does.
- Two families of dense embedding (count-then-compress vs. predict-based), and why the field converged on the predictive family.
- Dimensionality's direct, quantifiable effect on similarity — a fact that will resurface the moment you're debugging a real vector-search system.

## 5. What Module 01 covers

Module 00 deliberately dodged one piece the entire time: we split text into whole words with `.split()`, four episodes running, and never asked whether "words" are actually the right unit. Module 01 — **Representing Language as Numbers** — starts exactly there:

- **Ep. 01.00 — Tokenization**: why whole-word splitting breaks (misspellings, rare words, agglutinative languages), and how Byte-Pair Encoding (BPE) — the actual scheme behind GPT-family tokenizers — builds a vocabulary from data instead of a dictionary.
- **Ep. 01.01 — Subword embeddings in practice**: connecting tokenization back to the embedding machinery from Module 00, now built on subword units instead of whole words.
- **Ep. 01.02 onward**: the fixed-vector-per-word limitation we've flagged twice now (Episodes 00.03 and again implicitly here) finally gets addressed — the on-ramp into *contextual* representations, which is where the road to attention and transformers actually begins.

Module 02 (mathematical foundations — vectors, matrices, calculus, probability, formalized properly) will run partly in parallel, since Module 01 is about to lean on linear algebra harder than we've needed so far.

---

**Previous:** Episode 00.03 — Dense Embeddings: Compressing Meaning Into Fewer Dimensions
**Next:** Module 01, Episode 01.00 — Tokenization: Why Words Aren't the Right Unit