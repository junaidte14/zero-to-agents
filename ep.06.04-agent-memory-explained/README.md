# From Zero to Agents
## Module 06 — Agents
### Episode 06.04: Agent Memory — the Context Ceiling, and Retrieval as a Fix
 
---
 
## 0. Where we're starting from
 
Every agent built so far in this module kept its entire history in one growing text context, fed whole into the transformer at every step. Real agent sessions — long conversations, long multi-step tasks — eventually outgrow whatever context length a model was built and trained for. This episode demonstrates that ceiling directly, then builds the standard fix: retrieval-based memory, which turns out to run directly into a problem this course flagged five modules ago.
 
## 1. Theory: two different kinds of "running out of room"
 
**1.1 The hard ceiling — positional encoding has a fixed table.**
Episode 04.04 built sinusoidal positional encoding and noted, as an advantage, that the formula can technically be evaluated at any position — no fixed maximum built into the math itself. But a real implementation (including ours, and every production transformer) still precomputes a positional encoding table up to some maximum length, purely for efficiency — and the model's $W_Q, W_K, W_V$ weights were only ever *trained* on attention patterns within that length. The formula being defined beyond the table doesn't mean the trained weights behave sensibly there; in practice, most implementations simply don't allow it, exactly the failure Section 4 reproduces directly.
 
**1.2 The practical fix — don't keep everything, retrieve what's relevant.**
Rather than truncating the oldest context blindly (losing information that might still matter) or trying to fit an ever-growing conversation into a fixed window, the standard approach is **retrieval-based memory**: store the full history externally, and at each step, retrieve only the most *relevant* pieces to include in the current context — precisely the "compute relevance, select what matters" idea attention performs internally (Episode 02.01), just operating over much coarser units (whole past turns, not individual tokens) and via embedding similarity search rather than a trained softmax.
 
**1.3 This is retrieval-augmented generation, applied to the agent's own history.**
This is the exact mechanism behind retrieval-augmented generation (RAG) — usually discussed in the context of retrieving external documents, but structurally identical when the "document store" is simply the agent's own past turns instead. And because it's built on embedding similarity, it inherits every embedding-quality issue this course has already covered — including one flagged all the way back in Module 00.
 
## 2. Code: the ceiling, demonstrated directly
 
```python
print("Model's trained max_len:", max_len)          # 24
too_long_seq = torch.randint(0, 10, (1, max_len + 5))
out = model(too_long_seq)
```
```
Model's trained max_len: 24
RuntimeError: The size of tensor a (29) must match the size of tensor b (24) at non-singleton dimension 1
```
 
Not graceful degradation — a hard, immediate error, because the positional encoding table genuinely only has 24 rows. This is the concrete version of §1.1's claim: the ceiling is real and immediate, however elegant the underlying formula is in the abstract.
 
## 3. Code: retrieval-based memory, and the anisotropy problem showing up right on schedule
 
**3.1 A toy agent memory store, embedded with Module 00's own machinery**
 
Five stored past turns, embedded using exactly Episode 00.02–00.03's co-occurrence-plus-composition approach:
 
```python
past_turns = [
    "user asked about refund policy for damaged items",
    "user asked how to reset their account password",
    "user asked about shipping times to canada",
    "user asked to cancel a subscription plan",
    "user mentioned their favorite color is blue",
]
new_query = "user wants to know about shipping times"
```
 
**3.2 Retrieval without any adjustment**
 
```python
qv = turn_vector(new_query)
for t, s in sorted([(t, cosine(qv, turn_vector(t))) for t in past_turns], key=lambda x: -x[1]):
    print(f"  sim={s:.3f}  {t}")
```
```
sim=0.972  user asked about shipping times to canada      <-- correctly ranked #1
sim=0.893  user asked to cancel a subscription plan
sim=0.867  user asked how to reset their account password
sim=0.838  user asked about refund policy for damaged items
sim=0.596  user mentioned their favorite color is blue
```
 
The correct turn wins, technically — but look at the **margin**: `0.972` versus `0.893, 0.867, 0.838` — every candidate is crowded into a narrow band near the top, regardless of actual topical relevance. This is Episode 00.04's anisotropy finding, showing up exactly where it would matter in practice: every one of these sentences shares the common structural pattern `"user asked ... to/about"`, and that shared, high-frequency scaffolding inflates similarity across *every* pair, compressing the genuinely meaningful signal into a small residual gap near the top of an otherwise tightly-bunched ranking.
 
**3.3 The fix Episode 00.04 didn't get to apply, applied now**
 
```python
stopwords = {"user","asked","their","to","about","wants","know"}
qv_f = turn_vector(new_query, stopwords)
for t, s in sorted([(t, cosine(qv_f, turn_vector(t, stopwords))) for t in past_turns], key=lambda x: -x[1]):
    print(f"  sim={s:.3f}  {t}")
```
```
sim=0.968  user asked about shipping times to canada      <-- still #1, now with real separation
sim=0.476  user asked to cancel a subscription plan
sim=0.420  user asked how to reset their account password
sim=0.368  user asked about refund policy for damaged items
sim=0.128  user mentioned their favorite color is blue
```
 
Stripping the shared structural words before embedding doesn't change *which* turn wins here — it changes *how reliably* it wins. The gap between the correct match and its nearest competitor goes from `0.972 − 0.893 = 0.079` to `0.968 − 0.476 = 0.492` — a roughly 6x wider margin, meaning this retrieval decision is now robust to noise, ties, or a slightly different query, rather than one lucky ranking away from picking the wrong memory. This is the exact mechanism Module 01's authority article on RAG retrieval quality described in the abstract, now demonstrated directly, in this course's own retrieval code, using nothing but tools this course already built.
 
## 4. Where this leaves us
 
Two genuinely different limits, both real: a transformer's context window is a hard architectural ceiling with no graceful fallback, and retrieval-based memory built to work around it inherits the exact embedding-quality risks this course flagged five modules earlier — not a coincidence, a direct structural consequence of retrieval being built on the same cosine-similarity machinery from Episode 00.02 onward. The fix (strip shared structural content before embedding) isn't new either — it's the same idea behind removing common words before computing similarity that's been implicit in this course's embedding work since Module 00, now applied deliberately, with a measured, quantified improvement in retrieval reliability rather than an assumed one.
 
## 5. Before the next episode
 
> Retrieval in this episode used the same static, co-occurrence-based embeddings from early in the course — the exact kind Episode 01.02 showed can't distinguish a word's meaning by context. What do you think happens to retrieval quality if the stored memory (or the query) contains genuinely ambiguous language — the same "bank" problem from Module 01, but now for whole retrieved conversation turns rather than single words? Would a *contextual* embedding (Episode 01.02's mechanism, or a real transformer-based embedding model) meaningfully improve agent memory retrieval, and is that improvement worth its extra computational cost compared to the simple approach used here?
 
That's worth carrying into the next episode of Module 06.
 
---
 
**Previous:** Episode 06.03 — Fixing the Metric: Comparable Positions, Not Comparable Sequences
**Next:** Episode 06.05 — Contextual Memory Tested