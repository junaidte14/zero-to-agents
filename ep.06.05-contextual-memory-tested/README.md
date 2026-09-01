# From Zero to Agents
## Module 06 — Agents
### Episode 06.05: Does Contextual Embedding Actually Help Agent Memory? Testing It Honestly
 
---
 
## 0. Closing the open question
 
Episode 06.04 ended by asking whether contextual embeddings (Episode 01.02's mechanism) would meaningfully improve agent memory retrieval over the static, co-occurrence-based approach used there — and whether the improvement would be worth the extra cost. This episode tests it directly, on a genuinely ambiguous case, and finds an answer more useful than a simple "yes."
 
## 1. Theory: the disambiguation case retrieval memory should be vulnerable to
 
**1.1 A genuinely ambiguous scenario.**
Store two past agent turns that both use the word "bank" in different senses — one about opening a financial bank account, one about hiking along a river bank. A static embedding (Episode 00.02 onward) gives "bank" exactly one fixed vector, averaged across every context it appeared in during training — meaning it structurally *cannot* distinguish which stored turn's "bank" matches a new query's intended sense, only how much surface vocabulary the query happens to share with each stored turn.
 
**1.2 What contextualization should, in principle, fix.**
Episode 01.02 built exactly the mechanism that should help here: compute a word's representation using its own local sentence context, so "bank" in the financial turn and "bank" in the hiking turn end up as genuinely different vectors, each reflecting its own specific sense — in principle, letting a query about one sense correctly avoid matching a stored turn using the other.
 
## 2. Code: testing it, and finding out it isn't that simple
 
**2.1 The setup, and a first honest result — static retrieval, weakened but not broken**
 
```python
past_turns = [
    "user asked how to open a new bank account for savings",
    "user asked about hiking trails along the river bank",
    "user asked about shipping times to canada",
    "user asked to cancel a subscription plan",
]
query = "user needs help with their bank"   # shares ONLY "bank" as real content vocabulary
```
 
```python
qv = static_turn_vec(query, stopwords)
for t, s in sorted([(t, cosine(qv, static_turn_vec(t, stopwords))) for t in past_turns], key=lambda x: -x[1]):
    print(f"  sim={s:.3f}  {t}")
```
```
sim=0.823  user asked how to open a new bank account for savings   <-- correct, but barely
sim=0.722  user asked about hiking trails along the river bank
sim=0.306  user asked to cancel a subscription plan
sim=0.147  user asked about shipping times to canada
```
 
Static retrieval still picks the right turn, but with a margin of only `0.101` between the correct match and its wrong-sense competitor — a genuinely fragile ranking, exactly the vulnerability §1.1 predicted, even though it happens to land correctly this time.
 
**2.2 Contextual retrieval — and a real, honest surprise**
 
Building per-turn "bank" vectors using Episode 01.02's own neighbor-averaging mechanism, computed fresh from each turn's specific local context:
 
```python
qv_ctx = contextual_turn_vec(query, stopwords)
for t, s in sorted([(t, cosine(qv_ctx, contextual_turn_vec(t, stopwords))) for t in past_turns], key=lambda x: -x[1]):
    print(f"  sim={s:.3f}  {t}")
```
```
sim=0.663  user asked about shipping times to canada        <-- WRONG, and unrelated
sim=0.589  user asked to cancel a subscription plan
sim=0.586  user asked about hiking trails along the river bank
sim=0.537  user asked how to open a new bank account for savings   <-- the CORRECT turn, ranked LAST
```
 
```
Margin, STATIC:     0.101
Margin, CONTEXTUAL: 0.074 -- and pointing at the WRONG answer entirely
```
 
This is not the result the theory in §1.2 predicted, and it's worth reporting exactly as it happened rather than adjusted to fit expectations: **the contextual version performed worse than the static baseline, not better** — it didn't just fail to fix the ambiguity, it actively ranked the correct turn last.
 
## 3. Diagnosing the real cause — and why it's a genuinely useful finding
 
**3.1 The honest explanation.**
Episode 01.02's context-mixing mechanism worked because it was demonstrated on a purpose-built corpus of ten sentences, carefully constructed with enough repeated structure for meaningful co-occurrence patterns to emerge around the ambiguous word. This episode's memory store has **four sentences total.** Neighbor-averaging computed from a handful of words' worth of context, drawn from a corpus this sparse, has nowhere near enough signal to reliably capture genuine sense distinctions — it's mostly amplifying noise, and the result (a completely unrelated turn about shipping ranking first) shows exactly that: the "contextualization" wasn't disambiguating "bank," it was injecting essentially random perturbation from an under-informed neighbor average.
 
**3.2 The real, practical lesson — and why production systems never build this from scratch.**
This is precisely why real retrieval-augmented generation and agent-memory systems use **pretrained** contextual embedding models — sentence-transformer-style models built on transformer encoders (Episode 04.02's real learned attention, not a hand-rolled neighbor average) and trained on enormous corpora — rather than anyone building a bespoke contextualizer from their own small, task-specific dataset. The theoretical benefit of contextualization (§1.2) is real, but it's a benefit that requires substantial pretraining data to actually realize; attempted on too little data, it can make things measurably worse than the simpler static baseline, exactly as demonstrated here. **This episode's own failure is the clearest possible argument for why nobody should roll their own toy contextual embedding model for a production memory system — not a caveat to the theory, a direct, measured demonstration of when it breaks.**
 
## 4. Where this leaves us
 
Two consecutive episodes' worth of honest testing — Episode 06.04 found static retrieval works but with a fragile margin; this episode found that naively "fixing" it with an under-resourced contextual approach makes things actively worse. Combined, the practical guidance is precise rather than a vague "use better embeddings": static, cheap embeddings are a reasonable default for retrieval when queries and stored content share real vocabulary; genuinely ambiguous cases need contextualization, but only a properly pretrained contextual model — not a small, hand-built one — reliably delivers the benefit the theory predicts.
 
## 5. Module 06 checkpoint
 
Six episodes into Module 06: the ReAct loop built from existing course machinery (06.00), multi-tool selection and chaining with an honest data-imbalance failure (06.01), two signals for catching that failure automatically, one of which needed its own fix (06.02–06.03), and today, two back-to-back honestly-tested claims about agent memory — one confirming a real vulnerability, one confirming that the "obvious" fix doesn't work at toy scale. Every episode in this module, like every module before it, prioritized measuring over asserting.
 
---
 
**Previous:** Episode 06.04 — Agent Memory: the Context Ceiling, and Retrieval as a Fix
**Next:** Episode 06.06 — Reflexion Self Correction