# From Zero to Agents
## Module 06 — Agents
### Episode 06.07: Closing the Loop — Why "More Training" Didn't Fully Fix the DOUBLE Tool
 
---
 
## 0. Where we're starting from
 
Episode 06.01 diagnosed the DOUBLE tool as undertrained. Episode 06.02–06.03 built and fixed the metric that confirmed exactly how badly. This episode does the obvious next thing — actually apply more training — and finds that the naive version of the fix doesn't fully work either, for a precise, diagnosable reason that goes one level deeper than "not enough data."
 
## 1. The straightforward fix, applied directly
 
**1.1 What changed.**
DOUBLE's training examples were oversampled from 5x repetition (Episode 06.01's original setup) to 20x — four times more training signal on the same underlying examples, with everything else (architecture, other tools' data, training steps) held constant.
 
**1.2 The result — partial improvement, not a fix**
 
```python
for d1 in dbl_vals:
    r = run_loop(model, [QDBL, d1, A, TDBL, d1], [lambda d1=d1: double_tool(d1)])
    pred = extract_final(r)
    print(f"  double({d1}): pred={pred} true={double_tool(d1)} ok={pred==double_tool(d1)}")
```
```
double(0): pred=0 true=0 ok=True  (train)
double(1): pred=2 true=2 ok=True  (train)
double(2): pred=4 true=4 ok=True  (train)
double(3): pred=6 true=6 ok=True  (train)
double(4): pred=6 true=8 ok=False (held-out)
```
 
Genuine, reportable progress — the held-out prediction moved from `4` (Episode 06.01's original wrong answer) to `6` (closer, but still wrong) — but not a fix. All four *training* values are now handled perfectly, and the one genuinely held-out value is still wrong.
 
## 2. Diagnosing why more repetition wasn't enough — a level deeper than "not enough data"
 
**2.1 The real problem, precisely.**
DOUBLE's entire usable input domain, in this toy setup, is five values: `{0, 1, 2, 3, 4}` (constrained to keep outputs single-digit). Training on four of them and holding out the fifth means the model has *never once seen the number 4 as an input* during training — not "seen it rarely," seen it **zero times.** Oversampling the same four training examples 20 times instead of 5 gives the model more practice on those four specific input-output pairs, but repetition of already-seen data adds no new information about a value that was never included at all. This is a fundamentally different problem than the original data-imbalance issue — that was about *quantity relative to other tools*; this is about **domain coverage relative to what's being tested**, and no amount of additional training on the same four points can be expected to reliably extrapolate to a genuinely unseen fifth one.
 
**2.2 Contrast this with why ADD and CHAIN generalize just fine.**
ADD draws from 25 possible pairs and trains on 20, leaving 5 genuinely unseen *combinations* — but every individual digit (0 through 4) still appears many times across different training pairs, giving the model ample signal about how each digit behaves in the addition rule generally. CHAIN is similar, at a larger scale (27 possible triples, 20 trained). DOUBLE's domain is simply too small, and the held-out split too aggressive relative to that size, for the same kind of generalization to be fair to expect — this isn't a flaw in DOUBLE specifically, it's a mismatch between domain size and evaluation methodology that this episode's own setup created.
 
## 3. The actual, correct fix — and an honest limitation of the toy setup itself
 
**3.1 What would genuinely fix this.**
Two real options, not oversampling: either expand DOUBLE's usable input domain substantially (the way ADD's 25-pair and CHAIN's 27-triple domains support genuine held-out testing), so a fair train/test split leaves enough signal in the training portion for real generalization to be learnable and measurable — or, honestly, accept that a tool with an intrinsically tiny input domain may not be a fair candidate for a held-out generalization test at all, and instead train on its full domain and evaluate purely on whether it's learned correctly (a train-accuracy check, not a generalization check) — a legitimate, different evaluation standard for a genuinely small-domain tool.
 
**3.2 Confirming everything else in the system still works.**
```python
print(f"ADD (unseen): {correct_add}/5")
print(f"CHAIN (unseen): {correct_chain}/5")
```
```
ADD (unseen): 5/5
CHAIN (unseen): 5/5
```
Both tools with adequately large domains continue to generalize perfectly under the exact same training regime — confirming this episode's diagnosis is specific to DOUBLE's domain size, not a symptom of some broader problem with the training setup, architecture, or masking approach.
 
## 4. Where this leaves the whole running thread
 
Three episodes tracked one problem to its actual root: Episode 06.01 found DOUBLE failing and initially attributed it to data quantity; Episode 06.02–06.03 built and fixed a metric confirming exactly how undertrained it was; this episode applied the straightforward fix (more training) and found it only partially worked, revealing a deeper, more precise cause — domain size relative to the held-out split, not repetition count. This is worth sitting with as the actual takeaway from the whole arc, more than any single episode's result: **a metric correctly flagging a problem tells you *that* something is wrong, not automatically *why* — and the first fix that seems obvious (more training) can itself expose a more precise diagnosis when it only partially works, exactly as happened here.**
 
## 5. Module 06 wrap-up
 
Eight episodes: the ReAct loop assembled entirely from existing course machinery (06.00); multi-tool selection and chaining, with an honest data-imbalance failure (06.01); two automatic signals for catching that failure, one needing its own fix (06.02–06.03); agent memory and the hard context ceiling, with retrieval inheriting Module 00's anisotropy problem (06.04); contextual embeddings tested and found to fail at toy scale (06.05); Reflexion tested against exactly the failure it cannot fix (06.06); and today, the original diagnosed failure revisited, partially fixed, and traced to its real, deeper cause. Every episode in this module — like every module before it — was built to be tested, not just described, and reported exactly as measured, including every time the result complicated rather than confirmed the expectation going in.
 
---
 
**Previous:** Episode 06.06 — Reflexion: Where In-Context Self-Correction Stops Working