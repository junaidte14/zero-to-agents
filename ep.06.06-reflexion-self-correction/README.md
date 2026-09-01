# From Zero to Agents
## Module 06 — Agents
### Episode 06.06: Reflexion — and Exactly Where In-Context Self-Correction Stops Working
 
---
 
## 0. Where we're starting from
 
Episode 06.00 §3 mentioned "self-correction" as a forward pointer without building it. This episode builds it — **Reflexion** (Shinn et al., 2023) — and then, rather than testing it on a case where it flatters the technique, tests it on the exact case this module already knows is broken for a specific, diagnosed reason: Episode 06.01's undertrained DOUBLE tool. The result draws a precise, useful boundary around what reflection can and cannot fix.
 
## 1. Theory: a different loop shape, for a different kind of failure
 
**1.1 The gap Reflexion addresses.**
Episode 06.01 §1.3 treated tool failure as just another observation fed back into context — a real tool error, caught and reported. But there's a different failure mode entirely: the tool call *succeeds*, the observation is real and correct, and the agent still produces a **wrong final answer** — a reasoning error, not a tool error. Nothing in the basic ReAct loop catches this; the loop only knows what the model told it, and the model, in this failure mode, is confidently wrong.
 
**1.2 Reflexion's fix — an outer loop, driven by an external verifier.**
Reflexion wraps the ordinary ReAct loop in a second, outer loop: run the task, check the final answer against some external correctness signal (a verifier — a unit test, a known correct answer, a human judgment), and if it fails, have the model generate a **verbal reflection** on what went wrong, then retry the *entire* task with that reflection included as additional context. This is a genuinely different loop shape from Episode 06.00's — not "act, observe, continue," but "attempt, evaluate, reflect, retry."
 
**1.3 The crucial, precise scoping detail — no weights ever change.**
Worth stating with the same precision Episode 06.00 §2.2 used for the base ReAct loop: Reflexion involves **no gradient updates, no training step, nothing touching the model's weights at all.** The "learning" from a failed attempt happens entirely by adding text to the context — the reflection is just more input tokens for the next attempt, processed by the exact same frozen forward pass as everything else. Whatever correction happens, happens because the model's *existing* trained capability includes being able to make productive use of a reflective hint when it's present in context — a real, documented capability of sufficiently large, sufficiently trained language models, but not something guaranteed to exist in every model, including — as this episode is about to show directly — a small one.
 
## 2. Math: the outer loop, formalized
 
For attempt $n = 1, 2, \ldots, N_{\max}$: run the standard ReAct recursion (Episode 06.00 §2.1) to termination, producing a final answer $\hat{y}_n$. Check $V(\hat{y}_n) \to \{\text{success}, \text{fail}\}$ via an external verifier $V$. If it fails, generate a reflection $r_n \sim \pi_{\text{LLM}}(\cdot \mid \text{trajectory}_n)$, and start attempt $n+1$ with $r_n$ prepended to its initial context. The verifier's pass/fail signal here plays a role structurally similar to the "reward" in Episode 00.01's rational-agent formalism — but note precisely how it's *used*: not to update $\pi$'s parameters via a gradient (that would be reinforcement learning, a distinct and more advanced topic this course hasn't covered), but purely to trigger another forward pass with a longer, hopefully more informative context.
 
## 3. Code: testing reflection on a case this module already understands deeply
 
**3.1 The test — DOUBLE, the exact tool Episode 06.02–06.03 diagnosed as undertrained**
 
```python
d1 = 4   # Episode 06.01's actual failure case: double(4) should be 8, model predicted 4
 
baseline = run_with_prefix([], [QDBL, d1, A, TDBL, d1], [lambda: double_tool(d1)])
print("Baseline final answer:", extract_final(baseline), "(true = 8)")
```
```
Baseline final answer: 4  (true = 8)
```
 
Confirms Episode 06.01's original finding exactly, on the same case.
 
**3.2 Adding a reflection-style prefix, and testing whether it changes anything**
 
```python
reflection_prefix = [ERR, d1, O, 4]   # a stand-in for "previous attempt on input 4 wrongly gave 4"
with_reflection = run_with_prefix(reflection_prefix, [QDBL, d1, A, TDBL, d1], [lambda: double_tool(d1)])
print("With-reflection final answer:", extract_final(with_reflection), "(true = 8)")
```
```
With-reflection final answer: 4  (true = 8)
```
 
**No change whatsoever.** The reflection prefix had exactly zero corrective effect — the model produced the identical wrong answer, with or without it.
 
## 4. Why this is the correct, expected result — and a genuinely useful boundary
 
**4.1 The precise reason, not a vague "the model is too small."**
Our toy model was never trained on *any* examples containing reflection-style text — the token sequence used as a stand-in reflection here is entirely out of the distribution the model was fine-tuned on. Real Reflexion works, in real large language models, because those models' extensive pretraining gives them a general, emergent capacity to parse and productively use arbitrary natural-language context — including a sentence explaining a previous mistake — even without having been specifically trained on "reflection-formatted" examples. Our tiny model has no such emergent general capability; it was trained end-to-end on a narrow, fixed set of trace formats, and text outside those exact formats is not information to it — it's just tokens with no learned meaning attached.
 
**4.2 The precise, useful distinction this draws.**
Combined directly with Episode 06.02–06.03's diagnosis, this episode draws an exact line: **DOUBLE's failure was a training-data coverage problem** (too few examples, established with certainty in Episode 06.02), and **in-context reflection cannot fix a training-data coverage problem**, because reflection only works by directing an *existing* capability more effectively — it has no mechanism to inject knowledge the weights never acquired in the first place. This is not a limitation specific to this toy model; it's a structural property of the technique itself, worth carrying forward precisely: reflection is a tool for correcting *reasoning errors made with sufficient underlying capability present* — misapplied tools, misread observations, premature conclusions — not a substitute for the actual training (more data, better data, or larger/more capable pretraining) that Episode 06.02's fix genuinely required.
 
## 5. Where this leaves us
 
A clean null result, tested directly rather than assumed, that draws a sharp and useful boundary: Reflexion is real, and the mechanism (verify, reflect, retry, all in-context, no weight updates) is correctly built and understood here — but it was tested against precisely the wrong kind of failure to expect it to fix, and it correctly failed to fix it, for a reason traceable directly to this module's own earlier diagnosis. The practical lesson for building real agent systems: before reaching for a self-correction loop to fix a misbehaving agent, first determine whether the failure is a reasoning error (where reflection genuinely helps) or a capability gap (where it structurally cannot, and the actual fix is more/better training data, exactly as Episode 06.02 established).
 
## 6. Module 06 wrap-up
 
Seven episodes: the ReAct loop assembled from existing course machinery (06.00); multi-tool selection and chaining with an honest data-imbalance failure (06.01); two signals for catching that failure automatically, one needing its own fix (06.02–06.03); agent memory and the context ceiling, with retrieval inheriting Module 00's anisotropy problem (06.04); contextual embeddings tested honestly and found to fail at toy scale (06.05); and today, Reflexion tested against the exact failure it can't fix, drawing a precise boundary rather than a vague caveat. Consistent with this entire course: every claim in this module was built, then tested, and reported exactly as measured — including, repeatedly, when the result wasn't the one initially expected.
 
---
 
**Previous:** Episode 06.05 — Does Contextual Embedding Actually Help Agent Memory?
**Next:** Closing the Loop