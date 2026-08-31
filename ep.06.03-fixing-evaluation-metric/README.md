# From Zero to Agents
## Module 06 — Agents
### Episode 06.03: Fixing the Metric — Comparable Positions, Not Comparable Sequences
 
---
 
## 0. Closing the open question
 
Episode 06.02 ended with two disagreeing signals and an open question: how do you decide which to trust, or combine them? The fix turns out to be more specific and more satisfying than picking a favorite — the held-out loss metric itself was measuring the wrong thing, and once that's corrected, both signals agree cleanly.
 
## 1. Theory: the bug wasn't the metric's math, it was its scope
 
**1.1 Diagnosing precisely what went wrong.**
Episode 06.02 §3.3 found that a uniform `[:4]` masking cutoff, applied across trace types of meaningfully different length (10, 12, and 19 tokens), left different *amounts* of genuinely-comparable content inside each type's aggregate loss. The fix isn't a bigger evaluation set or a cleverer masking heuristic tuned per trace type — it's recognizing that the *only* position genuinely comparable across all three task types is the one place every trace type has in common: **the single digit immediately following the final-answer marker.** Everything else — how many tool calls happened, how long the intermediate reasoning was — differs structurally between tasks by design, and averaging loss across those structurally different amounts of content was never a fair comparison to begin with.
 
**1.2 A general principle worth keeping.**
When comparing a metric across categories that differ in some structural way unrelated to what you're actually trying to measure (sequence length, number of steps, format complexity), the fix is almost always to **narrow the evaluation to the smallest unit that's genuinely equivalent across every category**, not to find a way to fairly weight the unequal parts. This is a direct generalization of Episode 05.07's finding — that episode fixed the problem by choosing "response-only" over "full-sequence"; this one goes one level further, to "the one response token that's actually the same kind of thing everywhere."
 
## 2. Code: the corrected metric, confirmed
 
```python
def final_answer_loss(traces_maker, examples):
    traces = [traces_maker(*ex) if isinstance(ex, tuple) else traces_maker(ex) for ex in examples]
    losses = []
    for t in traces:
        f_idx = t.index(Fi)
        seq = torch.tensor([t])
        inp, tgt = seq[:, :-1], seq[:, 1:]
        masked_tgt = torch.full_like(tgt, -100)
        masked_tgt[0, f_idx] = tgt[0, f_idx]   # ONLY the final-answer digit -- the one truly comparable position
        with torch.no_grad():
            loss = F.cross_entropy(model(inp).reshape(-1, vocab_size), masked_tgt.reshape(-1), ignore_index=-100)
        losses.append(loss.item())
    return sum(losses) / len(losses)
 
print(f"ADD:    {final_answer_loss(trace_add, test_add):.4f}")
print(f"DOUBLE: {final_answer_loss(trace_double, test_dbl):.4f}")
print(f"CHAIN:  {final_answer_loss(trace_chain, test_chain):.4f}")
```
```
ADD:    0.0025
DOUBLE: 0.7402
CHAIN:  0.0003
```
 
This is the clean result Episode 06.02 was reaching for. CHAIN now correctly ranks as the *best*-performing tool (`0.0003`, lower even than ADD), matching its perfect 5/5 generation accuracy from Episode 06.01 — no longer artificially inflated by unmasked positions that were never comparable to begin with. DOUBLE stands out unambiguously (`0.7402`, roughly 300x ADD's and over 2000x CHAIN's) — now agreeing cleanly with Episode 06.02's generation-confidence signal, which never had this scope problem in the first place (it only ever measured tokens the model actually committed to during generation, never an aggregate over a variable-length teacher-forced sequence).
 
## 3. A principled way to combine signals, now that they agree
 
**3.1 Why "average the two numbers" is still the wrong move, even now that they agree.**
Held-out loss and generation confidence are on different scales, measure subtly different things (one needs labels and reflects true correctness; the other needs no labels and only reflects the model's own certainty, which can occasionally be wrong even when confident), and shouldn't be blended into one number by naive averaging.
 
**3.2 A defensible combination rule — outlier detection, not averaging.**
The practical, principled approach: treat each signal as independent evidence, and flag a tool for review if **either** signal is a clear outlier relative to the others — for instance, more than some multiple of the median across all tools being evaluated. Applied here: DOUBLE's held-out loss (`0.74`) is roughly 300x the median of the other two; its generation confidence (`0.176`) is roughly 100x the others' median. Both independently clear an outlier threshold by a wide margin — agreement between two differently-scoped, differently-derived signals is much stronger evidence than either alone, and neither needed to be numerically combined for that agreement to be actionable.
 
## 4. Where this leaves us
 
The lesson from Episodes 06.02–06.03, taken together, is more valuable than either the original failure or the fix in isolation: a metric that looked broken wasn't measuring the wrong *thing*, it was measuring the wrong *scope* — and the diagnosis (trace-length confound) and fix (narrow to the genuinely comparable unit) both followed directly from careful reasoning about what was actually being averaged together, rather than from more data or a fundamentally different metric. This is the same discipline this course applied to QLoRA's honest accuracy cost, loss masking's two attempts, and catastrophic forgetting's precise claim — measure, find the real explanation when something doesn't match expectations, fix the actual cause, and confirm the fix rather than assuming it worked.
 
## 5. Module 06 checkpoint, and where this heads next
 
Four episodes into Module 06: the ReAct loop, built from nothing but Module 04's transformer and Module 05's masking (06.00); multi-tool selection and chained, multi-step tool use, with an honest data-imbalance failure (06.01); two candidate signals for catching that kind of failure automatically (06.02); and today, fixing the flawed signal and landing on a principled way to combine both (06.03). The next natural direction is broadening beyond a single agent with a fixed tool set — toward agent memory across multiple turns, planning over longer horizons, or coordination between multiple agents — any of which builds directly on the loop, evaluation, and fine-tuning machinery this module and the last one already established.
 
---
 
**Previous:** Episode 06.02 — Catching an Undertrained Tool Before It Fails
**Next:** Episode 06.04 — Agent Memory Context Limit