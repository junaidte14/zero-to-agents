# From Zero to Agents
## Module 06 — Agents
### Episode 06.02: Catching an Undertrained Tool Before It Fails
 
---
 
## 0. Closing the open question
 
Episode 06.01 ended by asking what could be measured automatically to catch an undertrained tool before it fails in production, rather than noticing a suspiciously small example count by eyeballing the data. This episode builds two candidate signals — and in testing them honestly, walks directly into a real instance of the exact evaluation-scope trap Episode 05.07 warned about, not manufactured for the lesson, just genuinely encountered.
 
## 1. Theory: two different signals, needing two different things
 
**1.1 Per-tool held-out loss — needs labels, computed offline.**
The direct extension of Episode 05.07's lesson: instead of one aggregate perplexity/loss number across an entire evaluation set, compute it **separately per tool** — a held-out loss for ADD examples, a separate one for DOUBLE examples, a separate one for CHAIN examples. A tool whose held-out loss is dramatically higher than the others is a strong, automatic, numeric signal that it's undertrained, without needing a human to notice anything about dataset sizes by hand.
 
**1.2 Generation confidence — no labels needed, usable live.**
A second, complementary signal: at generation time, for each token the model actually chooses (its own argmax pick), its own softmax distribution assigned that token *some* probability — and the negative log of that probability is a direct, computable measure of how confident the model was in its own choice, averaged across a generated response. Crucially, this needs **no ground truth at all** — it can be computed on brand-new, unlabeled production inputs, which is exactly what a live monitoring system needs, unlike §1.1's held-out loss, which requires labeled examples set aside before deployment.
 
## 2. Math: the same formula, two different inputs
 
**2.1 Held-out loss, formalized.**
For tool $k$'s held-out examples, $\mathcal{L}_k = \frac{1}{N_k}\sum_i -\log P(x_i \mid x_{<i})$, summed only over that tool's response positions (Episode 05.05's masking) — exactly Episode 05.07's response-only perplexity computation, sliced per tool instead of computed in aggregate.
 
**2.2 Generation confidence, formalized.**
For a freshly generated sequence with no ground truth available, $\text{Conf} = \frac{1}{T}\sum_t -\log P(\hat{x}_t \mid x_{<t})$, where $\hat{x}_t$ is specifically the token the model *actually chose* at each step (not a true label — there isn't one). This is the identical cross-entropy formula, evaluated against the model's own decisions instead of ground truth — a genuinely different quantity mathematically (it can never diagnose the model choosing a *confidently wrong* token, only an *uncertain* one), but computable anywhere, anytime, without labels.
 
## 3. Code: computing both signals on Episode 06.01's trained model — and an honest surprise
 
**3.1 Per-tool held-out loss**
 
```python
def per_type_holdout_loss(traces_maker, examples):
    traces = torch.tensor([pad(traces_maker(*ex) if isinstance(ex,tuple) else traces_maker(ex), max_trace_len)
                            for ex in examples])
    inp, tgt = traces[:, :-1], traces[:, 1:]
    masked_tgt = tgt.clone(); masked_tgt[:, :4] = -100
    with torch.no_grad():
        loss = F.cross_entropy(model(inp).reshape(-1,vocab_size), masked_tgt.reshape(-1), ignore_index=-100)
    return loss.item()
 
print(f"ADD:    {per_type_holdout_loss(trace_add, test_add):.4f}")
print(f"DOUBLE: {per_type_holdout_loss(trace_double, test_dbl):.4f}")
print(f"CHAIN:  {per_type_holdout_loss(trace_chain, test_chain):.4f}")
```
```
ADD:    0.0142
DOUBLE: 0.1266
CHAIN:  0.4452
```
 
**3.2 Label-free generation confidence**
 
```python
def generation_confidence(prompt_tokens, real_tool_fn_sequence):
    seq = torch.tensor([prompt_tokens])
    tool_idx, neg_log_probs = 0, []
    while True:
        probs = F.softmax(model(seq)[0,-1], dim=-1)
        next_id = probs.argmax().item()
        neg_log_probs.append(-torch.log(probs[next_id] + 1e-12).item())
        seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
        if next_id == O and tool_idx < len(real_tool_fn_sequence):
            seq = torch.cat([seq, torch.tensor([[real_tool_fn_sequence[tool_idx]()]])], dim=1)
            tool_idx += 1
        if next_id == EOS or len(neg_log_probs) > 12: break
    return sum(neg_log_probs) / len(neg_log_probs)
 
print(f"ADD:    {avg_conf_add:.4f}")
print(f"DOUBLE: {avg_conf_dbl:.4f}")
print(f"CHAIN:  {avg_conf_chain:.4f}")
```
```
ADD:    0.0006
DOUBLE: 0.1762
CHAIN:  0.0020
```
 
**3.3 Reading both results honestly — including the one that doesn't match expectations**
 
Generation confidence works exactly as hoped: DOUBLE's value (`0.1762`) is roughly **90–300x larger** than ADD's (`0.0006`) and CHAIN's (`0.0020`) — an unambiguous, automatic, label-free signal correctly flagging the one tool that actually failed its held-out test in Episode 06.01.
 
Held-out loss tells a *messier* story worth not glossing over: DOUBLE (`0.1266`) is indeed higher than ADD (`0.0142`), correctly flagging it — but **CHAIN's held-out loss (`0.4452`) is the highest of all three**, despite CHAIN generating perfectly, 5/5, in Episode 06.01's actual test. Investigating why turns up a direct instance of Episode 05.07's exact lesson: trace lengths differ substantially across task types (ADD: 12 tokens, DOUBLE: 10, CHAIN: 19), and the same fixed masking rule (`[:4]`) applied uniformly across all three types under-masks CHAIN's much longer sequence relative to the others, leaving more genuinely hard-to-predict positions (a second action's arguments, including a fresh random digit) inside CHAIN's aggregate loss than ADD's or DOUBLE's shorter, more thoroughly-masked sequences. **The held-out loss metric, exactly as built here, is comparing different evaluation scopes across the three task types without accounting for it** — precisely the trap Episode 05.07 named in the abstract, now caught happening for real, in a metric built specifically to be more rigorous than the previous episode's simple pass/fail check.
 
## 4. Where this leaves us
 
Two label-adjacent signals for the same underlying question, and they don't agree cleanly — which is itself the useful finding, not a failure of the experiment. Generation confidence, computed only over tokens the model actually committed to during a real autoregressive rollout, gave a clean, unambiguous, correctly-ranked signal. Held-out loss, computed via teacher-forcing over sequences of meaningfully different length and structure with a uniform masking rule, produced a misleading ranking for reasons that trace directly back to evaluation scope — the exact issue this course flagged as a general risk two episodes ago, now shown to bite even a metric built by someone who already knew to watch for it. The practical lesson: **a per-tool evaluation metric needs to control for structural differences between tools (sequence length, number of tool calls, amount of genuinely unpredictable content) or it risks measuring "how long and complex is this tool's trace format" instead of "how well-trained is this tool."**
 
## 5. Before the next episode
 
> Section 3.3's fix would need something like normalizing held-out loss by the number of genuinely-evaluated (unmasked) positions per example, rather than using a uniform masking cutoff across trace types of different length. Beyond that specific fix, this episode surfaced two signals measuring roughly the same thing through different lenses, and they disagreed. When two evaluation signals for the same underlying question disagree, how would you decide which one to trust — and is there a principled way to combine them rather than picking one?
 
That's worth sitting with directly before the next episode of Module 06.
 
---
 
**Previous:** Episode 06.01 — Multi-Tool Agents: Selection, Chaining, and an Honest Failure
**Next:** Episode 06.03 — Fixing Evaluation Metric