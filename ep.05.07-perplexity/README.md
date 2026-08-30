# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.07: Perplexity — and Why Evaluation Scope Can Flip Your Conclusion
 
---
 
## 0. Where we're starting from
 
Episode 05.06 evaluated masked versus unmasked training using exact-match accuracy — a clean metric for a task with one unambiguous correct answer. Most language modeling evaluation instead uses **perplexity**, a metric derived directly from cross-entropy loss. This episode derives it precisely, and then uses the exact two models from Episode 05.06 to demonstrate something genuinely important: **perplexity isn't one number** — what positions it's computed over changes not just the magnitude, but the entire conclusion you'd draw from it.
 
## 1. Theory: perplexity as an interpretable version of cross-entropy
 
**1.1 Why cross-entropy alone is hard to interpret intuitively.**
A cross-entropy loss of `2.3` is mathematically precise (Episode 02.04 §2.3) but not intuitively meaningful on its own — "2.3 nats" doesn't immediately tell you how good or bad a model's predictions actually are in a way that's easy to reason about.
 
**1.2 Perplexity — converting log-loss back into an "effective number of choices."**
**Perplexity** is defined as $e$ raised to the average cross-entropy loss — converting back out of log-space into something with a direct, intuitive reading: **the effective number of equally-likely options the model was, on average, choosing between.** A perplexity of exactly 1 means the model assigned essentially all probability to the correct answer, every time — as good as prediction can get. A perplexity of $V$ (the vocabulary size) means the model's predictions were, on average, no better than picking uniformly at random across the entire vocabulary — as bad as a well-calibrated model can get.
 
## 2. Math: the formula, and the scope-dependence that's easy to overlook
 
**2.1 The formula, precisely.**
$$\text{Perplexity} = \exp\left(\frac{1}{N}\sum_{i=1}^N -\log P(x_i \mid x_{<i})\right) = \exp(\overline{\mathcal{L}})$$
 
where $\overline{\mathcal{L}}$ is the average cross-entropy loss per token, and $N$ is however many token positions are included in that average. **That last detail — which positions are included in $N$ — is the entire subject of this episode**, because it's rarely stated as carefully as it should be, and it can change which model looks better.
 
**2.2 The uniform-distribution sanity check.**
If a model's predictions for some set of positions are exactly uniform over $V$ possible outcomes, cross-entropy at those positions is exactly $\log V$ (Episode 05.06 §2.1's entropy floor, restated), so perplexity there is exactly $\exp(\log V) = V$. For the reversal task from Episode 05.06, with 10 possible digits, that's a perplexity of exactly `10` — a hard reference point for "no better than random guessing," worth keeping in mind for §4.
 
## 3. Decoding real notation — what a reported perplexity number actually promises
 
Papers routinely report results like "our model achieves a perplexity of $X$ on [some benchmark]." Read cold, this promises much less than it might seem to: it's a single number, averaged over whatever the paper's evaluation set consists of — and if that evaluation set mixes token types with very different inherent predictability (as this episode's experiment will show happens even in a tiny toy example), a single aggregate perplexity can average away exactly the distinction that matters. Recognizing this is a genuinely useful habit for reading benchmark comparisons: a lower reported perplexity is only meaningful once you know precisely what it was computed over.
 
## 4. Code: the exact same two models from Episode 05.06, evaluated two different ways
 
**4.1 Perplexity, computed over the full sequence vs. response-only**
 
Reusing Episode 05.06's masked-trained and unmasked-trained models on the digit-reversal task, exactly as they finished training there — computing perplexity two ways for each: over the **entire sequence** (prompt and response positions both), and over **response positions only**:
 
```python
def perplexity(model, seqs, response_only=True):
    inputs, targets = build_batch(seqs)
    with torch.no_grad():
        logits = model(inputs)
        if response_only:
            logits_eval = logits[:, 3:, :].reshape(-1, vocab_size)
            targets_eval = targets[:, 3:].reshape(-1)
        else:
            logits_eval = logits.reshape(-1, vocab_size)
            targets_eval = targets.reshape(-1)
        ce = F.cross_entropy(logits_eval, targets_eval)
    return math.exp(ce.item()), ce.item()
 
for name, model in [("masked", model_masked), ("unmasked", model_unmasked)]:
    ppl_resp, _ = perplexity(model, test_seqs, response_only=True)
    ppl_full, _ = perplexity(model, test_seqs, response_only=False)
    print(f"{name:9s} -- response-only PPL: {ppl_resp:.3f}   full-sequence PPL: {ppl_full:.3f}")
```
```
masked    -- response-only PPL: 1.000   full-sequence PPL: 148.402
unmasked  -- response-only PPL: 1.042   full-sequence PPL: 9.242
 
(reference: uniform-random-over-10-digits perplexity = 10.000)
```
 
**4.2 Reading this result precisely — this is the whole episode.**
 
Look at what happens depending on which number you trust. **Response-only perplexity** — the metric that actually reflects what these models need to do (predict the correct reversal) — correctly ranks the masked model as better (`1.000` vs. `1.042`), consistent with Episode 05.06's accuracy result (100/100 vs. 97/100). But **full-sequence perplexity** tells the *opposite* story: the unmasked model looks dramatically better (`9.242` vs. `148.402`) — a naive reader comparing only this number would conclude unmasked training won, decisively, which is the wrong conclusion entirely.
 
**Both numbers are computed correctly.** Neither is a bug. The explanation is exactly Episode 05.06 §2's entropy-floor argument, now visible through a second, independent metric: the masked model was *never trained* to predict prompt-continuation positions at all (they were excluded from its loss, by design), so evaluating it there produces near-nonsense predictions — a genuinely bad, but entirely expected, perplexity at those positions specifically. The unmasked model, having spent training effort on those same positions, does reasonably well there — reaching a perplexity of `9.24`, close to the theoretical floor of `10` for genuinely unpredictable random digits, essentially as good as is achievable at that task. Averaging that in with the response positions *dilutes* the full-sequence number toward looking artificially good for the unmasked model, and artificially bad for the masked model — even though the masked model is unambiguously better at the thing that actually matters.
 
## 5. Where this leaves us
 
A single perplexity number, without knowing precisely what it was computed over, can flip which of two models looks better — not through any error in the computation, but purely through a choice of evaluation scope that happens to interact badly with how the two models were trained. This is a direct, concrete illustration of a broader principle worth carrying forward from this course's earlier authority piece on testing whether a system is "actually intelligent or just a very good parrot": a single aggregate metric is never sufficient on its own — the *design* of the evaluation (what exactly gets measured, over which positions, against what reference) matters as much as the number itself, and a model that looks worse on a poorly-scoped metric can be the genuinely better one.
 
## 6. Module 05 wrap-up
 
Eight episodes across Module 05: LoRA's mechanism, matched exactly against full fine-tuning where the intrinsic-rank hypothesis holds (05.00); Eckart-Young predicting exact loss floors for any chosen rank (05.01); optimal budget allocation across multiple matrices (05.02); quantization's honest, measured accuracy cost (05.03); catastrophic forgetting's real, precise benefit — structural recoverability (05.04); loss masking tested twice, with an instructive failure followed by a properly-scoped success (05.05–05.06); and today, perplexity's scope-dependence, demonstrated on the same two models rather than asserted abstractly. Every claim across this entire module was computed, and — as this pair of episodes made especially clear — sometimes recomputed properly after an honest first attempt fell short.
 
---
 
**Previous:** Episode 05.06 — Loss Masking, Properly Tested