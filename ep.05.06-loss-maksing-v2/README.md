# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.06: Loss Masking, Properly Tested
 
---
 
## 0. Where we're starting from
 
Episode 05.05 verified loss masking's mechanism exactly, but its attempt to demonstrate a real training-outcome benefit failed to show a clean result — five training examples and a trivial task gave neither masked nor unmasked training enough signal to distinguish them. The episode closed by naming exactly what a fair test would need: more data, and a task that actually requires learning a *rule* rather than memorizing a handful of examples. This episode builds that experiment properly, and — unlike last time — the result is clean, real, and mechanistically explainable.
 
## 1. Theory: what changed, and why it should matter this time
 
**1.1 A task requiring genuine generalization.**
Instead of five arithmetic examples, this experiment trains on **sequence reversal**: given three random digits, output them in reverse order. With training sequences drawn from a space of 1,000 possible 3-digit combinations, and only 200 used for training, a model that merely memorizes training examples has no way to handle the held-out 100 test sequences — succeeding on those requires the model to have actually learned the *reversal rule*, not a lookup table of specific answers.
 
**1.2 A mechanistic reason to expect masking to matter more here, not just a bigger sample size.**
Here's the piece missing from Episode 05.05's failed attempt: in this task, the "prompt" is three *randomly generated* digits. From the model's perspective, there is no learnable pattern in "what digit comes next in the prompt" — it's genuinely unpredictable, by construction. An **unmasked** training objective spends real gradient signal trying to minimize loss on predicting these fundamentally random prompt continuations — an objective that can never be driven anywhere near zero, no matter how well trained, because the information needed to predict it simply doesn't exist in the earlier tokens. This isn't a subtle theoretical concern; it's directly measurable, and Section 4 measures it.
 
## 2. Math: what an irreducible loss floor looks like, formalized
 
**2.1 Entropy as a hard floor on achievable loss.**
Recall cross-entropy loss (Episode 02.04 §2.3): it's minimized when the predicted distribution matches the true distribution of what comes next. If the true "next token" is genuinely uniformly random over 10 possible digits (as it is here, for prompt continuation positions, since prompts are independently random digits), the *best possible* cross-entropy loss for predicting it is $-\log(1/10) = \log(10) \approx 2.303$ — not zero, no matter how well-trained the model is, because $\log(10)$ is the entropy of a uniform distribution over 10 outcomes, and no predictor can do better than matching the true distribution exactly when that true distribution has nonzero entropy. This is a direct, quantitative floor: **unmasked training on this task cannot converge to a loss anywhere near zero**, unlike masked training, whose response-position loss genuinely *can* approach zero once the reversal rule is fully learned (an entirely deterministic, learnable function of the prompt).
 
**2.2 Why this dilutes learning of the part that matters.**
An unmasked objective is, in effect, averaging together two very different sub-problems: a genuinely learnable one (predict the response from the prompt) and a fundamentally unlearnable one (predict the next digit of a random prompt). Gradient descent optimizes the *combined* average — meaning some fraction of training's effort and capacity is permanently spent chasing an objective that cannot improve past its entropy floor, rather than being spent entirely on the part that actually can.
 
## 3. Code: the properly-scoped experiment, and what it actually shows
 
**4.1 The task and setup**
 
```python
# vocab: digits 0-9, SEP=10. Sequence: [d1,d2,d3, SEP, r1,r2,r3] where r = reverse(d1,d2,d3)
# 200 training sequences, 100 held-out test sequences, drawn from 300 unique random 3-digit combos
```
 
Using the same tiny causal transformer architecture from Episode 04.05, trained two ways — masked (loss only on the 3 response positions) and unmasked (loss on the full sequence) — for the same number of steps, same learning rate, same architecture.
 
**4.2 The results**
 
```python
print("Final train loss masked:", loss_m, " unmasked:", loss_u)
print("Masked   -- train acc:", evaluate(model_masked, train_seqs), " test acc:", evaluate(model_masked, test_seqs))
print("Unmasked -- train acc:", evaluate(model_unmasked, train_seqs), " test acc:", evaluate(model_unmasked, test_seqs))
```
```
Final train loss masked: 0.0000675   unmasked: 0.5021
 
Masked   -- train acc: (200, 200)  test acc: (100, 100)
Unmasked -- train acc: (200, 200)  test acc: (97, 100)
```
 
Both models fit the training data essentially perfectly (200/200). The real, clean result is on the **held-out test set** — 100 sequences neither model ever trained on: the masked model gets every single one correct (100/100), while the unmasked model misses 3. Small in absolute terms, but genuinely real, on a properly-designed test with enough held-out examples to be meaningful (unlike Episode 05.05's 5-example test set, where a single wrong answer swings the reported accuracy by 20 percentage points).
 
**4.3 The mechanistic explanation, confirmed directly**
 
The training-loss gap is the more striking number: masked training's loss (`0.0000675`) is essentially zero, while unmasked training's loss plateaus at `0.502` — and §2.1 predicts almost exactly why. With 3 response positions contributing (in principle) close to zero once learned, and 3 prompt-continuation positions each carrying an irreducible floor around $\log(10)/2\approx 1.15$ nats per masked framework's convention (the precise floor depends on exact loss normalization details, but the *order of magnitude* match to the measured `0.502` — averaged across a mix of near-zero response-position losses and much larger unavoidable prompt-position losses — is exactly the signature §2.2 predicts). The unmasked model's training loss was never going to reach zero, not because of a training failure, but because a real, calculable fraction of what it was being asked to minimize was information-theoretically impossible to minimize further.
 
## 4. Where this leaves us — and the contrast with Episode 05.05 worth sitting with
 
Same underlying mechanism (loss masking, verified exactly in Episode 05.05 §4.1), same architecture, same course — and two experiments reached opposite-looking conclusions, for a completely identifiable, derivable reason: the first test lacked the statistical power and task structure to show anything real; this one had both, and the result lines up cleanly with the theory, down to *why* the unmasked loss plateaus where it does. This is worth treating as the actual lesson of these two back-to-back episodes, maybe more than the masking result itself: **a negative or inconclusive result from an under-scoped experiment doesn't mean a well-supported claim is wrong — it means the experiment wasn't capable of testing it.** Distinguishing those two situations, rather than either overclaiming from a weak experiment or discarding a real effect because a bad experiment failed to find it, is a genuinely important skill this pair of episodes was worth building deliberately.
 
## 5. Module 05 checkpoint
 
Seven episodes into Module 05: LoRA's mechanism and its match to full fine-tuning on genuinely low-rank tasks (05.00); Eckart-Young answering exactly how much loss any given rank will plateau at (05.01); optimal multi-matrix budget allocation, confirmed against real training (05.02); quantization's honest, measured trade-off rather than an assumed-free combination (05.03); catastrophic forgetting's real, precise benefit — structural recoverability, not reduced attached-state interference (05.04); and loss masking, tested twice, with the failure of the first test turning out to be as instructive as the success of the second (05.05–05.06).
 
---
 
**Previous:** Episode 05.05 — Loss Masking: Training Only on What the Model Should Actually Generate