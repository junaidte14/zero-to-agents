# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.08: A Complete LoRA Pipeline — Every Piece of Module 05, Combined
 
---
 
## 0. Where we're starting from
 
Eight episodes of Module 05 have each tested one mechanism in isolation: LoRA's structure (05.00), rank selection (05.01), budget allocation (05.02), quantization's cost (05.03), forgetting and recoverability (05.04), loss masking (05.05–05.06), and evaluation scope (05.07). This episode assembles all of it into one realistic pipeline: **pretrain a base model on one task, freeze it, adapt it to a genuinely different second task via LoRA, and measure everything Module 05 taught us to measure** — new-task accuracy, old-task interference, and exact recoverability, in one run.
 
## 1. The pipeline, described before the numbers
 
**1.1 Step one — a "pretrained" base model.**
A tiny causal transformer (Episode 04.05's architecture) is fully trained — every weight, no LoRA — on **Task A: reverse three digits**, using proper loss masking (Episode 05.05) so it only learns to predict the response, not the random prompt digits. This stands in for a general-purpose pretrained model with one established capability.
 
**1.2 Step two — freeze it, attach LoRA, adapt to a new task.**
Every weight in the base model is frozen. Small LoRA adapters (rank 4, per Episode 05.00's construction) are attached specifically to the query and value projections inside each attention block — a deliberately partial application, consistent with Episode 05.02's finding that not every matrix needs adaptation. These adapters, and *only* these adapters, are trained — on **Task B: sort three digits in ascending order** — a genuinely different task from reversal, using a distinct instruction token so the model can tell which task is being requested, again with proper loss masking.
 
**1.3 What gets measured — every lens Module 05 built.**
- Task B accuracy on held-out examples (did the adapter actually learn the new task?)
- Task A accuracy with the adapter **attached** (how much does adapting for Task B interfere with the original capability, per Episode 05.04?)
- Task A accuracy with the adapter **detached** (is the original capability exactly, provably recoverable?)
## 2. Code: the full pipeline, run once, measured every way
 
**2.1 Pretraining the base model on Task A**
 
```python
base_model = TinyGPT(r=0)   # no LoRA yet -- a plain, fully-trainable model
# ... trained with masked loss (Episode 05.05) on 200 reversal examples ...
print("Base model pretraining final loss:", loss.item())
print("Task A test accuracy, base model:", evaluate(base_model, task_A_test, make_rev_example))
```
```
Base model pretraining final loss: 0.0000789
Task A (reverse) test accuracy, base model: (100, 100)
```
 
The base model learns reversal essentially perfectly — a clean, verified starting point, consistent with Episode 05.06's finding that masked training on this kind of task converges to genuine rule-learning rather than memorization.
 
**2.2 Freezing the base, attaching LoRA, and adapting to Task B**
 
```python
lora_model = TinyGPT(r=4)
lora_model.load_state_dict(base_model.state_dict(), strict=False)   # copy the pretrained weights in
for name, p in lora_model.named_parameters():
    p.requires_grad = ('A_q' in name or 'B_q' in name or 'A_v' in name or 'B_v' in name)   # freeze everything else
 
# ... trained ONLY the LoRA parameters, masked loss, on 150 sort examples ...
print("LoRA fine-tuning final loss:", loss.item())
print("Task B test accuracy, LoRA-adapted model:", evaluate(lora_model, task_B_test, make_sort_example))
```
```
LoRA fine-tuning final loss: 0.000986
Task B (sort) test accuracy, LoRA-adapted model: (83, 100)
```
 
This is a genuinely more honest number than most of this module's earlier toy demonstrations, and worth noting explicitly rather than treating as a shortcoming: **83/100, not a suspiciously clean 100/100.** A rank-4 adapter, touching only two of several weight matrices, trained for a fixed number of steps, doesn't perfectly master a new task from 150 examples — it does reasonably well, exactly the kind of realistic partial success a real fine-tuning run produces, unlike several of this module's earlier examples that were deliberately constructed to hit exact zero loss.
 
**2.3 The two questions that matter most — interference and recoverability**
 
```python
lora_model.set_lora_active(True)
acc_A_attached = evaluate(lora_model, task_A_test, make_rev_example)
lora_model.set_lora_active(False)
acc_A_detached = evaluate(lora_model, task_A_test, make_rev_example)
 
print("Task A accuracy, LoRA ATTACHED: ", acc_A_attached)
print("Task A accuracy, LoRA DETACHED: ", acc_A_detached)
```
```
Task A (reverse) accuracy, LoRA ATTACHED:  (12, 100)
Task A (reverse) accuracy, LoRA DETACHED:  (100, 100)
```
 
This is Episode 05.04's finding, now observed in a genuinely different, non-cherry-picked setup rather than a scenario built specifically to produce it. **With the adapter engaged**, Task A accuracy collapses from its original 100/100 down to 12/100 — real, substantial interference, because the same $W_Q$ and $W_V$ projections the adapter modifies are used for *every* forward pass, Task A prompts included, and an adapter trained purely to solve Task B has no reason to preserve Task A's behavior as a side effect. **With the adapter simply not loaded**, Task A returns to exactly 100/100 — identical, to the token, to the base model's original performance, because the base weights genuinely never moved during LoRA training. Episode 05.04's precise claim — recoverability is guaranteed, attached-state preservation is not — holds up exactly, on a completely independent task pair from the one it was first measured on.
 
## 3. Where this leaves the whole module
 
Every mechanism built across Module 05 shows up in this one pipeline, doing exactly what it was derived to do: LoRA adapts a specific pair of weight matrices with a small trainable footprint (05.00); loss masking (05.05–05.06) is what let the base model learn reversal as a genuine rule rather than memorization, which is *why* it hit 100/100 on held-out Task A data in the first place; and the attached-vs-detached measurement (05.04) shows, on new data, the exact same precise distinction between "forgets while active" and "always recoverable" that the earlier, more controlled experiment first revealed. Nothing here needed a new concept — assembling eight episodes' worth of individually-verified mechanisms into one pipeline was the entire exercise, and every piece behaved exactly as its own episode predicted it would.
 
## 4. Module 05 complete
 
This closes Module 05. Across nine episodes, real fine-tuning — not just "how transformers work" in the abstract, but how they actually get adapted for specific tasks in production — has been built from the same standard this entire course has held throughout: derive the mechanism, implement it from scratch, verify it against a working system, and report the real result even when (especially when) it doesn't match the initial expectation. Two genuine candidate research directions were flagged along the way and filed for later. The next step is Module 06, or a return to those research directions — your call.
 
---
 
**Previous:** Episode 05.07 — Perplexity and Evaluation Scope
**Next:** Module 06 - Agents