# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.05: Loss Masking — Training Only on What the Model Should Actually Generate
 
---
 
## 0. Where we're starting from
 
Every fine-tuning example so far in this module used a toy regression setup — fit $X$ to $Y$, no notion of "prompt" versus "response." Real instruction fine-tuning data looks different: each training example is a *pair* — a prompt (the instruction, the context, the question) and a response (what the model should learn to generate). Feeding this through the exact same next-token training objective from Episode 04.05 raises an immediate, practical question this episode answers precisely: should the model be trained to predict *every* token in the sequence, prompt included, or only the response?
 
## 1. Theory: why training on the prompt is actively the wrong objective
 
**1.1 What the prompt actually is, functionally.**
A prompt is *given* — it's context the model receives, not something it should be learning to produce on its own initiative. Training the model to predict prompt tokens from earlier prompt tokens (exactly what an unmasked next-token loss would do, applying Episode 02.04's cross-entropy objective uniformly across the whole sequence) teaches it something true but useless: "when this specific prompt text appears, more of that same prompt text tends to follow" — a pattern about the *training data's* prompt distribution, not about how to respond well to arbitrary new prompts.
 
**1.2 Loss masking — computing loss only where it matters.**
The fix: compute the cross-entropy loss (Episode 02.04, Episode 03.06) only at positions corresponding to the **response**, and explicitly exclude every prompt position from the loss sum entirely — not down-weight, exclude. This is called **loss masking**, and it's standard practice for instruction and chat fine-tuning, described in papers on supervised fine-tuning (including the InstructGPT line of work, Ouyang et al., 2022) as "computing loss only on completion tokens."
 
## 2. Math: the masked loss, and PyTorch's exact mechanism for it
 
**2.1 The masked cross-entropy formula.**
Extending Episode 03.06's batched cross-entropy with an explicit binary mask $m_i \in \{0,1\}$ — 1 for response positions, 0 for prompt positions:
 
$$\mathcal{L} = -\frac{\sum_i m_i \log P(x_i \mid x_{<i})}{\sum_i m_i}$$
 
Every prompt position contributes exactly $m_i \cdot(\cdot) = 0$ to the numerator, and is excluded from the denominator's count entirely — the loss is a plain average, but only over the positions that were supposed to be generated.
 
**2.2 How this is actually implemented — `ignore_index`.**
PyTorch's `F.cross_entropy` supports this directly via an `ignore_index` parameter (conventionally set to $-100$ by strong convention, not a mathematical requirement): any target position whose label equals `ignore_index` contributes exactly zero to both the loss *and* the gradient — not approximately zero, exactly, verified directly in Section 4. This is precisely $m_i=0$ from §2.1, implemented as a sentinel label value rather than an explicit separate mask tensor — functionally identical, just a different way of encoding the same information.
 
## 3. Decoding real notation — and the practice this connects to
 
Papers and technical reports describing instruction fine-tuning routinely state, in prose rather than an explicit equation, that "loss is computed only on the completion" or "the prompt is masked out of the loss" — this is exactly §2.1/§2.2, described informally. Recognizing this phrase should immediately translate to: target labels at prompt positions are set to an ignore value (or an explicit mask multiplies them to zero) before the loss is computed, with the response positions contributing normally.
 
## 4. Code: the masking mechanism, verified exactly, then an honest look at a toy training comparison
 
**4.1 Verifying `ignore_index` does exactly what §2.1 claims — down to the gradient**
 
```python
import torch
import torch.nn.functional as F
 
vocab_size = 10
seq = torch.tensor([1, 2, 3, 9, 4, 5])   # 9 = a separator token between prompt and response
inputs, targets = seq[:-1], seq[1:]       # standard next-token shift
 
IGNORE_INDEX = -100
masked_targets = targets.clone()
masked_targets[:3] = IGNORE_INDEX          # positions 0,1,2 are "predict more prompt" -- excluded
 
logits = torch.randn(5, vocab_size, requires_grad=True)
loss_unmasked = F.cross_entropy(logits, targets)
loss_masked = F.cross_entropy(logits, masked_targets, ignore_index=IGNORE_INDEX)
manual_check = F.cross_entropy(logits[3:], targets[3:])   # compute by hand on JUST the response positions
 
print(f"Unmasked loss (all 5 positions): {loss_unmasked.item():.4f}")
print(f"Masked loss (ignore_index):      {loss_masked.item():.4f}")
print(f"Manual CE on response positions only: {manual_check.item():.4f}")
print("Masked loss matches manual computation exactly?", torch.allclose(loss_masked, manual_check))
 
loss_masked.backward()
print("\nGradient norm per position:", logits.grad.norm(dim=1).numpy().round(4))
print("Prompt positions have EXACTLY zero gradient?", torch.allclose(logits.grad[:3].norm(dim=1), torch.zeros(3)))
```
```
Unmasked loss (all 5 positions): 2.4016
Masked loss (ignore_index):      3.2611
Manual CE on response positions only: 3.2611
Masked loss matches manual computation exactly? True
 
Gradient norm per position: [0.     0.     0.     0.4134 0.6045]
Prompt positions have EXACTLY zero gradient? True
```
 
This confirms §2.2 precisely: `ignore_index` isn't an approximation or a down-weighting — masked positions receive exactly zero gradient, confirmed to full numerical precision, and the masked loss matches a completely independent manual computation exactly.
 
**4.2 An honest attempt at measuring the training-outcome benefit — and an honest report of what happened**
 
The natural next question: does masking actually produce a *better* model, not just a cheaper computation? A toy experiment was built to test this directly — a tiny causal transformer (Episode 04.05's architecture) trained on `"N -> (2N) mod 10"` examples for even $N \in \{0,2,4,6,8\}$, comparing masked training (loss only on the response digit) against unmasked training (loss on the full sequence), then testing generalization to unseen odd $N \in \{1,3,5,7,9\}$:
 
```
Final train loss (masked):    0.0000188
Final train loss (unmasked):  0.0000197
Masked model:   train acc=5/5, generalization acc=1/5
Unmasked model: train acc=5/5, generalization acc=2/5
```
 
**Reported honestly, this result does not support a clean "masking generalizes better" conclusion** — the unmasked model, if anything, edged out the masked one on this specific toy test. This isn't a failure of the theory in §1–§2; it's a real limitation of the experiment: five training examples and a two-layer toy transformer is nowhere near enough signal for *either* version to learn the actual underlying rule ("multiply by two, mod ten") rather than partially memorizing the small training set — both models are dominated by memorization noise at this scale, and the comparison genuinely doesn't have the statistical power to distinguish the two training regimes. Rather than construct a toy example carefully rigged to show the expected result, it's more honest — and consistent with this course's approach throughout — to report what actually happened and be precise about why it doesn't settle the question.
 
**4.3 What the real, correctly-scoped justification for masking actually is.**
The genuine case for loss masking, separate from this episode's inconclusive toy comparison, rests on two things that *are* directly verifiable, and were verified in §4.1: **exact computational efficiency** (zero gradient signal, and therefore zero wasted computation, spent on positions the model was never supposed to be optimized to generate), and a **behavioral risk that a small toy task can't expose but real deployment can** — a model trained to predict prompt content as if it were generating it can, at real scale with diverse enough training prompts, learn to imitate prompt-like patterns during its own generation (for instance, fabricating a fake continuation of a conversation, inventing a new "user turn" instead of stopping to let the actual user respond) — a genuine, documented failure mode in poorly-masked instruction-tuning setups, not measurable in a five-example arithmetic toy but real in production-scale fine-tuning.
 
## 5. Where this leaves us
 
Loss masking's *mechanism* is simple, exact, and fully verified here — a masked position contributes precisely zero to both loss and gradient, nothing approximate about it. Its *benefit* is real and well-documented in the literature and in production fine-tuning practice, but this episode's own toy experiment wasn't capable of demonstrating it convincingly — worth sitting with as a reminder that a clean mechanism verification (§4.1) and a convincing outcome demonstration (§4.2) are different bars, and conflating them is exactly the kind of overclaiming this course has tried to avoid throughout.
 
## 6. Before the next episode
 
> Section 4.2's toy task failed to show a training-outcome difference because five examples and a tiny model gave neither version enough signal to learn the real underlying rule. What would need to change about that experiment — more examples, a harder task where prompt-reproduction is more clearly a *wrong* thing to learn, a larger model, a different evaluation metric — to actually give loss masking a fair, convincing test? Sit with this concretely enough that it could be re-run correctly, rather than left as an open question.
 
That's worth returning to directly, with a properly designed experiment, before treating the loss-masking benefit as anything more than "well-supported by the literature and the mechanism, not yet demonstrated by this course's own code."
 
---
 
**Previous:** Episode 05.04 — Catastrophic Forgetting
**Next:** To be determined — a corrected version of this episode's experiment, further Module 05 territory, or the flagged research directions, your call