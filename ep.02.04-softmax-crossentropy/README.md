# From Zero to Agents
## Module 02 — Mathematical Foundations
### Episode 02.04: Probability, Softmax, and Cross-Entropy — The Loss Function Every LLM Actually Trains On
 
---
 
## 0. Closing the open question, and closing the module
 
Episode 02.03 ended by asking what tool lets you reason precisely about quantities you don't know exactly but know something about — like $\mathbb{E}[\cdot]$, sitting unformalized in Episode 00.01's rational-agent equation since the very first episode of this course. This episode builds probability and expectation properly, and then uses them for something specific: deriving **softmax** and **cross-entropy loss** — the exact mechanism, no substitutions, that trains every modern language model to predict the next token. This is the capstone of Module 02: by the end, vectors, matrices, gradients, and probability all combine into one real, working training step.
 
## 1. Theory: measuring uncertainty, precisely
 
**1.1 A random variable and its distribution.**
A **random variable** $X$ is a quantity whose value isn't fixed — it's determined by some random process. A **probability distribution** assigns a probability to every possible value $X$ could take, with two hard requirements: every probability is non-negative, and they all sum (or, for continuous variables, integrate) to exactly 1 — the outcome has to be *something*.
 
**1.2 Three examples, deliberately building toward the payoff.**
- **A fair coin**: $P(\text{heads}) = 0.5$, $P(\text{tails}) = 0.5$. Two outcomes, sums to 1.
- **A weighted next-word prediction**: over a tiny vocabulary `{the, cat, sat, mat}`, a model might assign $P = (0.5, 0.2, 0.2, 0.1)$ — this is *exactly* the kind of output a language model produces at every single generation step: a full probability distribution over its entire vocabulary, one number per possible next token.
- **Weight initialization**: real neural network weights are typically initialized by drawing from a continuous distribution (commonly Gaussian/normal) rather than a discrete one — a detail that becomes directly relevant the moment Module 03 starts training actual networks.
**1.3 Expectation — formalizing the $\mathbb{E}[\cdot]$ from Episode 00.01, at last.**
The **expectation** (or expected value) of a random variable is its probability-weighted average — not the value you'll get on any one draw, but the long-run average if you repeated the random process many times. This is precisely what Episode 00.01's rational-agent formula meant by $\mathbb{E}[\sum_t \gamma^t r_t]$: not a guaranteed reward total, but the reward total *averaged over every possible way the uncertain environment could unfold*, weighted by how likely each unfolding is.
 
**1.4 Softmax — turning arbitrary numbers into a valid probability distribution.**
Here's the connection worth the whole episode: attention (Episode 02.01) produces raw similarity scores — real numbers, possibly negative, that don't sum to anything meaningful. Episode 01.02 called the normalization step "softmax" without deriving it. Now, precisely: **softmax is the standard way to convert any vector of real numbers into a valid probability distribution** — satisfying both requirements from §1.1 (non-negative, sums to 1) — while preserving the *relative ordering* of the original scores (the biggest input score still produces the biggest output probability). This single function is doing identical work in two places we've already met: turning attention scores into attention *weights* (Episode 02.01), and turning word2vec's raw similarity scores into the prediction probability $p(w_O|w_I)$ (Episode 00.03) — same function, same reason, both times.
 
**1.5 Cross-entropy — measuring how wrong a predicted distribution is.**
If a model outputs a predicted probability distribution $q$ over possible next tokens, and there's a single true correct answer, **cross-entropy loss** measures how far $q$ is from being confidently, correctly right. It's the actual loss function that trains GPT-style models: at every position in a training sequence, the model produces a full probability distribution over the vocabulary for "what comes next," cross-entropy scores how good that distribution was against the token that actually came next, and gradient descent (Episode 02.03) adjusts every weight in the network to make that score better next time. This is not an analogy to how LLMs are trained — this is, almost exactly, the actual training objective, and Section 4 implements and verifies it directly.
 
## 2. Math: stated precisely
 
**2.1 Discrete expectation.**
 
$$\mathbb{E}[X] = \sum_{x} x \cdot P(x)$$
 
Multiply each possible value by its probability, and sum. For the fair coin example (assigning heads=1, tails=0): $\mathbb{E}[X] = 1(0.5) + 0(0.5) = 0.5$ — the familiar "on average, half heads" result, now derived from the definition rather than assumed.
 
**2.2 Softmax, stated precisely.**
 
$$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)}$$
 
Read it as: exponentiate every entry (guaranteeing everything becomes positive — satisfying §1.1's non-negativity requirement, since $\exp$ of any real number is always positive), then divide each one by the sum of all the exponentiated entries (guaranteeing everything sums to exactly 1). This is precisely the denominator-normalization structure from word2vec's skip-gram formula in Episode 00.03 — not a coincidence; that *was* softmax, named properly now.
 
**2.3 Cross-entropy loss, stated precisely.**
For a true distribution $p$ (in language modeling, almost always a one-hot vector — probability 1 on the actual next token, 0 everywhere else) and a predicted distribution $q$:
 
$$H(p, q) = -\sum_{x} p(x) \log q(x)$$
 
Because $p$ is one-hot in the language-modeling case, every term in this sum vanishes except the one at the true token's index — the formula collapses to the strikingly simple $-\log q(\text{true token})$: **just the negative log of the probability the model assigned to the actually-correct answer.** If the model was highly confident and correct, $q(\text{true}) \approx 1$, and $-\log(1) = 0$ — near-zero loss. If the model assigned the true token almost no probability, $q(\text{true}) \approx 0$, and $-\log(\text{near } 0)$ shoots toward infinity — a severe penalty for confident wrongness.
 
**2.4 The famous simplification — the gradient of softmax + cross-entropy, combined.**
This exact pairing (softmax feeding directly into cross-entropy) shows up so often specifically because their combined gradient simplifies beautifully. Skipping the derivation (a clean but involved chain-rule exercise, verified numerically in §4 instead of derived symbolically here):
 
$$\frac{\partial H}{\partial z_i} = q_i - p_i$$
 
**"Predicted minus true."** The gradient with respect to each raw score is just how much the predicted probability overshot (or undershot) the true probability at that position — nothing more complicated. This is a large part of *why* this specific loss/activation pairing became the near-universal default for classification and language modeling: the training signal it produces is about as clean and interpretable as gradients get.
 
## 3. Decoding a real equation — the actual language-modeling objective
 
Papers introducing autoregressive language models (the GPT family among them) write the training objective, in essence, as:
 
$$\mathcal{L} = -\sum_{i} \log P(x_i \mid x_{<i})$$
 
Read cold: $\mathcal{L}$ (script L) is the total loss over a sequence; $x_i$ is the $i$-th token; $x_{<i}$ means "every token before position $i$" (the context the model has seen so far). $P(x_i \mid x_{<i})$ is exactly the softmax output from §2.2, evaluated at the true next token — and $-\log(\cdot)$ of that is exactly §2.3's collapsed cross-entropy formula, summed over every position in the sequence. **This equation is nothing more than "run cross-entropy loss at every position in the sequence, and add them up."** Once softmax and cross-entropy are solid, this notation — which can look intimidating stripped of context — decodes immediately.
 
## 4. Code: every piece, verified against real data and real libraries
 
**4.1 Expectation — analytical formula vs. Monte Carlo simulation**
 
```python
import numpy as np
 
values = np.array([1, 2, 3, 4])           # toy numeric mapping for {the, cat, sat, mat}
probs = np.array([0.5, 0.2, 0.2, 0.1])
print("Sum of probabilities:", probs.sum())  # 1.0
 
analytical_E = np.sum(values * probs)
print("Analytical E[X]:", analytical_E)      # 1.9
 
rng = np.random.default_rng(42)
samples = rng.choice(values, size=200000, p=probs)
print("Monte Carlo estimate (200,000 draws):", samples.mean())
```
```
Sum of probabilities: 1.0
Analytical E[X]: 1.9
Monte Carlo estimate (200,000 draws): 1.90052
```
 
Analytical (§2.1's formula) and simulated (actually drawing 200,000 random samples and averaging) agree to three decimal places — confirming expectation isn't an abstract definition, it's a genuinely predictive average of what repeated random draws produce.
 
**4.2 Softmax — from scratch vs. PyTorch**
 
```python
def softmax(z):
    z_shifted = z - np.max(z)   # subtract the max first -- prevents exp() overflow, doesn't change the result
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)
 
scores = np.array([2.0, 1.0, 0.1, 3.0])
sm = softmax(scores)
print("Softmax:", sm, " sum:", sm.sum())
 
import torch
import torch.nn.functional as F
torch_sm = F.softmax(torch.tensor(scores), dim=0).numpy()
print("Match with torch?", np.allclose(sm, torch_sm))
```
```
Softmax: [0.236 0.087 0.035 0.642]  sum: 1.0
Match with torch? True
```
 
**4.3 Cross-entropy — from scratch vs. PyTorch, then the gradient simplification, verified by autograd**
 
```python
true_idx = 3   # the actually-correct token
ce_scratch = -np.log(sm[true_idx])
print("Cross-entropy (scratch):", ce_scratch)
 
logits = torch.tensor([scores], dtype=torch.float32)
ce_torch = F.cross_entropy(logits, torch.tensor([true_idx]))
print("Cross-entropy (torch):  ", ce_torch.item())
print("Match?", np.isclose(ce_scratch, ce_torch.item()))
```
```
Cross-entropy (scratch): 0.4436
Cross-entropy (torch):   0.4436
Match? True
```
 
```python
scores_t = torch.tensor([2.0, 1.0, 0.1, 3.0], requires_grad=True)
probs_t = F.softmax(scores_t, dim=0)
loss = -torch.log(probs_t[3])
loss.backward()
 
one_hot_true = np.array([0, 0, 0, 1])
predicted = probs_t.detach().numpy()
print("Autograd gradient:       ", scores_t.grad.numpy())
print("predicted - one_hot_true:", predicted - one_hot_true)
```
```
Autograd gradient:        [ 0.236  0.087  0.035 -0.358]
predicted - one_hot_true: [ 0.236  0.087  0.035 -0.358]
```
 
Exact match, confirming §2.4's "predicted minus true" simplification directly — not asserted, *measured*, straight out of real autograd on a real (if tiny) softmax+cross-entropy computation.
 
## 5. Module 02 wrap-up — the full stack, connected
 
Five episodes, one destination: vectors gave us the objects (Episode 02.00); matrices gave us organized, simultaneous operations on many vectors at once, and fully decoded the real attention formula (02.01); derivatives and gradients gave us the mechanism for improving a bad prediction (02.02); iterated gradient descent gave us a provable understanding of *when* that improvement process actually converges (02.03); and today, probability gave us the actual loss function — softmax turning raw scores into a probability distribution, cross-entropy measuring how wrong that distribution is against the truth.
 
Put together, plainly: **a language model computes attention scores (matrices), turns them into a probability distribution over the vocabulary (softmax), measures how wrong that distribution was (cross-entropy), and uses the gradient of that loss (calculus) to nudge every weight in the network slightly toward being less wrong (gradient descent) — repeated billions of times.** Every piece of that sentence is now something you've built from scratch, verified in code, and can recognize inside a real research paper's notation.
 
## 6. What Module 03 covers
 
Module 02 built the mathematical vocabulary. Module 03 — **Neural Networks from First Principles** — assembles it into an actual trainable system: perceptrons, multi-layer networks, backpropagation done properly across many stacked layers (not the single-step examples we've used to introduce each idea), weight initialization (where that Gaussian distribution from §1.2 becomes directly relevant), and finally training a real, working neural network end to end, in both raw Python and PyTorch, watching it learn.
 
---
 
**Previous:** Episode 02.03 — Iterating Gradient Descent
**Next:** Module 03, Episode 03.00 — Perceptrons and the Simplest Possible Neural Network