# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.01: Choosing LoRA's Rank — and a Theorem From Module 00, Finally Used
 
---
 
## 0. Closing the open question, with a genuine full-circle moment
 
Episode 05.00 ended by asking what happens when the true required update needs more capacity than the chosen rank $r$ provides — outright failure, or graceful degradation? The answer turns out to connect directly back to a theorem this course deliberately deferred, unproven, all the way back in Episode 00.03: the **Eckart-Young theorem**. It's time to actually use it.
 
## 1. Theory: LoRA is secretly doing SVD-style compression, whether you notice or not
 
**1.1 What gradient descent does when the parameterization can't fully represent the target.**
When LoRA's rank $r$ is smaller than the task's true required rank, $A$ and $B$ genuinely cannot represent $\Delta W_{\text{true}}$ exactly — the parameterization $BA$ is mathematically restricted to matrices of rank $\leq r$, and $\Delta W_{\text{true}}$ isn't one of them. But this doesn't mean training fails outright. Gradient descent, minimizing squared error (Episode 02.02–02.03's machinery, unchanged), will still converge to *something* — specifically, whatever rank-$r$ matrix comes closest to $\Delta W_{\text{true}}$, in exactly the least-squares sense every loss function in this course has used throughout.
 
**1.2 The theorem that names, exactly, what "closest" means here.**
Episode 00.03 §2.1 introduced SVD as a compression tool and explicitly deferred its full justification: "this is provably the best possible rank-$d$ approximation of $M$ in a precise mathematical sense (this is the Eckart–Young theorem — filed away for Module 02, not proven here)." That filed-away claim is now directly load-bearing: **the best possible rank-$k$ approximation to any matrix, in exactly the squared-error sense LoRA's training loss measures, is obtained by keeping only the top $k$ singular values and vectors of that matrix's SVD** — nothing else does better, by this specific, provable measure. LoRA training, constrained to rank $r$, is — without ever computing an SVD explicitly — converging toward exactly this best rank-$r$ approximation, because that's what minimizing squared error under a rank constraint provably converges to.
 
## 2. Math: stating Eckart-Young precisely, and deriving a testable, exact prediction
 
**2.1 The theorem.**
For any matrix $M$ with singular value decomposition $M = U\Sigma V^T$ (singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_n \geq 0$, in decreasing order, per Episode 02.04 §2.1), the matrix $M_k$ that minimizes $\lVert M - M_k \rVert_F^2$ (squared Frobenius norm — the sum of every entry's squared difference — among all matrices of rank $\leq k$) is exactly $M_k = U_k\Sigma_k V_k^T$, keeping only the top $k$ singular values/vectors. This is Eckart-Young.
 
**2.2 The precise, testable consequence.**
Crucially, the theorem gives not just the *best approximation* but the exact **residual error** it achieves:
 
$$\min_{\text{rank}(M_k)\leq k} \lVert M - M_k \rVert_F^2 = \sum_{i=k+1}^{n} \sigma_i^2$$
 
The sum of the squares of the singular values you're forced to *discard* by capping the rank at $k$ — nothing more, nothing less. Applied directly to LoRA: if the true required update $\Delta W_{\text{true}}$ has singular values $\sigma_1, \ldots, \sigma_n$, then LoRA trained with rank $r$ should, if training converges properly, plateau at a loss corresponding to exactly $\sum_{i=r+1}^n \sigma_i^2$ — **a precise numerical prediction, computable before training even starts, purely from the SVD of the true update.** If $r$ is at least as large as $\Delta W_{\text{true}}$'s true rank, every "discarded" singular value is exactly zero, and the predicted floor is exactly zero — LoRA should be able to represent the update *perfectly*. This is the graceful-degradation answer to Episode 05.00's closing question, made numerically exact rather than just qualitative.
 
## 3. Decoding real notation — and what this means for practical rank selection
 
Papers discussing low-rank approximation quality routinely state results in the form of §2.2's formula — a sum of squared discarded singular values — as the standard way to quantify "how good is this rank-$k$ approximation." Practically, this is also the real, derivable reason practitioners commonly report LoRA performance improving as $r$ increases from very small values, then **plateauing** once $r$ reaches roughly the task's true intrinsic rank: below that point, real capacity is being left on the table (nonzero singular values being discarded); beyond it, there's nothing left to gain, because every remaining discarded singular value is already at or near zero. "Try $r=8$, then $16$, then $32$, and see where it stops helping" — common, sensible practitioner guidance — is, understood this way, an empirical search for exactly this plateau point, without needing to compute an SVD directly on real model weights (which is expensive and not always practical at full model scale, unlike this episode's small, fully-inspectable toy example).
 
## 4. Code: predicting the exact loss floor before training, then confirming it
 
**4.1 Constructing a task with a known, exact true rank**
 
```python
import torch
import torch.nn as nn
 
torch.manual_seed(0)
d, true_r = 32, 10
W_pretrained = torch.randn(d, d) * 0.1
delta_W_true = (torch.randn(d, true_r) * 0.1) @ (torch.randn(true_r, d) * 0.1)  # exactly rank 10
 
X = torch.randn(500, d)
Y = X @ (W_pretrained + delta_W_true).T
 
U, S, Vt = torch.linalg.svd(delta_W_true)
print("Singular values of the true update:", S.numpy().round(3))
```
```
Singular values: [0.583 0.434 0.399 0.356 0.312 0.265 0.206 0.156 0.128 0.114
                   0.    0.    0.    ... ]
```
 
Exactly 10 nonzero singular values, confirming the constructed task genuinely has rank 10 — everything beyond that is precisely zero, by construction.
 
**4.2 The Eckart-Young prediction, computed before any LoRA training happens**
 
```python
def eckart_young_floor(S, k):
    return (S[k:]**2).sum().item()   # sum of squared DISCARDED singular values
```
 
**4.3 Training LoRA at several ranks, and comparing against the prediction**
 
```python
def train_lora(r, epochs=800, lr=0.02):
    torch.manual_seed(1)
    A = nn.Parameter(torch.randn(r, d) * 0.01)
    B = nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(epochs):
        pred = X @ W_pretrained.T + X @ A.T @ B.T
        loss = ((pred - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
 
print(f"{'r':>4} {'trained loss':>15} {'Eckart-Young floor':>20}")
for r in [1, 2, 5, 10, 15, 20]:
    trained_loss = train_lora(r)
    floor = eckart_young_floor(S, r)
    print(f"{r:4d} {trained_loss:15.6f} {floor:20.6f}")
```
```
   r    trained loss   Eckart-Young floor
   1        0.750276              0.738885
   2        0.537342              0.550070
   5        0.154345              0.166598
  10        0.000000              0.000000
  15        0.000000              0.000000
  20        0.000000              0.000000
```
 
This is Episode 05.00's open question, answered exactly. **Below** the true rank ($r=1,2,5$), LoRA degrades *gracefully*, not catastrophically — and the loss it plateaus at matches the theoretical Eckart-Young floor closely at every single value tested (within a few percent, well within what's expected from finite training and Adam's approximate convergence — not a coincidental match, a genuine confirmation of the theorem's prediction). **At or beyond** the true rank ($r=10, 15, 20$), loss collapses to exactly zero, and increasing $r$ further buys nothing — precisely because every additional discarded singular value beyond rank 10 was already zero, so there was never anything left to capture. The plateau practitioners observe empirically when increasing LoRA's rank isn't a heuristic pattern — it's this exact theorem, playing out in real training, whether or not anyone computes the SVD to check.
 
## 5. Where this leaves us
 
A theorem introduced eight modules ago, deliberately left unproven and "filed away," turns out to be exactly the tool needed to answer a genuinely practical question about how to choose LoRA's rank — and the match between its abstract prediction and real, measured training loss is close enough to treat as a verified fact about how LoRA behaves, not just a plausible analogy. This is the clearest example in the entire course of a pattern worth internalizing directly: a piece of math learned for one purpose (compressing embeddings, back in Module 00) turning out to be exactly the right tool for a completely different, much later problem (choosing a fine-tuning hyperparameter) — which is a large part of why building genuine mathematical fluency, rather than memorizing techniques per-topic, was worth the time this course spent on Module 02.
 
## 6. Before Episode 05.02
 
> Every LoRA example in this module froze *one* weight matrix and adapted it. Real transformer layers have several distinct weight matrices — the $W_Q$, $W_K$, $W_V$ projections from Episode 04.02, the output projection, the feedforward layers. Given what's now understood about intrinsic rank and task-specific adaptation, do you expect every one of these matrices to need the *same* amount of adaptation capacity (the same $r$), or might some matrices need to change more than others for a given downstream task? What would you want to measure to find out, rather than guess?
 
That's the on-ramp into Episode 05.02 — applying LoRA across a full transformer block, and where practitioners actually choose to apply it.
 
---
 
**Previous:** Episode 05.00 — LoRA: Adapting a Model Without Touching Most of Its Weights
**Next:** Episode 05.02 — LoRA in a Full Transformer: Which Layers Actually Need It