# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.00: LoRA — Adapting a Model Without Touching Most of Its Weights
 
---
 
## 0. Where we're starting from
 
Module 04 ended with a complete, working (if tiny) GPT trained entirely from scratch. Real language models are pretrained once, at enormous cost, on massive general-purpose data — and then **adapted** for specific tasks or domains, rather than retrained from nothing every time a new use case appears. This module is about that adaptation step, and it opens with the technique most directly relevant to real production systems: **LoRA** (Low-Rank Adaptation, Hu et al., 2021) — a method for adapting a model's behavior while updating only a tiny fraction of its parameters.
 
## 1. Theory: why full fine-tuning becomes impractical at real scale
 
**1.1 What "full fine-tuning" means, and its real cost.**
The straightforward way to adapt a pretrained model: unfreeze every parameter, and run ordinary gradient descent (Episode 02.02 onward) on task-specific data, updating the entire network. This works, but the cost scales with the *full* size of the model — for a single $4096\times4096$ weight matrix (a realistic size inside a mid-sized language model), that's roughly 16.7 million trainable parameters, and a model has many such matrices across many layers. Beyond just the parameters themselves, training requires storing gradients and (for Adam-style optimizers) multiple additional state tensors per parameter — routinely 3-4x the raw parameter count in additional memory. At real model scale, this is a genuinely serious cost, not a minor inconvenience.
 
**1.2 Why this specifically hurts a multi-tenant setup.**
If a system needs a separately-adapted model *per customer or per use case* — a directly relevant scenario for any multi-tenant SaaS platform — full fine-tuning means storing an entire separate copy of the full model's weights per tenant. For a model with billions of parameters, that cost multiplies by however many tenants exist, which becomes prohibitive fast. This is the exact problem **parameter-efficient fine-tuning (PEFT)** methods, LoRA foremost among them, were designed to solve: adapt the model's behavior while storing only a small, per-tenant *difference* rather than a full duplicate model.
 
**1.3 The core LoRA idea.**
Freeze the pretrained weight matrix $W$ entirely — never update it at all. Instead, learn a small **additive update**, structured specifically so it has far fewer parameters than $W$ itself, and add it to $W$'s effect at inference time. The specific structure: represent the update as the product of two much smaller matrices, $\Delta W = BA$, where $B$ and $A$ are both "thin" — far fewer rows or columns than $W$ has. This works because of a specific, testable hypothesis about what fine-tuning updates actually look like, made precise in §2.
 
## 2. Math: low-rank updates, precisely, and why they're a reasonable bet
 
**2.1 The intrinsic-rank hypothesis.**
Prior work (Aghajanyan et al., 2020, on "intrinsic dimensionality") found that the actual adjustment needed to adapt a large pretrained model to a new task tends to live in a surprisingly **low-dimensional subspace** — even though $W$ itself is enormous, the *change* $\Delta W$ needed to solve a specific downstream task doesn't need anywhere near as much freedom. LoRA takes this as a direct, testable design assumption: constrain $\Delta W$ to be **low-rank** by construction, rather than letting it be an arbitrary full-rank matrix update.
 
**2.2 The parameterization.**
For a frozen weight $W \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$, LoRA introduces two trainable matrices $A \in \mathbb{R}^{r\times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}}\times r}$, with rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$ chosen deliberately small (commonly single digits to a few dozen). The effective forward computation becomes:
 
$$h = Wx + \frac{\alpha}{r}BAx$$
 
$\alpha$ is a scaling hyperparameter (details in §2.4). Crucially, $BA$ has the same shape as $W$ (so it can be added to $W$'s effect directly), but is constructed from far fewer numbers: $B$ contributes $d_{\text{out}}\times r$ parameters, $A$ contributes $r \times d_{\text{in}}$ — a total of $r(d_{\text{in}}+d_{\text{out}})$, versus $W$'s full $d_{\text{in}}\times d_{\text{out}}$. For the $4096\times4096$, $r=8$ example from §1.1: full fine-tuning needs $16{,}777{,}216$ parameters for that one matrix; LoRA needs $2 \times 4096 \times 8 = 65{,}536$ — **a 256x reduction**, for that single layer, with the reduction growing larger the bigger $W$ is.
 
**2.3 A precise, worth-verifying initialization detail.**
LoRA specifically initializes $B$ to **all zeros**, and $A$ to small random values (mirroring the initialization principles from Episode 03.03, applied to just one of the two factors). The immediate consequence: $BA = 0$ exactly at the start of training, meaning $h = Wx$ exactly — **the adapted model is mathematically identical to the unmodified pretrained model before a single training step happens.** This is a deliberate, safe starting point: fine-tuning begins from exactly the pretrained model's existing behavior, and only diverges from it as training actually teaches it something new.
 
**2.4 A subtlety worth deriving precisely — this is *not* the symmetry problem from Episode 03.03.**
Episode 03.03 §1.1 proved that identical weights across a layer cause a permanent symmetry-breaking failure. $B=0$ initialization might look similar at first glance — but it isn't the same failure, and working out exactly why is a genuinely useful exercise in careful gradient reasoning. The gradient of the loss with respect to $B$ involves $A$ (specifically, something proportional to $(Ax)$, the *output* of the $A$ matrix) — and since $A$ is randomly initialized, $Ax \neq 0$ in general, so **$B$ receives a real, nonzero gradient from the very first training step**, despite starting at zero. The gradient with respect to $A$, on the other hand, involves $B$ — and since $B=0$ at initialization, $A$'s gradient genuinely *is* exactly zero on the very first step. Only after $B$ moves away from zero (which it does, immediately, on step one) does $A$ begin receiving a nonzero gradient too, from step two onward. This isn't a symmetry problem because $A$'s random (not identical/tied) values still differ from unit to unit — the moment $B$ moves and gradient starts flowing to $A$, different rows of $A$ receive genuinely different gradients, exactly as intended. Section 4 verifies this precise, somewhat subtle sequence of events numerically rather than asserting it.
 
## 3. Decoding real notation — this is (very close to) the paper's own formula
 
Hu et al.'s LoRA paper states the forward pass essentially as written in §2.2: $h = W_0x + \Delta Wx = W_0x + BAx$, with the $\alpha/r$ scaling applied to $BA$ specifically so that changing $r$ (say, going from rank 8 to rank 16 to try more capacity) doesn't automatically also change the *typical magnitude* of the update — keeping $\alpha$ fixed while varying $r$ lets a practitioner treat $\alpha$ similarly to a single, stable "adaptation strength" knob, tuned somewhat independently from the separate question of how much capacity ($r$) the adaptation is given. Seeing "$W_0$" specifically (rather than plain $W$) in a paper is itself informative — the subscript zero is a common convention for "the original, frozen, pretrained value," distinguishing it explicitly from any trainable delta added on top.
 
## 4. Code: parameter counts confirmed, the initialization subtlety verified, and LoRA matched against full fine-tuning
 
**4.1 The parameter-count reduction, computed directly**
 
```python
d, r = 4096, 8
full_finetune_params = d * d
lora_params = d * r + r * d
print(f"Full fine-tuning: {full_finetune_params:,} params")
print(f"LoRA (r={r}):        {lora_params:,} params")
print(f"Reduction: {full_finetune_params/lora_params:.1f}x")
```
```
Full fine-tuning: 16,777,216 params
LoRA (r=8):        65,536 params
Reduction: 256.0x
```
 
**4.2 Confirming the zero-init property**
 
```python
import torch
import torch.nn as nn
 
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=8):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.1, requires_grad=False)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.1)   # random init
        self.B = nn.Parameter(torch.zeros(out_features, r))         # ZERO init
        self.scaling = alpha / r
 
    def forward(self, x):
        return x @ self.W.T + self.scaling * (x @ self.A.T @ self.B.T)
 
layer = LoRALinear(16, 16, r=4, alpha=8)
x = torch.randn(3, 16)
print("At init, LoRA output == frozen-base-only output?",
      torch.allclose(layer(x), x @ layer.W.T))
```
```
At init, LoRA output == frozen-base-only output? True
```
 
**4.3 Verifying the subtle gradient sequence from §2.4**
 
```python
x = torch.randn(5, 16)
target = torch.randn(5, 16)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
 
for step in range(3):
    loss = ((layer(x) - target)**2).mean()
    optimizer.zero_grad(); loss.backward()
    print(f"step {step}: |dA|={layer.A.grad.norm():.6f}  |dB|={layer.B.grad.norm():.6f}  |B|={layer.B.norm():.6f}")
    optimizer.step()
```
```
step 0: |dA|=0.000000  |dB|=0.383626  |B|=0.000000
step 1: |dA|=0.050465  |dB|=0.379142  |B|=0.038363
step 2: |dA|=0.099076  |dB|=0.381137  |B|=0.076277
```
 
Exactly as derived: at step 0, $A$'s gradient is precisely `0.000000` while $B$'s is already substantial — $B$ moves first. By step 1, $B$ has moved away from zero (`|B|=0.038`), and $A$'s gradient is now genuinely nonzero (`0.050`) for the first time. This is the precise mechanical sequence §2.4 predicted, confirmed to six decimal places.
 
**4.4 LoRA matched against full fine-tuning, on a task with genuinely low intrinsic rank**
 
```python
d = 32
W_pretrained = torch.randn(d, d) * 0.1
true_r = 2
W_target = W_pretrained + (torch.randn(d, true_r)*0.1) @ (torch.randn(true_r, d)*0.1)  # a low-rank task shift
 
X = torch.randn(200, d)
Y = X @ W_target.T
 
# Full fine-tuning: every entry of W trainable
W_full = nn.Parameter(W_pretrained.clone())
opt = torch.optim.Adam([W_full], lr=0.01)
for _ in range(300):
    loss = ((X @ W_full.T - Y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Full fine-tuning: final loss={loss.item():.6f}  params={d*d}")
 
# LoRA: only A, B trainable, rank matched to the task's true rank
A = nn.Parameter(torch.randn(true_r, d) * 0.01)
B = nn.Parameter(torch.zeros(d, true_r))
opt = torch.optim.Adam([A, B], lr=0.01)
for _ in range(300):
    loss = ((X @ W_pretrained.T + X @ A.T @ B.T - Y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"LoRA (r={true_r}):    final loss={loss.item():.6f}  params={d*true_r*2}")
```
```
Full fine-tuning: final loss=0.000000  params=1024
LoRA (r=2):    final loss=0.000000  params=128
```
 
Both converge to essentially zero loss — LoRA matches full fine-tuning's accuracy **exactly**, using $128$ trainable parameters instead of $1024$, an 8x reduction on this toy example (and the 256x figure from §4.1 at realistic scale). This isn't a coincidence of this specific toy problem — it's the intrinsic-rank hypothesis from §2.1 doing exactly what it claims: because the task's true required update genuinely was low-rank (constructed that way deliberately here, to test the hypothesis honestly), a low-rank *parameterization* had enough capacity to represent it exactly.
 
## 5. Where this leaves us
 
LoRA isn't a heuristic or an approximation accepted for convenience — it's a specific, testable bet (the intrinsic-rank hypothesis) about the *structure* of what fine-tuning actually needs to change, turned into an architecture that only pays for the parameters that bet requires. Verified directly: the zero-init property, the precise order in which $A$ and $B$ begin receiving gradient signal, and — on a task honestly constructed to have low intrinsic rank — an exact match to full fine-tuning's final accuracy at a fraction of the trainable parameters.
 
## 6. Before Episode 05.01
 
> Section 4.4's toy task was deliberately constructed so the *true* required update had exactly rank 2, and LoRA was given exactly rank 2 to work with — a best-case setup. What do you think happens if the true task actually requires a higher-rank update than the $r$ chosen for LoRA — say, the task genuinely needs rank 10 worth of adaptation capacity, but LoRA is only given $r=2$ to work with? Would you expect it to fail outright, or degrade gracefully — and what would "degrade gracefully" even look like in the loss curve?
 
That's the on-ramp into Episode 05.01 — choosing $r$ in practice, and what happens when the low-rank assumption doesn't hold as cleanly as it did here.
 
---
 
**Previous:** Module 04, Episode 04.05 — Causal Masking and a Complete GPT-Style Model
**Next:** Episode 05.01 — Choosing LoRA's Rank, and When the Low-Rank Assumption Breaks