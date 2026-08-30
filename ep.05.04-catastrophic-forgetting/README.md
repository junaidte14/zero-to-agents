# From Zero to Agents
## Module 05 — Fine-Tuning and Adaptation
### Episode 05.04: Catastrophic Forgetting — What LoRA Actually Protects, Precisely
 
---
 
## 0. Where we're starting from
 
A common claim about LoRA, repeated often enough to sound settled: it reduces **catastrophic forgetting** — the well-documented tendency (McCloskey & Cohen, 1989; French, 1999) for a network trained on a new task to lose performance on whatever it learned before. This episode tests that claim directly, measures it precisely, and — consistent with Episode 05.03's discovery that a plausible-sounding claim doesn't always survive contact with real numbers — finds the truth is more specific, and arguably more useful, than the common phrasing suggests.
 
## 1. Theory: what forgetting actually is, and the naive expectation for LoRA
 
**1.1 Why full fine-tuning forgets.**
When every weight in a network is free to move, and training shifts entirely to a new task's data, nothing structurally protects the specific weight configurations that encoded the *previous* task's knowledge. Gradient descent (Episode 02.02 onward) only ever optimizes for the loss it's currently given — it has no built-in notion of "don't disturb what already worked," and previous-task performance can degrade substantially as a direct, unavoidable side effect of optimizing purely for the new task.
 
**1.2 The naive expectation for LoRA.**
Since LoRA (Episode 05.00) freezes the base weight $W$ entirely and only trains a small additive adapter, the intuitive hope is that the original knowledge, encoded in $W$, simply can't be disturbed — LoRA should therefore forget less. Section 4 tests this directly, and the result requires a real distinction the naive version of the claim glosses over.
 
## 2. Math: two genuinely different notions of "forgetting," made precise
 
**2.1 Attached-adapter performance vs. base-model performance — not the same question.**
There are two separate things worth asking about a LoRA-adapted model, and they have different answers:
1. **"How does the model, adapter attached, perform on the original task?"** The effective computation is $h = Wx + BAx$ for *every* input, old-task or new-task alike — nothing prevents the adapter, trained purely to minimize new-task loss, from producing outputs that interfere with old-task performance for old-task inputs too. If the adapter has enough capacity to fit the new task well, there's no structural reason its side effects on old-task inputs should be small.
2. **"How does the model perform with the adapter simply not loaded?"** This is $h = Wx$ alone — and because $W$ was never touched during LoRA training (Episode 05.00 §1.3's frozen-weight property, unconditionally true regardless of how well the adapter fits the new task), this is provably, exactly identical to the model's performance immediately after whatever training produced $W$ in the first place.
**2.2 The precise claim LoRA actually supports.**
LoRA does not generally guarantee small forgetting for question 1 — Section 4 shows a case where attached-adapter interference is comparable to full fine-tuning's forgetting. What LoRA *does* guarantee, exactly and unconditionally, is question 2: **the original model is always exactly recoverable, at zero additional cost beyond storing the adapter itself**, because $W$ was never modified. Full fine-tuning has no equivalent property — recovering the pre-fine-tuning model requires having separately saved a full copy of $W$ *before* fine-tuning started, an entire duplicate model's worth of storage, precisely the cost Episode 05.00 §1.2 identified LoRA as designed to avoid in the first place.
 
## 3. Decoding real notation — and a genuinely different philosophy worth knowing about
 
Catastrophic forgetting is a term from the connectionist neural network literature going back to McCloskey & Cohen (1989), well before deep learning's current era — worth knowing as a term with real history, not a phrase invented for transformers. It's worth contrasting LoRA's approach against a different family of mitigation entirely: **Elastic Weight Consolidation** (EWC, Kirkpatrick et al., 2017) takes the opposite philosophy — rather than freezing anything structurally, it lets every weight move during new-task training, but adds a penalty term to the loss discouraging movement specifically in directions estimated to be important for the old task (using something like the Fisher information matrix to estimate "importance" — a topic beyond this episode's scope, but worth recognizing by name). LoRA's protection is **structural** (certain weights literally cannot move); EWC's is **statistical** (weights can move, but movement in important directions is penalized). Recognizing which philosophy a given continual-learning technique uses — structural freezing versus penalized movement — is a useful lens for reading any paper in this space.
 
## 4. Code: testing the naive claim directly, then finding the real, precise result
 
**4.1 Setting up a genuine sequential two-task scenario**
 
```python
import torch
import torch.nn as nn
 
torch.manual_seed(0)
d = 32
W_A_target = torch.randn(d, d) * 0.1                 # Task A's ideal transformation
X_A, Y_A = torch.randn(300, d), None
Y_A = X_A @ W_A_target.T
 
delta_B = (torch.randn(d, 6)*0.1) @ (torch.randn(6, d)*0.1)   # Task B needs a further, rank-6 shift
W_B_target = W_A_target + delta_B
X_B = torch.randn(300, d)
Y_B = X_B @ W_B_target.T
 
def train_full(W_init, X, Y, epochs=500, lr=0.02):
    W = nn.Parameter(W_init.clone())
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(epochs):
        loss = ((X @ W.T - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach()
 
W_after_A = train_full(torch.randn(d, d) * 0.1, X_A, Y_A)
loss_A_baseline = ((X_A @ W_after_A.T - Y_A)**2).sum().item() / X_A.shape[0]
print(f"Task A loss right after training on Task A: {loss_A_baseline:.6f}")
```
```
Task A loss right after training on Task A: 0.000000
```
 
**4.2 Full fine-tuning on Task B, then re-measuring Task A**
 
```python
W_after_B_full = train_full(W_after_A, X_B, Y_B)
loss_A_after_full = ((X_A @ W_after_B_full.T - Y_A)**2).sum().item() / X_A.shape[0]
print(f"Task A loss AFTER full fine-tuning on Task B: {loss_A_after_full:.6f}")
```
```
Task A loss AFTER full fine-tuning on Task B: 0.628934
```
 
Genuine, substantial forgetting — exactly the textbook phenomenon.
 
**4.3 LoRA on Task B, tested both attached and detached**
 
```python
def train_lora(W_frozen, X, Y, r, epochs=500, lr=0.02):
    A = nn.Parameter(torch.randn(r, d)*0.01)
    B = nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(epochs):
        pred = X @ W_frozen.T + X @ A.T @ B.T
        loss = ((pred - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return A.detach(), B.detach()
 
A_lora, B_lora = train_lora(W_after_A, X_B, Y_B, r=6)
 
loss_A_detached = ((X_A @ W_after_A.T - Y_A)**2).sum().item() / X_A.shape[0]
pred_attached = X_A @ W_after_A.T + X_A @ A_lora.T @ B_lora.T
loss_A_attached = ((pred_attached - Y_A)**2).sum().item() / X_A.shape[0]
 
print(f"Task A loss, adapter DETACHED: {loss_A_detached:.6f}")
print(f"Task A loss, adapter ATTACHED: {loss_A_attached:.6f}")
```
```
Task A loss, adapter DETACHED: 0.000000
Task A loss, adapter ATTACHED: 0.629051
```
 
**This is the result that corrects the naive claim.** With the adapter attached, Task A's loss (`0.629`) is essentially *identical* to full fine-tuning's forgetting (`0.629`) — because the LoRA adapter here was given rank 6, exactly matching $\Delta_B$'s true rank (Episode 05.01), meaning it has more than enough capacity to fit Task B just as completely as full fine-tuning did, and the resulting interference with Task A is, correspondingly, just as severe. **LoRA's attached-adapter behavior did not forget less than full fine-tuning in this test.** But with the adapter simply detached, Task A's loss is exactly `0.000000` — a perfect, exact match to its performance immediately after Task A training, confirmed to machine precision, because $W_{\text{after A}}$ genuinely never changed.
 
## 5. Where this leaves us — the precise, correct claim
 
LoRA does not generally guarantee that an *engaged* adapter interferes less with prior knowledge than full fine-tuning would — Section 4.3 measured a case where it doesn't. What LoRA guarantees, exactly and for free, is that the pre-adaptation model is always sitting there, completely undisturbed, recoverable at zero cost beyond keeping the (tiny) adapter separate rather than merged in. Full fine-tuning has no equivalent option without having separately backed up an entire duplicate copy of the model beforehand — precisely the storage cost Episode 05.00 identified LoRA as built to avoid. For a genuinely multi-tenant system — swap adapters per request, base model never modified, every tenant's customization fully reversible and fully isolated from every other tenant's — this recoverability property, not reduced attached-state forgetting, is the real, precise, load-bearing benefit worth relying on.
 
## 6. Before the next episode
 
> This episode's forgetting test used a LoRA rank (6) exactly matched to the new task's true required rank — giving the adapter full capacity to fit Task B as completely as full fine-tuning did, which is exactly why attached-state interference came out comparable. What do you expect would happen to attached-state Task A performance if the LoRA rank were deliberately kept *smaller* than what Task B fully requires — trading some Task B performance for less attached-state interference with Task A? Is there a rank-dependent trade-off curve here worth measuring directly, the same way Episode 05.01 measured one for rank versus achievable loss?
 
That's a genuine, testable follow-up question — and, worth flagging directly: it's also a second candidate worth adding to the research-opportunities list, alongside Episode 05.02's rank-allocation question, since it's asking something with a precise, measurable answer that (as far as this course's material has covered) doesn't have an established, cited result the way Eckart-Young did for Episode 05.01's question.
 
---
 
**Previous:** Episode 05.03 — QLoRA: Quantizing the Frozen Base
**Next:** To be determined — the rank-vs-attached-forgetting trade-off, further Module 05 territory, or a return to the research-opportunities list, your call