# From Zero to Agents
## Module 04 — Sequence Models and Attention
### Episode 04.00: Recurrent Neural Networks — and the Same Old Problem, Across Time Instead of Depth
 
---
 
## 0. Where we're starting from
 
Module 03 built a complete, verified toolkit for networks that take a **fixed-size** input and produce a fixed-size output. Language doesn't come in fixed sizes — a sentence might be 3 tokens or 300. Module 04 is where this course's two major threads finally meet: Module 01's token sequences and embeddings, processed by Module 03's trainable network machinery. The first serious attempt at bridging them, historically, was the **recurrent neural network (RNN)** — and it's worth building properly, not skipped, because the specific way it struggles is exactly what motivated the invention of attention.
 
## 1. Theory: processing a sequence with a network built for fixed size
 
**1.1 The naive options, and why they're unsatisfying.**
You could pad every sequence to some fixed maximum length and feed it in as one giant vector — but this wastes computation on padding, and worse, a network trained this way learns nothing that transfers between "the important word is in position 3" and "the important word is in position 30" — no shared structure across positions at all.
 
**1.2 The RNN idea — one token at a time, with memory carried forward.**
Instead: process the sequence one token at a time, and maintain a **hidden state** — a vector summarizing "everything relevant seen so far" — that gets updated at each step and fed back in as an additional input at the *next* step. Critically, the **same weight matrices are reused at every single timestep** — exactly one small network, applied repeatedly, rather than a different network per position. This weight-sharing is what lets an RNN generalize across sequence positions and lengths at all: whatever the network learned about combining "previous memory" and "current token" at position 3 applies identically at position 30, because it's literally the same weights doing the work.
 
**1.3 What this buys, and what it costs.**
The hidden state is, in principle, a compressed summary of the entire sequence so far — token 50 can theoretically be influenced by information from token 1, passed forward through 49 intermediate updates. In principle. Section 2 and Section 4 both show exactly why that "in principle" is doing a lot of work, and where it breaks down in practice.
 
## 2. Math: the recurrence, and training it via backpropagation through time
 
**2.1 The RNN update, precisely.**
At each timestep $t$:
 
$$\mathbf{z}_t = W_{hh}\mathbf{h}_{t-1} + W_{xh}\mathbf{x}_t + \mathbf{b}_h, \qquad \mathbf{h}_t = \tanh(\mathbf{z}_t), \qquad \mathbf{y}_t = W_{hy}\mathbf{h}_t + \mathbf{b}_y$$
 
Look closely: this is exactly Episode 03.01's layer equation ($\mathbf{z} = W\mathbf{a}+\mathbf{b}$, then a nonlinearity), with one addition — an extra term, $W_{hh}\mathbf{h}_{t-1}$, feeding the *previous* timestep's hidden state back in as additional input. Nothing else about the underlying machinery changes; this is a small, specific extension of everything Module 03 already built, not a new paradigm.
 
**2.2 Backpropagation Through Time (BPTT) — the same backprop, applied to an unrolled graph.**
To train an RNN, conceptually "unroll" the recurrence across all $T$ timesteps into an equivalent $T$-layer feedforward network — timestep 1's hidden state feeding timestep 2, feeding timestep 3, and so on — with the crucial detail that *every one of those "layers" shares the identical weight matrices* $W_{hh}, W_{xh}, W_{hy}$. Backpropagation then proceeds exactly as in Episode 03.04: an error signal computed at the output, propagated backward — but now backward *through time* as well as through any stacked layers, accumulating a gradient contribution to the *same* shared weight matrix from every single timestep it participated in. This is why it's called backpropagation *through time* rather than a different algorithm — it's Episode 03.04's four equations, applied to a specific graph shape.
 
**2.3 Vanishing gradients across time — the direct generalization of Episode 03.02.**
Here's the part worth sitting with: because $W_{hh}$ is applied *repeatedly*, once per timestep, the gradient flowing back to an early timestep involves a product of roughly $T$ copies of a similar Jacobian term (informally, $W_{hh}^T$ combined with $\tanh'$ at each step) — structurally identical to Episode 03.02's product of $L$ activation derivatives across *layers*, with sequence length $T$ now playing the exact role network depth $L$ played there. The difference that makes this often *worse* in practice: network depth $L$ is a fixed architectural choice, typically tens or at most a few hundred layers even in very deep networks — but sequence length $T$ can be genuinely enormous (a long document, a long conversation), meaning the vanishing-gradient shrinkage from Episode 03.02 §2.2's $(0.25)^L$-style bound can compound over a far larger exponent than anything Module 03 encountered. Section 4 measures this directly, not just asserts it.
 
## 3. Decoding real notation — the standard RNN (Elman network) equations
 
The formulation in §2.1 is, close to verbatim, how Elman networks — one of the earliest RNN formulations, and still the notation most introductory papers use — are written: $\mathbf{h}_t = \tanh(W_{hh}\mathbf{h}_{t-1}+W_{xh}\mathbf{x}_t+\mathbf{b}_h)$. Papers discussing training RNNs on long sequences frequently mention **truncated BPTT** — deliberately limiting the backward pass to only the most recent $k$ timesteps rather than the entire sequence — which, given §2.3, should now read as an entirely sensible engineering compromise rather than an arbitrary shortcut: if gradients from 50+ timesteps back are vanishing to numerical noise anyway, computing them exactly costs computation for essentially zero benefit.
 
## 4. Code: building an RNN from scratch, verifying BPTT exactly, then measuring the vanishing gradient directly
 
**4.1 From scratch — forward pass and full BPTT**
 
```python
import numpy as np
 
def tanh(z): return np.tanh(z)
def tanh_deriv(z): return 1 - np.tanh(z)**2
 
class ManualRNN:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.default_rng(seed)
        self.Wxh = rng.normal(0, 1, (hidden_size, input_size)) * np.sqrt(1/input_size)
        self.Whh = rng.normal(0, 1, (hidden_size, hidden_size)) * np.sqrt(1/hidden_size)
        self.Why = rng.normal(0, 1, (output_size, hidden_size)) * np.sqrt(1/hidden_size)
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        self.hidden_size = hidden_size
 
    def forward(self, xs):
        h_prev = np.zeros((self.hidden_size, 1))
        hs, zs, ys = [h_prev], [], []
        for x_t in xs:
            z = self.Whh @ h_prev + self.Wxh @ x_t + self.bh
            h_prev = tanh(z)
            zs.append(z); hs.append(h_prev); ys.append(self.Why @ h_prev + self.by)
        return zs, hs, ys
 
    def bptt(self, xs, targets):
        T = len(xs)
        zs, hs, ys = self.forward(xs)
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))
        total_loss = 0
 
        for t in reversed(range(T)):
            dy = ys[t] - targets[t]
            total_loss += 0.5 * np.sum((ys[t]-targets[t])**2)
            dWhy += dy @ hs[t+1].T
            dby += dy
            dh = self.Why.T @ dy + dh_next          # error from THIS step's output + FUTURE timestep
            dz = dh * tanh_deriv(zs[t])
            dbh += dz
            dWxh += dz @ xs[t].T
            dWhh += dz @ hs[t].T                     # accumulates across every timestep -- same shared matrix
            dh_next = self.Whh.T @ dz                # propagate error one step further back in time
        return total_loss, dWxh, dWhh, dWhy, dbh, dby
```
 
**4.2 Verifying against PyTorch, on an identical unrolled computation**
 
```python
import torch
 
rnn = ManualRNN(input_size=3, hidden_size=4, output_size=2, seed=42)
rng = np.random.default_rng(1)
T = 5
xs = [rng.normal(0,1,(3,1)) for _ in range(T)]
targets = [rng.normal(0,1,(2,1)) for _ in range(T)]
loss, dWxh, dWhh, dWhy, dbh, dby = rnn.bptt(xs, targets)
 
Wxh_t = torch.tensor(rnn.Wxh, dtype=torch.float64, requires_grad=True)
Whh_t = torch.tensor(rnn.Whh, dtype=torch.float64, requires_grad=True)
Why_t = torch.tensor(rnn.Why, dtype=torch.float64, requires_grad=True)
bh_t = torch.tensor(rnn.bh, dtype=torch.float64, requires_grad=True)
by_t = torch.tensor(rnn.by, dtype=torch.float64, requires_grad=True)
 
h = torch.zeros((4,1), dtype=torch.float64)
total_loss_t = 0
for t in range(T):
    z = Whh_t @ h + Wxh_t @ torch.tensor(xs[t]) + bh_t
    h = torch.tanh(z)
    y_pred = Why_t @ h + by_t
    total_loss_t = total_loss_t + 0.5*torch.sum((y_pred - torch.tensor(targets[t]))**2)
total_loss_t.backward()
 
print("Loss match?", np.isclose(loss, total_loss_t.item()))
print("dWhh match?", np.allclose(dWhh, Whh_t.grad.numpy(), atol=1e-8))
print("dWxh match?", np.allclose(dWxh, Wxh_t.grad.numpy(), atol=1e-8))
```
```
Loss match? True
dWhh match? True
dWxh match? True
```
 
Exact agreement, same pattern verified throughout Module 03, now confirmed for a recurrent architecture: manual backpropagation-through-time, computed with nothing but NumPy, matches PyTorch's automatic differentiation precisely.
 
**4.3 Measuring the vanishing gradient across time, directly**
 
```python
import torch.nn as nn
 
def run_rnn(T, hidden_size=20):
    rnn_cell = nn.RNNCell(input_size=5, hidden_size=hidden_size, nonlinearity='tanh')
    x_seq = [torch.randn(1, 5, requires_grad=True) for _ in range(T)]
    h = torch.zeros(1, hidden_size)
    for t in range(T):
        h = rnn_cell(x_seq[t], h)
    h.sum().backward()
    return [x.grad.norm().item() for x in x_seq]
 
for T in [5, 15, 30, 50]:
    grads = run_rnn(T)
    print(f"T={T:3d}: earliest-timestep grad = {grads[0]:.3e}   latest-timestep grad = {grads[-1]:.3e}   ratio = {grads[0]/grads[-1]:.3e}")
```
```
T=  5: earliest-timestep grad = 4.157e-02   latest-timestep grad = 1.581e+00   ratio = 2.629e-02
T= 15: earliest-timestep grad = 8.006e-05   latest-timestep grad = 1.084e+00   ratio = 7.388e-05
T= 30: earliest-timestep grad = 1.228e-07   latest-timestep grad = 9.356e-01   ratio = 1.312e-07
T= 50: earliest-timestep grad = 2.864e-15   latest-timestep grad = 1.605e+00   ratio = 1.784e-15
```
 
This is exactly Episode 03.02's vanishing-gradient story, replaying with $T$ (sequence length) in place of $L$ (network depth) — and visibly compounding *faster*: by $T=50$, the gradient reaching the earliest timestep is roughly $10^{-15}$ times the gradient at the latest timestep — utterly negligible, computationally indistinguishable from zero. **A vanilla RNN, in practice, cannot meaningfully learn dependencies spanning more than a few dozen timesteps at most** — token 1 of a long document has essentially no way to influence how token 200 gets processed, no matter how the network is trained, because the gradient connecting them has vanished before training can use it.
 
## 5. Where this leaves us
 
RNNs solve the *architectural* problem of variable-length sequences elegantly — one small network, weight-shared across every position, in principle carrying memory arbitrarily far forward. But §2.3 and §4.3 together show this promise is largely broken in practice by the exact same vanishing-gradient mechanism Module 03 diagnosed for depth, now afflicting *time* — and generally afflicting it worse, since realistic sequence lengths dwarf realistic network depths. This is not a minor caveat; it's the central limitation that shaped the next decade of sequence-model research.
 
## 6. Before Episode 04.01
 
> Episode 03.02 fixed the layer-depth version of this problem by changing the *activation function* (ReLU instead of sigmoid). That fix doesn't transfer cleanly here — tanh is used in RNNs specifically because its bounded output keeps the hidden state numerically stable across arbitrarily many timesteps, and swapping to unbounded ReLU inside a recurrent loop introduces its own severe instability (values can grow every single step, with nothing capping them). If changing the activation function isn't the fix this time, what architectural change to the recurrence itself might let *some* information survive across many timesteps, even if not all of it?
 
That's the on-ramp into Episode 04.01 — LSTMs and GRUs, the gated architectures designed specifically to fix this.
 
---
 
**Previous:** Module 03, Episode 03.06 — Classification Networks (Module 03 wrap)
**Next:** Episode 04.01 — LSTMs, GRUs, and the Gating Mechanism