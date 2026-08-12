# From Zero to Agents
## Module 04 — Sequence Models and Attention
### Episode 04.01: LSTMs and GRUs — Giving the Gradient a Path That Doesn't Vanish
 
---
 
## 0. Closing the open question
 
Episode 04.00 ended by asking what architectural change — not a different activation function — might let information survive across many timesteps. The answer, from Hochreiter & Schmidhuber's 1997 **LSTM** (Long Short-Term Memory): give the network a *second*, separate piece of memory that updates mostly through addition rather than being fully rewritten and re-squashed through a nonlinearity at every single step. This episode derives exactly why that fixes the vanishing gradient, builds it from scratch, and measures the improvement directly.
 
## 1. Theory: the core idea, before any equations
 
**1.1 What actually went wrong in the vanilla RNN.**
Recall Episode 04.00 §2.1: the hidden state $\mathbf{h}_t$ gets **completely recomputed** at every step — multiplied by a weight matrix, combined with the new input, and squashed through $\tanh$. The gradient flowing backward through this path has to pass through that same weight-multiply-then-squash operation once per timestep, and (Episode 04.00 §2.3, §4.3) that repeated multiplication shrinks it toward zero for any sequence of meaningful length.
 
**1.2 The fix — an additional memory path that's mostly addition.**
An LSTM introduces a **cell state** $\mathbf{c}_t$, separate from the hidden state, updated primarily by *adding* new information to the previous cell state, rather than replacing it outright — closer to $\mathbf{c}_t \approx \mathbf{c}_{t-1} + (\text{new stuff})$ than to a full rewrite. Section 2 shows this design choice has a precise, provable consequence for how gradients flow.
 
**1.3 Gates — learned, per-dimension "how much to let through" controls.**
The mechanism controlling this addition is a set of **gates** — vectors of values between 0 and 1 (produced by sigmoid, Episode 03.02 §1.1), one value per dimension of the cell state, each acting like a dial for that dimension: near 1 means "let this through almost entirely," near 0 means "block this almost entirely." An LSTM uses three: a **forget gate** deciding how much of the old cell state to keep, an **input gate** deciding how much of the new candidate information to add, and an **output gate** deciding how much of the (updated) cell state to actually expose as the hidden state used for predictions. All three gates are themselves small learned functions of the current input and previous hidden state — the network learns *when* to remember and *when* to forget, rather than following a fixed rule.
 
## 2. Math: the four equations, and the gradient path they create
 
**2.1 The full LSTM update.**
Using $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ to denote concatenating the previous hidden state and current input into one vector (standard notation in essentially every paper describing LSTMs):
 
$$\mathbf{f}_t = \sigma(W_f[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_f) \quad \text{(forget gate)}$$
$$\mathbf{i}_t = \sigma(W_i[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_i) \quad \text{(input gate)}$$
$$\mathbf{g}_t = \tanh(W_g[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_g) \quad \text{(candidate values)}$$
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \mathbf{g}_t \quad \text{(cell state update)}$$
$$\mathbf{o}_t = \sigma(W_o[\mathbf{h}_{t-1},\mathbf{x}_t]+\mathbf{b}_o) \quad \text{(output gate)}$$
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$
 
Recall $\odot$ from Episode 03.04 §2.1 — elementwise multiplication, exactly what a per-dimension "how much to let through" gate requires.
 
**2.2 The derivation that actually explains why this works.**
Compare the sensitivity of $\mathbf{c}_t$ to $\mathbf{c}_{t-1}$ against the vanilla RNN's sensitivity of $\mathbf{h}_t$ to $\mathbf{h}_{t-1}$. From §2.1's cell-state equation, treating $\mathbf{f}_t$ as roughly constant with respect to $\mathbf{c}_{t-1}$ for this comparison:
 
$$\frac{\partial \mathbf{c}_t}{\partial \mathbf{c}_{t-1}} \approx \mathbf{f}_t$$
 
Compare this directly against the vanilla RNN's equivalent quantity from Episode 04.00 §2.3, which involves $W_{hh}^T$ multiplied elementwise by $\tanh'(\mathbf{z}_t)$ — a full matrix multiplication *and* a squashing derivative, at every single step. The LSTM's cell-state path has **neither**: it's just $\mathbf{f}_t$, a simple elementwise multiplication by a learned gate value. If the network learns to set $\mathbf{f}_t \approx 1$ for information worth preserving (which is exactly what training pushes it toward, for any information the loss function benefits from remembering), the gradient flowing back through the cell state across many timesteps is multiplied by something close to $1$ at each step — not by a shrinking product of matrix-and-derivative terms. This is the entire mechanism, made precise: **the LSTM doesn't eliminate the vanishing gradient problem in general, it gives the network an optional, learnable path where the multiplicative shrinkage can be made to nearly disappear**, for whatever specific information the network learns is worth keeping.
 
**2.3 GRUs — a lighter alternative.**
The **Gated Recurrent Unit** (Cho et al., 2014) simplifies this: merge the cell state and hidden state into one, and use only two gates (a **reset gate** and an **update gate**) instead of three. It's a real, widely-used architecture with somewhat fewer parameters per cell, achieving broadly similar gradient-preserving behavior through the same core idea — a gated, mostly-additive update path — without a full derivation here, since the mechanism is a direct variation on §2.1–§2.2 rather than a fundamentally different idea.
 
## 3. Decoding real notation — this is (close to) exactly how papers state it
 
The four-gate formulation in §2.1 is essentially verbatim how LSTMs are presented in the original Hochreiter & Schmidhuber paper and in virtually every textbook and survey since (commonly using Alex Graves' 2013 notation as the modern standard reference point). Two things worth flagging for reading papers cold: the bracket notation $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ always means **concatenation** into one longer vector, not any kind of multiplication or special operator — it's purely a notational convenience so one weight matrix $W_f$ can be written instead of two separate matrices for $\mathbf{h}_{t-1}$ and $\mathbf{x}_t$ each. And seeing three or four near-identical-looking equations differing only in their subscript ($f$, $i$, $g$, $o$) and which nonlinearity they use is the standard way gated architectures are presented — recognizing "these are all just gates, computed the same structural way, doing different jobs" is most of the pattern-matching needed to read any gated-RNN variant (and, as later modules will show, gating shows up again in modern architectures well beyond LSTMs specifically).
 
## 4. Code: building an LSTM cell, verifying it, then measuring the actual improvement
 
**4.1 From scratch — a full LSTM cell forward pass**
 
```python
import numpy as np
 
def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh(z): return np.tanh(z)
 
class ManualLSTMCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        concat_size = input_size + hidden_size
        scale = np.sqrt(1 / concat_size)
        self.Wf = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.Wi = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.Wg = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.Wo = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.bf = np.ones((hidden_size, 1))    # forget gate bias -- init to 1: "remember by default"
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
 
    def step(self, x_t, h_prev, c_prev):
        concat = np.vstack([h_prev, x_t])
        f_t = sigmoid(self.Wf @ concat + self.bf)
        i_t = sigmoid(self.Wi @ concat + self.bi)
        g_t = tanh(self.Wg @ concat + self.bg)
        o_t = sigmoid(self.Wo @ concat + self.bo)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)
        return h_t, c_t, f_t
```
 
**4.2 Verifying the manual implementation against PyTorch's `LSTMCell`**
 
Transferring the manual cell's weights directly into a `torch.nn.LSTMCell` and comparing outputs step by step:
 
```
step 0: manual h=[-0.0735  0.0348  0.0518 -0.0507]
        torch  h=[-0.0735  0.0348  0.0518 -0.0507]   match? True
step 1: manual h=[ 0.1141  0.1572  0.0562 -0.0179]
        torch  h=[ 0.1141  0.1572  0.0562 -0.0179]   match? True
step 2: manual h=[ 0.1695  0.1746  0.0154 -0.0010]
        torch  h=[ 0.1695  0.1746  0.0154 -0.0010]   match? True
```
 
Exact agreement across three timesteps — the manual implementation is computing precisely what PyTorch's production LSTM implementation computes, gate for gate.
 
**4.3 The actual payoff — measuring gradient preservation, LSTM vs. vanilla RNN**
 
```python
import torch.nn as nn
 
def run_rnn(T, hidden_size=20):
    cell = nn.RNNCell(5, hidden_size, nonlinearity='tanh')
    x_seq = [torch.randn(1, 5, requires_grad=True) for _ in range(T)]
    h = torch.zeros(1, hidden_size)
    for t in range(T): h = cell(x_seq[t], h)
    h.sum().backward()
    return [x.grad.norm().item() for x in x_seq]
 
def run_lstm(T, hidden_size=20):
    cell = nn.LSTMCell(5, hidden_size)
    x_seq = [torch.randn(1, 5, requires_grad=True) for _ in range(T)]
    h, c = torch.zeros(1, hidden_size), torch.zeros(1, hidden_size)
    for t in range(T): h, c = cell(x_seq[t], (h, c))
    h.sum().backward()
    return [x.grad.norm().item() for x in x_seq]
 
for T in [15, 30, 50]:
    torch.manual_seed(0); rnn_grads = run_rnn(T)
    torch.manual_seed(0); lstm_grads = run_lstm(T)
    print(f"T={T}: RNN ratio(early/late)={rnn_grads[0]/rnn_grads[-1]:.3e}   LSTM ratio={lstm_grads[0]/lstm_grads[-1]:.3e}")
```
```
T=15: RNN ratio=3.763e-05   LSTM ratio=8.917e-04
T=30: RNN ratio=2.408e-10   LSTM ratio=4.427e-07
T=50: RNN ratio=2.687e-16   LSTM ratio=3.716e-11
```
 
Read this precisely, without overclaiming: **the LSTM's gradient still shrinks with sequence length — it does not eliminate the vanishing gradient problem entirely.** But at every tested length, it preserves gradient magnitude several orders of magnitude better than the vanilla RNN — roughly $10^{5}$ times better at $T=50$ specifically. This matches §2.2's honest claim exactly: the LSTM provides an *available* near-identity path through the cell state, not a guarantee that gradients survive perfectly forever. It's a substantial, measured improvement — not a complete cure, which is itself an important, accurate thing to know before assuming an LSTM alone solves long-range dependency problems.
 
## 5. Where this leaves us
 
LSTMs (and GRUs) fixed enough of the vanishing gradient problem to make genuinely useful sequence modeling practical — they were the dominant architecture for machine translation, speech recognition, and language modeling for roughly two decades. But §4.3's own numbers hint at the ceiling: even with gating, very long sequences still degrade gradient flow substantially. And there's a second limitation neither RNNs nor LSTMs address at all: processing is inherently **sequential** — timestep 50 cannot be computed until timestep 49 finishes, which fundamentally limits parallelization, however good the gradient math is.
 
## 6. Before Episode 04.02
 
> Episode 01.02, back in Module 01, built a primitive contextualization mechanism — attention — using nothing but static embeddings and a similarity-weighted average, with no recurrence at all: every word attended directly to every other word in the sentence, all at once, in one matrix operation. Given everything just learned about *why* gradients vanish or survive across long chains of sequential computation, what advantage might a mechanism with **no sequential chain at all** — where any two positions, however far apart, are connected by a single direct computation rather than a chain of intermediate timesteps — have over even the best-gated recurrent architecture?
 
That's the on-ramp into Episode 04.02 — returning to attention, this time building it as a fully trainable layer with learned query/key/value projections, and seeing exactly why it displaced recurrence entirely.
 
---
 
**Previous:** Episode 04.00 — Recurrent Neural Networks and the Limits of Memory
**Next:** Episode 04.02 — Trainable Attention: Learned Queries, Keys, and Values