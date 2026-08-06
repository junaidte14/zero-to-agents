# From Zero to Agents
## Module 03 — Neural Networks from First Principles
### Episode 03.01: Multi-Layer Perceptrons — Solving the Problem That Broke a Single Neuron
 
---
 
## 0. Closing the open question
 
Episode 03.00 ended with a geometric question: if one perceptron draws exactly one straight line, what happens when you feed the *outputs* of several perceptrons into another perceptron as its input? Today we build the answer by hand, prove it works, prove *why* it needed to work that way, and then — for the first time in this course — train a network to find a solution on its own, via the exact gradient descent machinery built in Module 02.
 
## 1. Theory: composing simple pieces into something more expressive
 
**1.1 The geometric idea.**
A single perceptron carves input space into two regions with one straight cut. XOR's positive cases sit on opposite corners of a square — no single cut separates them from the negative cases. But two *different* straight cuts, considered together, can isolate the right region: one line separating "at least one input is 1" from "neither is," and a second line separating "not both inputs are 1" from "both are." The region satisfying *both* conditions simultaneously — at least one is 1, AND not both are 1 — is exactly XOR's positive region. A **multi-layer perceptron (MLP)** formalizes this: a **hidden layer** of several perceptrons, each drawing its own line, feeding into a further perceptron that combines their outputs.
 
**1.2 The classical XOR decomposition.**
$$\text{XOR}(x_1, x_2) = \text{AND}\big(\text{OR}(x_1,x_2),\ \text{NAND}(x_1,x_2)\big)$$
 
Check this logically before any weights: XOR is true exactly when the inputs differ. OR is true whenever at least one input is 1 — true for $(0,1)$, $(1,0)$, and $(1,1)$. NAND ("not and") is true whenever *not both* inputs are 1 — true for $(0,0)$, $(0,1)$, $(1,0)$. The only cases where **both** OR and NAND are true simultaneously are $(0,1)$ and $(1,0)$ — precisely XOR's two positive cases. Three perceptrons, each individually solving a linearly separable problem (OR, NAND, and AND are all solvable — Episode 03.00 §2.2 already proved AND works, and OR/NAND are structurally identical), combine into something none of them could do alone.
 
## 2. Math: building it, and proving why nonlinearity is load-bearing, not optional
 
**2.1 Explicit weights, verified against the truth table.**
Hidden neuron 1 (OR): $w=(1,1)$, $b=-0.5$. Hidden neuron 2 (NAND): $w=(-1,-1)$, $b=1.5$. Output neuron (AND of the two hidden outputs): $w=(1,1)$, $b=-1.5$ — the exact same AND weights proven correct in Episode 03.00 §2.2, reused unchanged. In matrix form, this is a two-layer network:
 
$$\mathbf{h} = \text{step}(W_1 \mathbf{x} + \mathbf{b}_1), \qquad y = \text{step}(W_2 \mathbf{h} + b_2)$$
 
with $W_1 = \begin{pmatrix} 1 & 1 \\ -1 & -1\end{pmatrix}$, $\mathbf{b}_1 = (-0.5, 1.5)$, $W_2 = (1, 1)$, $b_2 = -1.5$ — precisely the matrix-vector operation from Episode 02.01, applied twice in sequence. Section 4 verifies this produces correct XOR output on all four inputs.
 
**2.2 The critical theoretical point — why the activation function in between can't be skipped.**
It's tempting to think stacking layers helps regardless of what happens between them. It doesn't — and this is provable directly from Module 02's matrix algebra. Suppose the hidden layer used **no** nonlinearity — just $\mathbf{h} = W_1\mathbf{x}+\mathbf{b}_1$ directly. Substituting into the output layer:
 
$$y = W_2(W_1\mathbf{x}+\mathbf{b}_1) + b_2 = (W_2 W_1)\mathbf{x} + (W_2\mathbf{b}_1 + b_2)$$
 
Look at the right-hand side: $W_2 W_1$ is just some matrix (matrix-matrix multiplication, Episode 02.01 §2.3), and $W_2\mathbf{b}_1+b_2$ is just some vector. **The entire two-layer computation collapses algebraically into a single linear layer** — $y = W'\mathbf{x}+b'$ for some combined $W', b'$. No amount of stacking additional *linear* layers escapes this; any depth of purely linear layers is mathematically indistinguishable from one linear layer, and therefore exactly as limited as a single perceptron — still only able to draw one straight decision boundary, no matter how many layers deep. **The nonlinear activation function between layers is not a minor implementation detail — it's the entire reason depth adds any expressive power at all.** This is why every practical layer in every network from here forward pairs a linear transformation with a genuinely nonlinear activation.
 
**2.3 A forward pointer — how far this actually generalizes.**
The construction in §2.1 was hand-designed for one specific function. It turns out this generalizes enormously: the **Universal Approximation Theorem** (Cybenko, 1989; Hornik, 1991) proves that a network with even a single sufficiently-wide hidden layer, using a suitable nonlinear activation, can approximate *any* continuous function on a bounded input region to arbitrary precision. We won't prove this rigorously here — it's a substantial result on its own — but it's worth knowing it exists: XOR isn't a special case that happened to have a clever decomposition. It's a specific instance of a far more general capability that multi-layer networks with nonlinearities possess.
 
## 3. Decoding real notation — the layer-indexed forward pass
 
Papers describing deep networks almost universally write the forward pass with a superscript layer index:
 
$$\mathbf{h}^{(l)} = \sigma\left(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$
 
Decoded: $\mathbf{h}^{(l)}$ is the output of layer $l$ (with $\mathbf{h}^{(0)}$ conventionally meaning the raw input $\mathbf{x}$); $W^{(l)}$ and $\mathbf{b}^{(l)}$ are *that specific layer's* weights and bias, distinct from every other layer's; $\sigma$ (sigma) is a placeholder for whatever nonlinear activation function is being used (sigmoid, ReLU, and others — the specific choice varies by architecture and era, but the notation's *shape* doesn't). This single line, applied repeatedly for $l=1,2,\ldots,L$, is the complete forward pass of an arbitrarily deep network — our two-layer XOR network from §2.1 is exactly this equation, unrolled for $L=2$, with $\sigma = \text{step}$.
 
## 4. Code: hand-built solution, the linear-collapse proof, and a real trained network
 
**4.1 From scratch — the hand-constructed XOR network, verified**
 
```python
import numpy as np
 
def step(z): return np.where(z >= 0, 1, 0)
 
W1 = np.array([[1, 1], [-1, -1]])   # row 0: OR weights, row 1: NAND weights
b1 = np.array([-0.5, 1.5])
W2 = np.array([1, 1])                # AND weights
b2 = -1.5
 
def forward(x):
    h = step(W1 @ x + b1)
    return h, step(W2 @ h + b2)
 
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])
 
for x, yt in zip(X, y_xor):
    h, y = forward(x)
    print(f"{x}  hidden=OR:{h[0]},NAND:{h[1]}  predicted={y}  true={yt}")
```
```
[0 0]  hidden=OR:0,NAND:1  predicted=0  true=0
[0 1]  hidden=OR:1,NAND:1  predicted=1  true=1
[1 0]  hidden=OR:1,NAND:1  predicted=1  true=1
[1 1]  hidden=OR:1,NAND:0  predicted=0  true=0
```
 
All four correct — the exact XOR truth table Episode 03.00 proved a single perceptron could never reproduce, solved with zero training, purely by hand-composing three simple linear separators.
 
**4.2 Proving the linear-collapse claim from §2.2, numerically**
 
```python
np.random.seed(0)
W1r, b1r = np.random.randn(3,2), np.random.randn(3)
W2r, b2r = np.random.randn(2,3), np.random.randn(2)
 
def two_layer_linear(x):
    h = W1r @ x + b1r           # NOTE: no activation function here
    return W2r @ h + b2r
 
W_combined = W2r @ W1r
b_combined = W2r @ b1r + b2r
def one_layer_equiv(x): return W_combined @ x + b_combined
 
x_test = np.array([0.5, -1.2])
print("Two linear layers:", two_layer_linear(x_test))
print("Equivalent single layer:", one_layer_equiv(x_test))
```
```
Two linear layers: [3.463 3.126]
Equivalent single layer: [3.463 3.126]
```
 
Identical output, for randomly generated weights, confirming §2.2 isn't a special case of our particular XOR weights — it's a structural fact about linear layers in general. Two "layers" with no nonlinearity between them are provably, exactly, indistinguishable from one.
 
**4.3 Using real training — learning XOR via gradient descent, no hand-set weights**
 
```python
import torch
import torch.nn as nn
 
torch.manual_seed(42)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([[0.],[1.],[1.],[0.]])
 
model = nn.Sequential(nn.Linear(2,4), nn.Sigmoid(), nn.Linear(4,1), nn.Sigmoid())
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.BCELoss()
 
for epoch in range(3000):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss={loss.item():.4f}")
 
with torch.no_grad():
    for xi, yi, p in zip(X, y, model(X)):
        print(f"{xi.tolist()} -> predicted={p.item():.3f}  true={yi.item()}")
```
```
epoch    0  loss=0.7723
epoch  500  loss=0.6321
epoch 1000  loss=0.1307
epoch 1500  loss=0.0290
epoch 2000  loss=0.0147
epoch 2500  loss=0.0095
 
[0.0, 0.0] -> predicted=0.006  true=0.0
[0.0, 1.0] -> predicted=0.991  true=1.0
[1.0, 0.0] -> predicted=0.995  true=1.0
[1.0, 1.0] -> predicted=0.008  true=0.0
```
 
No OR/NAND/AND was ever specified. The network started from random weights, and pure gradient descent — the exact mechanism from Episode 02.02, applied automatically through two stacked layers via the chain rule (autograd's `.backward()`, Episode 02.02 §4.2) — found *a* working solution entirely on its own, converging to predictions within 1% of the true XOR values. This is the moment this course has been building toward since Module 02 opened: a network learning a genuinely non-linear function, from data, with no hand-engineering.
 
## 5. Where this leaves us
 
Episode 03.00 proved a hard ceiling on what one linear unit can represent. Today's episode proved that ceiling lifts the moment you combine multiple units through a genuine nonlinearity — not empirically observed, but derived: linear layers alone collapse to one layer regardless of depth (§2.2, confirmed numerically), while nonlinear layers compose into something strictly more expressive (§2.1, hand-verified; Universal Approximation Theorem, §2.3, as the general statement). And §4.3 showed, for the first time in this course, a network finding its own solution via gradient descent rather than one we designed by hand.
 
## 6. Before Episode 03.02
 
> Section 4.3 used `nn.Sigmoid()` as the nonlinearity. Section 2.4 of Episode 03.00 explained why the step function is *unusable* for gradient descent — its derivative is zero almost everywhere. Sigmoid clearly works (the training run converged). But is sigmoid actually a good choice, or just *a* choice that happens to be differentiable? Consider what sigmoid looks like far from zero — very large positive or negative inputs. What does its slope do out there, and what might that mean for how gradients behave in a network with many layers stacked deep?
 
That question is the on-ramp into Episode 03.02 — activation functions properly compared: sigmoid, tanh, and ReLU, and the vanishing-gradient problem that shaped which one actually won out in practice.
 
---
 
**Previous:** Episode 03.00 — The Perceptron and the XOR Problem
**Next:** Episode 03.02 — Activation Functions and the Vanishing Gradient Problem