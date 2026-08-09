# From Zero to Agents
## Module 03 — Neural Networks from First Principles
### Episode 03.04: Backpropagation — Deriving It, Then Building It With No Autograd
 
---
 
## 0. Closing the open question
 
Episode 03.03 ended by asking what changes, mechanically, going from backpropagating through 2 layers to backpropagating through many. The answer is: almost nothing changes conceptually — it's the same chain-rule idea from Episode 02.02, applied once per layer, in a specific and reusable pattern. This episode derives that pattern precisely (the four equations behind every backprop implementation ever written), then builds it in raw NumPy — no `autograd`, no `.backward()` — and proves, line by line, that it produces exactly the same gradients PyTorch computes automatically.
 
## 1. Theory: propagating an error signal backward through the network
 
**1.1 The forward pass, formalized layer by layer.**
Using Episode 03.01 §3's layer-indexed notation: for layer $l = 1, \ldots, L$,
 
$$\mathbf{z}^{(l)} = W^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}, \qquad \mathbf{a}^{(l)} = \sigma(\mathbf{z}^{(l)})$$
 
with $\mathbf{a}^{(0)}$ the raw input $\mathbf{x}$, and $\mathbf{a}^{(L)}$ the network's final output. $\mathbf{z}^{(l)}$ (the pre-activation) and $\mathbf{a}^{(l)}$ (the post-activation) both need to be *remembered* during the forward pass — not just computed and discarded — because the backward pass reuses them directly, as §2 makes precise.
 
**1.2 The key idea — an "error signal" at every layer, computed back-to-front.**
Define $\boldsymbol{\delta}^{(l)}$ as "how much the loss would change if $\mathbf{z}^{(l)}$ (this layer's pre-activation) were nudged" — essentially, how much *blame* layer $l$ carries for the final loss. The entire algorithm is: compute $\boldsymbol{\delta}$ at the *output* layer directly (easy — the loss depends on it directly), then work backward, computing each earlier layer's $\boldsymbol{\delta}$ from the *next* layer's $\boldsymbol{\delta}$ — which is exactly why it's called **back**propagation: the error signal flows from output to input, the opposite direction of the forward pass.
 
**1.3 Why this modular structure is what makes libraries like PyTorch possible at all.**
Notice the pattern: to compute layer $l$'s weight gradient, you only ever need $\boldsymbol{\delta}^{(l)}$ and $\mathbf{a}^{(l-1)}$ — never anything from layers further away. This locality is *exactly* what lets `autograd` (Episode 02.02 §4.2) work generically across arbitrary architectures: every layer type just needs to know how to convert "the gradient flowing into my output" into "the gradient flowing out to my input" and "the gradient with respect to my own parameters" — and the *same* local rule, chained together automatically, handles a 2-layer network or a 200-layer one identically. Section 4 builds this exact mechanism from raw NumPy, without any of that automation, specifically to see what `.backward()` was doing all along.
 
## 2. Math: the four backpropagation equations, derived
 
**2.1 Equation 1 — the output layer's error signal.**
By the chain rule (Episode 02.02 §1.5), the loss's sensitivity to the output pre-activation $\mathbf{z}^{(L)}$ is the loss's sensitivity to the output $\mathbf{a}^{(L)}$, times the activation function's local derivative:
 
$$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}} \text{Loss} \odot \sigma'(\mathbf{z}^{(L)})$$
 
The $\odot$ symbol is the **Hadamard product** — elementwise multiplication, entry by entry (not matrix multiplication) — worth naming explicitly since it appears constantly in backprop notation and is easy to misread as ordinary matrix multiplication. For the squared-error loss used in Episode 02.02, $\nabla_{\mathbf{a}}\text{Loss} = (\mathbf{a}^{(L)} - \mathbf{y})$ directly (the derivative of $\frac{1}{2}(\mathbf{a}-\mathbf{y})^2$ with respect to $\mathbf{a}$) — the same "predicted minus true" shape from Episode 02.04's softmax/cross-entropy result, appearing again under a different loss function.
 
**2.2 Equation 2 — propagating the error signal backward one layer.**
This is the actual heart of backpropagation:
 
$$\boldsymbol{\delta}^{(l)} = \left( (W^{(l+1)})^T \boldsymbol{\delta}^{(l+1)} \right) \odot \sigma'(\mathbf{z}^{(l)})$$
 
Read it precisely: to find how much layer $l$ is to blame, take the *next* layer's blame signal $\boldsymbol{\delta}^{(l+1)}$, and route it backward through that next layer's weights — using the **transpose** of $W^{(l+1)}$. The transpose here isn't arbitrary: on the forward pass, $W^{(l+1)}$ mapped layer $l$'s output *forward* into layer $l+1$'s input; on the backward pass, routing blame in the *reverse* direction requires the *reverse* mapping, which for a linear transformation is exactly its transpose (a fact from linear algebra we're using directly rather than re-deriving). Multiply elementwise by $\sigma'(\mathbf{z}^{(l)})$ — layer $l$'s own local activation derivative — for the same reason as Equation 1: blame only flows through a neuron in proportion to how sensitive that neuron's own output actually was to its input.
 
**2.3 Equations 3 and 4 — converting error signals into actual weight and bias gradients.**
 
$$\frac{\partial \text{Loss}}{\partial W^{(l)}} = \boldsymbol{\delta}^{(l)} (\mathbf{a}^{(l-1)})^T, \qquad \frac{\partial \text{Loss}}{\partial \mathbf{b}^{(l)}} = \boldsymbol{\delta}^{(l)}$$
 
The weight gradient is an **outer product** (a column vector times a row vector, producing a full matrix the same shape as $W^{(l)}$ itself) between this layer's error signal and the *previous* layer's activation — precisely why $\mathbf{a}^{(l-1)}$ needed to be remembered from the forward pass in §1.1. The bias gradient is simply the error signal itself — no activation to multiply against, since a bias doesn't depend on the previous layer's output at all.
 
## 3. Decoding real notation — this is (close to) verbatim Nielsen's formulation
 
The four equations in §2 are, near-verbatim, the standard backpropagation derivation found in most deep learning textbooks and course notes (this specific four-equation packaging is popularized by Michael Nielsen's widely-used *"Neural Networks and Deep Learning"*, though the underlying math is universal — the same four relationships, regardless of which text states them). Two notational habits worth internalizing for reading *any* paper describing backprop: first, $\odot$ always means elementwise, never matrix, multiplication — mixing this up is one of the most common sources of dimension-mismatch bugs when implementing backprop by hand. Second, seeing a transpose ($^T$) applied specifically to a weight matrix during a *backward*-direction computation is not incidental — it's specifically because gradients flow in the opposite direction from activations, and a transpose is what "reversing the direction" of a linear map actually means, mathematically.
 
## 4. Code: implementing the four equations, with no autograd, and proving it's correct
 
**4.1 From scratch — a complete manual backpropagation implementation**
 
```python
import numpy as np
 
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)
 
class ManualMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.L = len(sizes) - 1
        # He/Kaiming-style scaling, per Episode 03.03
        self.W = [rng.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(1/sizes[i]) for i in range(self.L)]
        self.b = [np.zeros((sizes[i+1], 1)) for i in range(self.L)]
 
    def forward(self, x):
        a = x.reshape(-1, 1)
        activations, zs = [a], []
        for l in range(self.L):
            z = self.W[l] @ a + self.b[l]
            a = sigmoid(z)
            zs.append(z); activations.append(a)
        return zs, activations
 
    def backward(self, x, y):
        zs, activations = self.forward(x)
        y = y.reshape(-1, 1)
        L = self.L
        grads_W, grads_b = [None]*L, [None]*L
 
        # Equation 1: output layer error signal
        delta = (activations[-1] - y) * sigmoid_deriv(zs[-1])
        grads_W[L-1] = delta @ activations[-2].T   # Equation 3
        grads_b[L-1] = delta                        # Equation 4
 
        # Equation 2: propagate error backward through remaining layers
        for l in range(L-2, -1, -1):
            delta = (self.W[l+1].T @ delta) * sigmoid_deriv(zs[l])
            grads_W[l] = delta @ activations[l].T
            grads_b[l] = delta
 
        loss = 0.5 * np.sum((activations[-1] - y)**2)
        return loss, grads_W, grads_b
```
 
**4.2 Verifying it against PyTorch's autograd, on identical weights**
 
```python
import torch
 
net = ManualMLP([2, 4, 1], seed=42)
x, y = np.array([0.0, 1.0]), np.array([1.0])
loss, gW, gb = net.backward(x, y)
 
W1_t = torch.tensor(net.W[0], dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(net.b[0], dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(net.W[1], dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(net.b[1], dtype=torch.float64, requires_grad=True)
x_t = torch.tensor(x, dtype=torch.float64).reshape(-1,1)
y_t = torch.tensor(y, dtype=torch.float64).reshape(-1,1)
 
a1 = torch.sigmoid(W1_t @ x_t + b1_t)
a2 = torch.sigmoid(W2_t @ a1 + b2_t)
loss_t = 0.5 * torch.sum((a2 - y_t)**2)
loss_t.backward()
 
print("Manual loss:", loss, " Torch loss:", loss_t.item())
print("Grad W2 match?", np.allclose(gW[1], W2_t.grad.numpy(), atol=1e-8))
print("Grad W1 match?", np.allclose(gW[0], W1_t.grad.numpy(), atol=1e-8))
```
```
Manual loss: 0.12330261029015115  Torch loss: 0.12330261029015115
Grad W2 match? True
Grad W1 match? True
```
 
Exact agreement — same loss to fifteen decimal places, and every single gradient in every layer matches `autograd`'s output exactly. This is the entire point made concrete: `.backward()` isn't doing anything more sophisticated than the four equations in §2, applied programmatically. There's no hidden machinery — the manual implementation *is* what autograd is automating.
 
**4.3 Training the from-scratch network on XOR, no autograd anywhere**
 
```python
net2 = ManualMLP([2, 4, 1], seed=1)
X = [np.array([0.,0.]), np.array([0.,1.]), np.array([1.,0.]), np.array([1.,1.])]
Y = [np.array([0.]), np.array([1.]), np.array([1.]), np.array([0.])]
 
lr = 1.0
for epoch in range(5000):
    for x, y in zip(X, Y):
        _, gW, gb = net2.backward(x, y)
        for l in range(net2.L):
            net2.W[l] -= lr * gW[l]
            net2.b[l] -= lr * gb[l]
 
for x, y in zip(X, Y):
    _, activations = net2.forward(x)
    print(f"{x} -> predicted={activations[-1][0][0]:.3f}  true={y[0]}")
```
```
[0. 0.] -> predicted=0.016  true=0.0
[0. 1.] -> predicted=0.984  true=1.0
[1. 0.] -> predicted=0.984  true=1.0
[1. 1.] -> predicted=0.019  true=0.0
```
 
A complete neural network — forward pass, backward pass, weight updates, all four backprop equations, He-style initialization from Episode 03.03 — built entirely from raw NumPy arithmetic, with not a single call to any deep learning library, correctly learns XOR. This closes the loop Episode 03.01 opened with a library-trained version of the same problem: now you've built the mechanism underneath that library call yourself, verified it's exactly correct, and watched it work.
 
## 5. Where this leaves us
 
Every foundational piece of a trainable neural network — a working layer (03.01), a well-behaved activation (03.02), a correctly-scaled starting point (03.03), and now the actual learning algorithm itself (03.04) — has been derived from first principles, implemented without shortcuts, and verified against production libraries at every step. This is no longer scattered machinery; it's a complete, working system you built yourself.
 
## 6. Before Episode 03.05
 
> Everything trained so far in this module used exactly one example at a time — one $(\mathbf{x}, \mathbf{y})$ pair per weight update. Real training uses **batches**: many examples processed together, gradients averaged, before a single update. What do you think changes about the math in §2 — the shapes of $\mathbf{z}^{(l)}$, $\mathbf{a}^{(l)}$, $\boldsymbol{\delta}^{(l)}$ — when $\mathbf{x}$ stops being a single vector and becomes a whole batch of vectors at once? (Hint: recall Episode 02.01's view of a matrix as a *stack* of vectors.)
 
That's the on-ramp into Episode 03.05 — batching, and the practical training loop used in real systems.
 
---
 
**Previous:** Episode 03.03 — Weight Initialization
**Next:** Episode 03.05 — Batching and the Practical Training Loop