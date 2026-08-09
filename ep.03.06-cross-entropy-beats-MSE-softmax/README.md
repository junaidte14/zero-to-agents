# From Zero to Agents
## Module 03 — Neural Networks from First Principles
### Episode 03.06: Classification Networks — Why Cross-Entropy Beats Squared Error
 
---
 
## 0. Closing the open question
 
Episode 03.05 ended by asking what changes in the four backprop equations to train with cross-entropy and softmax instead of squared error and sigmoid — and whether Episode 02.04's "predicted minus true" gradient simplification makes that change easier than expected. It does — dramatically so. This episode makes the swap precisely, and uncovers a real, provable pathology in the squared-error approach that's been quietly sitting in every network this module has built so far.
 
## 1. Theory: two loss functions, one hidden problem
 
**1.1 What every network in this module has used until now.**
Every XOR network built since Episode 03.01 used sigmoid output activations with squared-error loss — treating a 0/1 classification target as if it were a regression target to hit exactly. This works, as demonstrated repeatedly, but it's not what real classification systems use, and there's a specific, derivable reason why.
 
**1.2 The real setup for classification — softmax output, cross-entropy loss.**
Episode 02.04 built exactly this pairing: a softmax output layer producing a genuine probability distribution over classes (§1.4 there), and cross-entropy loss measuring how wrong that distribution is (§1.5 there). For $K$-class classification, the output layer produces $K$ numbers summing to 1, and the true label is represented as a one-hot vector — this is the actual setup behind essentially every classification network and every language model's next-token prediction (Episode 02.04 §3), not a special case reserved for "classification problems" narrowly defined.
 
**1.3 The pathology in sigmoid + squared error, named precisely.**
Here's the problem, stated plainly before the math: sigmoid saturates for extreme inputs (Episode 03.02 §1.1) — meaning its derivative shrinks toward zero exactly when the pre-activation is far from zero. But "pre-activation far from zero" is *exactly* what happens when a network is confidently, badly wrong — a strongly negative $z$ when the true label is 1, say. The squared-error output gradient (Episode 03.04 §2.1) multiplies the raw error $(a-y)$ by that same shrinking $\sigma'(z)$ — meaning **the correction signal gets weaker exactly when the network is most wrong and needs the strongest possible correction.** This isn't a minor inefficiency; it's a structural mismatch between what the loss function should be signaling and what it actually delivers, and Section 4 measures exactly how severe it is.
 
## 2. Math: deriving the fix, and why it has no such pathology
 
**2.1 The output-layer error signal for softmax + cross-entropy.**
Recall Episode 02.04 §2.4's result, already verified against autograd there: for softmax feeding directly into cross-entropy, the combined gradient with respect to the *pre-activation* logits collapses to
 
$$\boldsymbol{\delta}^{(L)} = \mathbf{a}^{(L)} - \mathbf{y}$$
 
with no extra activation-derivative term multiplied in at all — compare this directly against Episode 03.04 §2.1's $\boldsymbol{\delta}^{(L)} = (\mathbf{a}^{(L)}-\mathbf{y}) \odot \sigma'(\mathbf{z}^{(L)})$ for sigmoid + squared error. **The $\sigma'(\mathbf{z}^{(L)})$ term is simply absent.** This isn't a coincidence of algebra — the softmax function's own derivative and cross-entropy loss's own derivative cancel each other out exactly during the chain-rule combination (a genuinely elegant piece of calculus, verified numerically rather than re-derived symbolically here, consistent with how Episode 02.04 handled it). The practical consequence: **the correction signal is directly, linearly proportional to how wrong the prediction is — full stop, with no saturating term to weaken it, no matter how confidently wrong the network currently is.**
 
**2.2 Everything else in backpropagation stays exactly the same.**
This is worth stating explicitly: Episode 03.04's Equations 2, 3, and 4 — how error propagates backward through hidden layers, and how weight/bias gradients are computed from the error signal — are completely unchanged. Only the *output layer's* error-signal formula changes, from §2.1 above replacing Episode 03.04 §2.1's version. Everything built in Episodes 03.04 and 03.05 (hidden-layer backward recursion, batching) carries over untouched.
 
## 3. Decoding real notation — the shorthand you'll see in real implementations
 
Framework source code and papers implementing classification networks frequently state the final-layer gradient directly as a comment or a one-line formula: `dz = y_pred - y_true` (or equivalently $\partial L/\partial z = \hat{y}-y$) for the output layer, without re-deriving it each time. This is exactly §2.1's result, written in shorthand — recognizing this exact pattern on sight (it appeared already, independently derived, in Episode 02.04 §4.3, and now again here) is a genuinely high-value pattern-match: whenever softmax feeds directly into cross-entropy, expect this simplification, and expect it to be treated as a known, standard shortcut rather than re-derived from scratch in most technical writing.
 
## 4. Code: verified against autograd, then the pathology measured directly
 
**4.1 From scratch — a classifier with softmax output, cross-entropy loss**
 
```python
import numpy as np
 
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z):
    s = sigmoid(z); return s * (1 - s)
 
def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)
 
class ClassifierMLP:
    """Hidden layers: sigmoid. Output layer: softmax + cross-entropy."""
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.L = len(sizes) - 1
        self.W = [rng.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(1/sizes[i]) for i in range(self.L)]
        self.b = [np.zeros((sizes[i+1], 1)) for i in range(self.L)]
 
    def forward(self, X):
        A = X; activations, zs = [A], []
        for l in range(self.L - 1):
            Z = self.W[l] @ A + self.b[l]
            A = sigmoid(Z); zs.append(Z); activations.append(A)
        Z = self.W[-1] @ A + self.b[-1]
        A = softmax(Z); zs.append(Z); activations.append(A)
        return zs, activations
 
    def backward(self, X, Y):   # Y is one-hot, shape (n_classes, batch)
        m = X.shape[1]
        zs, activations = self.forward(X)
        L = self.L
        grads_W, grads_b = [None]*L, [None]*L
 
        delta = activations[-1] - Y                      # §2.1: NO sigmoid_deriv term
        grads_W[L-1] = (delta @ activations[-2].T) / m
        grads_b[L-1] = delta.sum(axis=1, keepdims=True) / m
 
        for l in range(L-2, -1, -1):                      # unchanged from Episode 03.04/03.05
            delta = (self.W[l+1].T @ delta) * sigmoid_deriv(zs[l])
            grads_W[l] = (delta @ activations[l].T) / m
            grads_b[l] = delta.sum(axis=1, keepdims=True) / m
 
        loss = -np.mean(np.sum(Y * np.log(activations[-1] + 1e-12), axis=0))
        return loss, grads_W, grads_b
```
 
**4.2 Verifying against PyTorch's `cross_entropy`, on a 3-class toy problem**
 
```python
import torch
 
net = ClassifierMLP([2, 4, 3], seed=42)
X = np.array([[0.1, 0.9, 0.5, 0.2, 0.8], [0.2, 0.8, 0.5, 0.9, 0.1]])
Y_labels = np.array([0, 1, 2, 1, 0])
Y = np.eye(3)[Y_labels].T
 
loss, gW, gb = net.backward(X, Y)
 
W1_t = torch.tensor(net.W[0], dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(net.b[0], dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(net.W[1], dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(net.b[1], dtype=torch.float64, requires_grad=True)
X_t = torch.tensor(X, dtype=torch.float64)
 
Z2 = W2_t @ torch.sigmoid(W1_t @ X_t + b1_t) + b2_t
loss_t = torch.nn.functional.cross_entropy(Z2.T, torch.tensor(Y_labels, dtype=torch.long))
loss_t.backward()
 
print("Loss match?", np.isclose(loss, loss_t.item()))
print("Output-layer grad match?", np.allclose(gW[1], W2_t.grad.numpy(), atol=1e-8))
print("Hidden-layer grad match?", np.allclose(gW[0], W1_t.grad.numpy(), atol=1e-8))
```
```
Loss match? True
Output-layer grad match? True
Hidden-layer grad match? True
```
 
Exact agreement again — same pattern established in Episode 03.04, now confirmed for the classification setup specifically.
 
**4.3 The pathology, measured directly**
 
```python
# Scenario: true label is 1, but the network is CONFIDENTLY wrong (predicted near 0)
z = -6.0
a = sigmoid(z)   # 0.0025 -- very confidently wrong
y = 1.0
 
delta_mse = (a - y) * sigmoid_deriv(z)          # sigmoid + squared-error correction signal
 
logits = np.array([0.0, z])                     # softmax + cross-entropy, same scenario
probs = softmax(logits)
delta_ce = probs - np.array([0.0, 1.0])
 
print(f"Predicted probability of the true class: {a:.4f}")
print(f"MSE+sigmoid correction signal:   {abs(delta_mse):.6f}")
print(f"Softmax+CE correction signal:    {abs(delta_ce[1]):.6f}")
print(f"CE signal is {abs(delta_ce[1])/abs(delta_mse):.1f}x stronger")
```
```
Predicted probability of the true class: 0.0025
MSE+sigmoid correction signal:   0.002460
Softmax+CE correction signal:    0.997527
CE signal is 405.4x stronger
```
 
This is §1.3's pathology, quantified exactly: when the network is confidently wrong (predicting the true class has only a 0.25% chance), sigmoid+squared-error's own saturation shrinks its correction signal down to nearly nothing — `0.0025`, barely distinguishable from zero. Softmax+cross-entropy's correction signal in the *identical* scenario is `0.998` — essentially maximal, because §2.1 proved this gradient has no saturating term to shrink it in the first place. A network trained with squared error in this state would take a very long time to correct itself, precisely because its own gradient tells it, misleadingly, that things are almost fine. A network trained with cross-entropy gets an immediate, proportionate, strong correction the moment it's confidently wrong — which is exactly the behavior you'd want from a loss function.
 
## 5. Where this leaves us
 
Two loss functions that look similar on the surface — both "measure prediction error" — turn out to behave completely differently in the regime that matters most: when the network is badly, confidently wrong early in training. This isn't a matter of taste or convention; it's a provable, measured, 400x difference in corrective signal strength in the scenario tested, arising directly from a specific cancellation in the calculus of softmax combined with cross-entropy. This is precisely why cross-entropy is the standard loss for classification and language modeling, not squared error — and now it's a derived fact rather than received wisdom.
 
## 6. Module 03 wrap-up
 
Seven episodes, one complete, working, self-verified neural network toolkit: a perceptron's provable ceiling (03.00) lifted by nonlinear multi-layer composition (03.01); saturating activations diagnosed and fixed (03.02); initialization variance derived and verified (03.03); full backpropagation built with zero autograd and matched exactly against PyTorch (03.04); batching shown to require no new mathematics, just matrices standing in for vectors (03.05); and today, the loss function real classification and language-modeling systems actually use, with its advantage over the naive alternative measured directly rather than assumed. Every claim in this module has been proven, implemented from scratch, and checked against a production library — the entire mechanism behind "a neural network learns" is no longer a black box anywhere in this stack.
 
## 7. What Module 04 covers
 
Module 04 — **Sequence Models and Attention** — is next, and it's where this course's two major threads finally merge: the representation work from Module 01 (tokenization, embeddings, the primitive attention preview in Episode 01.02) and the trainable-network machinery just completed in Module 03. We'll build recurrent networks first (to see why they were the pre-transformer default, and exactly where they struggle with long sequences), then construct the full transformer architecture — real learned query/key/value projections this time, not the reused static embeddings from Episode 02.01 — trained end to end with everything this course has built.
 
---
 
**Previous:** Episode 03.05 — Batching and the Practical Training Loop
**Next:** Module 04, Episode 04.00 — Recurrent Neural Networks and the Limits of Memory