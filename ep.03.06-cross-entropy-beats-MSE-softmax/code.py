#verified against autograd, then the pathology measured directly

#1 From scratch — a classifier with softmax output, cross-entropy loss

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

#2 Verifying against PyTorch's cross_entropy, on a 3-class toy problem

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

#3 The pathology, measured directly

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