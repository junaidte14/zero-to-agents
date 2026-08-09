#implementing the four equations, with no autograd, and proving it's correct

#1 From scratch — a complete manual backpropagation implementation

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

#2 Verifying it against PyTorch's autograd, on identical weights

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

#3 Training the from-scratch network on XOR, no autograd anywhere

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


