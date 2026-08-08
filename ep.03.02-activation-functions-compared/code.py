#the derivatives, verified, and the vanishing gradient measured directly

#1 From scratch — all three activations and their derivatives, verified against autograd

import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

zs = np.array([-5, -2, -0.5, 0, 0.5, 2, 5], dtype=float)
print("sigmoid'(z):", np.round(sigmoid_deriv(zs), 4))
print("max possible sigmoid':", sigmoid_deriv(np.array([0.0]))[0])

#using library

import torch
zt = torch.tensor(zs, requires_grad=True)
torch.sigmoid(zt).sum().backward()
print("torch gradient:", np.round(zt.grad.numpy(), 4))
print("matches manual derivative?", np.allclose(zt.grad.numpy(), sigmoid_deriv(zs), atol=1e-4))


#2 Measuring vanishing gradients directly, in a real 15-layer network

import torch.nn as nn

def build_deep_net(activation_cls, n_layers=15, width=20):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(width, width), activation_cls()]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)

def report(name, activation_cls):
    torch.manual_seed(0)
    net = build_deep_net(activation_cls)
    x = torch.randn(1, 20)
    loss = (net(x) - torch.tensor([[1.0]]))**2
    loss.backward()
    grad_norms = [m.weight.grad.norm().item() for m in net if isinstance(m, nn.Linear)]
    print(f"{name}: layer 0 (near input) grad = {grad_norms[0]:.2e}   layer 15 (near output) grad = {grad_norms[-1]:.2e}")
    print(f"  ratio (input-layer / output-layer): {grad_norms[0]/grad_norms[-1]:.2e}")

report("Sigmoid, 15 layers", nn.Sigmoid)
report("ReLU, 15 layers", nn.ReLU)