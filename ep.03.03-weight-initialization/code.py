#both failure modes demonstrated, then the fix, verified against PyTorch

#1 From scratch — the symmetry problem, watched failing to break

import torch
import torch.nn as nn

torch.manual_seed(0)
net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
with torch.no_grad():
    net[0].weight.fill_(0.3)   # every weight in layer 1 IDENTICAL (not even zero -- the point is symmetry, not zero)
    net[2].weight.fill_(0.5)   # every weight in layer 2 also identical

x = torch.randn(1, 4)
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
for _ in range(20):
    loss = (net(x) - torch.tensor([[1.0]]))**2
    optimizer.zero_grad(); loss.backward(); optimizer.step()

rows = net[0].weight.data
print("Layer 1 neurons after 20 training steps:")
print(rows)
print("Still identical to each other?", all(torch.allclose(rows[0], rows[i], atol=1e-6) for i in range(1, 4)))

#2 Verifying the variance formula from §2.1, empirically

import numpy as np

rng = np.random.default_rng(0)
n = 50   # fan-in
trials = 200000

y_samples = [np.sum(rng.normal(0, 1, n) * rng.normal(0, 1, n)) for _ in range(trials)]
print("Empirical Var(y):", np.var(y_samples))
print("Analytical n * Var(w) * Var(x):", n * 1.0 * 1.0)

#3 The real payoff — activation scale across 15 layers, three initialization choices

def build_net(n_layers=15, width=100):
    layers = []
    for _ in range(n_layers):
        layers += [nn.Linear(width, width, bias=False), nn.ReLU()]
    return nn.Sequential(*layers)

def report(name, std):
    torch.manual_seed(0)
    net = build_net()
    with torch.no_grad():
        for m in net:
            if isinstance(m, nn.Linear):
                m.weight.normal_(0, std)
    activations = torch.randn(1, 100)
    print(f"\n{name} (std={std:.4f}):")
    for layer in net:
        activations = layer(activations)
        if isinstance(layer, nn.ReLU):
            print(f"  std = {activations.std().item():.6g}")

fan_in = 100
report("Too small", 0.01)
report("He/Kaiming (sqrt(2/fan_in))", (2.0/fan_in)**0.5)
report("Too large", 0.5)

#4 Confirming against PyTorch's built-in initializers

layer = nn.Linear(100, 100, bias=False)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
print("torch kaiming_normal_ std:", layer.weight.std().item())
print("He formula sqrt(2/fan_in):", (2.0/100)**0.5)