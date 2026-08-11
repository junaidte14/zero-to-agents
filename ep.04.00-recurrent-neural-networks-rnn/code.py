#Code: building an RNN from scratch, verifying BPTT exactly, then measuring the vanishing gradient directly

#1 From scratch — forward pass and full BPTT

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

#2 Verifying against PyTorch, on an identical unrolled computation

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

#3 Measuring the vanishing gradient across time, directly

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