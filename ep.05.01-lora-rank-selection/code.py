#Code: predicting the exact loss floor before training, then confirming it

#1 Constructing a task with a known, exact true rank

import torch
import torch.nn as nn

torch.manual_seed(0)
d, true_r = 32, 10
W_pretrained = torch.randn(d, d) * 0.1
delta_W_true = (torch.randn(d, true_r) * 0.1) @ (torch.randn(true_r, d) * 0.1)  # exactly rank 10

X = torch.randn(500, d)
Y = X @ (W_pretrained + delta_W_true).T

U, S, Vt = torch.linalg.svd(delta_W_true)
print("Singular values of the true update:", S.numpy().round(3))

#2 The Eckart-Young prediction, computed before any LoRA training happens

def eckart_young_floor(S, k):
    return (S[k:]**2).sum().item()   # sum of squared DISCARDED singular values

#3 Training LoRA at several ranks, and comparing against the prediction

def train_lora(r, epochs=800, lr=0.02):
    torch.manual_seed(1)
    A = nn.Parameter(torch.randn(r, d) * 0.01)
    B = nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(epochs):
        pred = X @ W_pretrained.T + X @ A.T @ B.T
        loss = ((pred - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

print(f"{'r':>4} {'trained loss':>15} {'Eckart-Young floor':>20}")
for r in [1, 2, 5, 10, 15, 20]:
    trained_loss = train_lora(r)
    floor = eckart_young_floor(S, r)
    print(f"{r:4d} {trained_loss:15.6f} {floor:20.6f}")


