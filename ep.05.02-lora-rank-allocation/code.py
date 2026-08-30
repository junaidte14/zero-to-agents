#Code: computing the optimal allocation, then confirming it against real training

#1 Two matrices, deliberately given very different true intrinsic ranks

import torch
import torch.nn as nn

torch.manual_seed(0)
d = 24
rank_Q_true, rank_V_true = 3, 12   # W_Q needs little adaptation; W_V needs much more

W_Q_pretrained, W_V_pretrained = torch.randn(d, d)*0.1, torch.randn(d, d)*0.1
delta_Q_true = (torch.randn(d, rank_Q_true)*0.1) @ (torch.randn(rank_Q_true, d)*0.1)
delta_V_true = (torch.randn(d, rank_V_true)*0.1) @ (torch.randn(rank_V_true, d)*0.1)

_, S_Q, _ = torch.linalg.svd(delta_Q_true)
_, S_V, _ = torch.linalg.svd(delta_V_true)

def eckart_young_floor(S, k):
    return (S[k:]**2).sum().item()

#2 Searching for the budget-optimal split

total_budget = 12   # e.g., "12 total rank, split however is best, between these two matrices"

best_split, best_total_floor = None, float('inf')
for r_Q in range(total_budget + 1):
    r_V = total_budget - r_Q
    total_floor = eckart_young_floor(S_Q, r_Q) + eckart_young_floor(S_V, r_V)
    if total_floor < best_total_floor:
        best_total_floor, best_split = total_floor, (r_Q, r_V)

equal_split = (total_budget // 2, total_budget // 2)
equal_floor = eckart_young_floor(S_Q, equal_split[0]) + eckart_young_floor(S_V, equal_split[1])
print(f"Equal split {equal_split}: predicted total floor = {equal_floor:.6f}")
print(f"Best split  {best_split}: predicted total floor = {best_total_floor:.6f}")

#3 Confirming with real, actual LoRA training — not just the theoretical prediction

X = torch.randn(400, d)
Y_Q = X @ (W_Q_pretrained + delta_Q_true).T
Y_V = X @ (W_V_pretrained + delta_V_true).T

def train_pair(r_Q, r_V, epochs=600, lr=0.02):
    torch.manual_seed(2)
    A_Q, B_Q = nn.Parameter(torch.randn(r_Q, d)*0.01), nn.Parameter(torch.zeros(d, r_Q))
    A_V, B_V = nn.Parameter(torch.randn(r_V, d)*0.01), nn.Parameter(torch.zeros(d, r_V))
    opt = torch.optim.Adam([A_Q, B_Q, A_V, B_V], lr=lr)
    for _ in range(epochs):
        pred_Q = X @ W_Q_pretrained.T + X @ A_Q.T @ B_Q.T
        pred_V = X @ W_V_pretrained.T + X @ A_V.T @ B_V.T
        loss = ((pred_Q-Y_Q)**2).sum()/X.shape[0] + ((pred_V-Y_V)**2).sum()/X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

print("Equal split  actual loss:", train_pair(*equal_split))
print("Best split   actual loss:", train_pair(*best_split))