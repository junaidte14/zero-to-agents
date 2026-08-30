#Code: testing the naive claim directly, then finding the real, precise result

#1 Setting up a genuine sequential two-task scenario

import torch
import torch.nn as nn

torch.manual_seed(0)
d = 32
W_A_target = torch.randn(d, d) * 0.1                 # Task A's ideal transformation
X_A, Y_A = torch.randn(300, d), None
Y_A = X_A @ W_A_target.T

delta_B = (torch.randn(d, 6)*0.1) @ (torch.randn(6, d)*0.1)   # Task B needs a further, rank-6 shift
W_B_target = W_A_target + delta_B
X_B = torch.randn(300, d)
Y_B = X_B @ W_B_target.T

def train_full(W_init, X, Y, epochs=500, lr=0.02):
    W = nn.Parameter(W_init.clone())
    opt = torch.optim.Adam([W], lr=lr)
    for _ in range(epochs):
        loss = ((X @ W.T - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach()

W_after_A = train_full(torch.randn(d, d) * 0.1, X_A, Y_A)
loss_A_baseline = ((X_A @ W_after_A.T - Y_A)**2).sum().item() / X_A.shape[0]
print(f"Task A loss right after training on Task A: {loss_A_baseline:.6f}")

#2 Full fine-tuning on Task B, then re-measuring Task A

W_after_B_full = train_full(W_after_A, X_B, Y_B)
loss_A_after_full = ((X_A @ W_after_B_full.T - Y_A)**2).sum().item() / X_A.shape[0]
print(f"Task A loss AFTER full fine-tuning on Task B: {loss_A_after_full:.6f}")

#3 LoRA on Task B, tested both attached and detached

def train_lora(W_frozen, X, Y, r, epochs=500, lr=0.02):
    A = nn.Parameter(torch.randn(r, d)*0.01)
    B = nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(epochs):
        pred = X @ W_frozen.T + X @ A.T @ B.T
        loss = ((pred - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return A.detach(), B.detach()

A_lora, B_lora = train_lora(W_after_A, X_B, Y_B, r=6)

loss_A_detached = ((X_A @ W_after_A.T - Y_A)**2).sum().item() / X_A.shape[0]
pred_attached = X_A @ W_after_A.T + X_A @ A_lora.T @ B_lora.T
loss_A_attached = ((pred_attached - Y_A)**2).sum().item() / X_A.shape[0]

print(f"Task A loss, adapter DETACHED: {loss_A_detached:.6f}")
print(f"Task A loss, adapter ATTACHED: {loss_A_attached:.6f}")

