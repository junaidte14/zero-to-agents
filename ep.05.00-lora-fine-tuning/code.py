#Code: parameter counts confirmed, the initialization subtlety verified, and LoRA matched against full fine-tuning

#1 The parameter-count reduction, computed directly

d, r = 4096, 8
full_finetune_params = d * d
lora_params = d * r + r * d
print(f"Full fine-tuning: {full_finetune_params:,} params")
print(f"LoRA (r={r}):        {lora_params:,} params")
print(f"Reduction: {full_finetune_params/lora_params:.1f}x")

#2 Confirming the zero-init property

import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=8):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.1, requires_grad=False)
        self.A = nn.Parameter(torch.randn(r, in_features) * 0.1)   # random init
        self.B = nn.Parameter(torch.zeros(out_features, r))         # ZERO init
        self.scaling = alpha / r

    def forward(self, x):
        return x @ self.W.T + self.scaling * (x @ self.A.T @ self.B.T)

layer = LoRALinear(16, 16, r=4, alpha=8)
x = torch.randn(3, 16)
print("At init, LoRA output == frozen-base-only output?",
      torch.allclose(layer(x), x @ layer.W.T))

#3 Verifying the subtle gradient sequence from §2.4

x = torch.randn(5, 16)
target = torch.randn(5, 16)
optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

for step in range(3):
    loss = ((layer(x) - target)**2).mean()
    optimizer.zero_grad(); loss.backward()
    print(f"step {step}: |dA|={layer.A.grad.norm():.6f}  |dB|={layer.B.grad.norm():.6f}  |B|={layer.B.norm():.6f}")
    optimizer.step()

#4 LoRA matched against full fine-tuning, on a task with genuinely low intrinsic rank

d = 32
W_pretrained = torch.randn(d, d) * 0.1
true_r = 2
W_target = W_pretrained + (torch.randn(d, true_r)*0.1) @ (torch.randn(true_r, d)*0.1)  # a low-rank task shift

X = torch.randn(200, d)
Y = X @ W_target.T

# Full fine-tuning: every entry of W trainable
W_full = nn.Parameter(W_pretrained.clone())
opt = torch.optim.Adam([W_full], lr=0.01)
for _ in range(300):
    loss = ((X @ W_full.T - Y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Full fine-tuning: final loss={loss.item():.6f}  params={d*d}")

# LoRA: only A, B trainable, rank matched to the task's true rank
A = nn.Parameter(torch.randn(true_r, d) * 0.01)
B = nn.Parameter(torch.zeros(d, true_r))
opt = torch.optim.Adam([A, B], lr=0.01)
for _ in range(300):
    loss = ((X @ W_pretrained.T + X @ A.T @ B.T - Y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"LoRA (r={true_r}):    final loss={loss.item():.6f}  params={d*true_r*2}")

