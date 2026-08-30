#Code: optimal quantization derived, then the real accuracy trade-off measured

#1 Uniform vs. optimal (Lloyd-Max) quantization, on realistic Gaussian-distributed weights

import torch

torch.manual_seed(0)
W = torch.randn(10000) * 0.5   # roughly-Gaussian, like real trained weights

def uniform_quantize(x, bits=4):
    levels = 2**bits
    xmin, xmax = x.min(), x.max()
    scale = (xmax - xmin) / (levels - 1)
    q = torch.clamp(torch.round((x - xmin) / scale), 0, levels - 1)
    return q * scale + xmin

def lloyd_max_quantize(x, bits=4, iters=30):
    levels = 2**bits
    centers = torch.quantile(x, torch.linspace(0.05, 0.95, levels))   # reasonable starting point
    for _ in range(iters):
        assignment = (x.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
        for k in range(levels):
            mask = assignment == k
            if mask.sum() > 0:
                centers[k] = x[mask].mean()
    assignment = (x.unsqueeze(1) - centers.unsqueeze(0)).abs().argmin(dim=1)
    return centers[assignment]

mse_uniform = ((W - uniform_quantize(W))**2).mean().item()
mse_optimal = ((W - lloyd_max_quantize(W))**2).mean().item()
print(f"4-bit uniform quantization MSE: {mse_uniform:.6f}")
print(f"4-bit optimal (Lloyd-Max) MSE:  {mse_optimal:.6f}")
print(f"Improvement: {mse_uniform/mse_optimal:.2f}x lower error")

#2 The honest accuracy trade-off — measured, not assumed

d, true_r = 32, 8
W_pretrained = torch.randn(d, d) * 0.1
delta_true = (torch.randn(d, true_r)*0.1) @ (torch.randn(true_r, d)*0.1)
X = torch.randn(400, d)
Y = X @ (W_pretrained + delta_true).T

W_quantized = uniform_quantize(W_pretrained, bits=4)   # simulate storing the frozen base at 4-bit

def train_lora(W_base, r, epochs=600, lr=0.02):
    torch.manual_seed(3)
    A, B = torch.nn.Parameter(torch.randn(r, d)*0.01), torch.nn.Parameter(torch.zeros(d, r))
    opt = torch.optim.Adam([A, B], lr=lr)
    for _ in range(epochs):
        pred = X @ W_base.T + X @ A.T @ B.T
        loss = ((pred - Y)**2).sum() / X.shape[0]
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

print("Full-precision base, LoRA r=8:", train_lora(W_pretrained, true_r))
print("Quantized base,      LoRA r=8:", train_lora(W_quantized, true_r))
print("Quantized base, NO LoRA at all:", ((X @ W_quantized.T - Y)**2).sum().item()/X.shape[0])
for r in [8, 16, 24, 32]:
    print(f"Quantized base, LoRA r={r:2d}: loss={train_lora(W_quantized, r):.6f}")

