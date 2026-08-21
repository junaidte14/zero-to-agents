#Code: training real Q/K/V weights on a task raw embeddings cannot solve

#1 A task specifically designed so embedding similarity gives zero useful signal

import torch
import torch.nn as nn
import torch.nn.functional as F

d_model, d_k, seq_len = 8, 8, 5

def make_batch(batch_size):
    content = torch.randn(batch_size, seq_len, d_model - 1)
    flag_pos = torch.randint(0, seq_len, (batch_size,))
    flags = torch.zeros(batch_size, seq_len, 1)
    for b in range(batch_size):
        flags[b, flag_pos[b], 0] = 1.0
    X = torch.cat([content, flags], dim=-1)
    targets = content[torch.arange(batch_size), flag_pos]
    return X, targets, flag_pos

class LearnedAttention(nn.Module):
    def __init__(self, d_model, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_k, bias=False)
        self.Wk = nn.Linear(d_model, d_k, bias=False)
        self.Wv = nn.Linear(d_model, d_model - 1, bias=False)
        self.query_vec = nn.Parameter(torch.randn(1, d_model))  # a single learnable "what to look for"

    def forward(self, X):
        batch = X.shape[0]
        Q = self.Wq(self.query_vec).expand(batch, -1)
        K = self.Wk(X)
        V = self.Wv(X)
        scores = torch.einsum('bd,bsd->bs', Q, K) / (d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bs,bsd->bd', weights, V)
        return output, weights

#2 Attention before training — as expected, unfocused

torch.manual_seed(0)
untrained = LearnedAttention(d_model, d_k)
X, targets, flag_pos = make_batch(3)
with torch.no_grad():
    _, weights_before = untrained(X)
for i in range(3):
    print(f"example {i}: flagged pos={flag_pos[i].item()}  weights={weights_before[i].numpy().round(3)}")

#3 Training, and attention afterward

model = LearnedAttention(d_model, d_k)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(2000):
    X, targets, flag_pos = make_batch(32)
    output, weights = model(X)
    loss = F.mse_loss(output, targets)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 400 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

X, targets, flag_pos = make_batch(10)
with torch.no_grad():
    output, weights = model(X)
    for i in range(10):
        top = weights[i].argmax().item()
        print(f"example {i}: flagged pos={flag_pos[i].item()}  top-attended pos={top}  match={top==flag_pos[i].item()}")

#4 Confirming this is exactly what nn.MultiheadAttention implements internally

mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
X = torch.randn(3, 5, 8)
query = torch.randn(3, 1, 8)
output, attn_weights = mha(query, X, X)
print("Output shape:", output.shape)              # torch.Size([3, 1, 8])
print("Attention weights shape:", attn_weights.shape)  # torch.Size([3, 1, 5])
print("Total learnable parameters:", sum(p.numel() for p in mha.parameters()))  # 288

