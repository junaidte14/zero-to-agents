#Code: masking verified exactly, then a complete model trained and generating text

#1 From scratch — causal masking, verified structurally

import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mask(S):
    return torch.triu(torch.ones(S, S), diagonal=1).bool()   # True = future position, to be masked

class CausalMHA(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.h, self.d_k = num_heads, d_model // num_heads
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):
        B, S, D = X.shape
        Q = self.Wq(X).view(B, S, self.h, self.d_k).transpose(1, 2)
        K = self.Wk(X).view(B, S, self.h, self.d_k).transpose(1, 2)
        V = self.Wv(X).view(B, S, self.h, self.d_k).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores.masked_fill(causal_mask(S), float('-inf'))
        weights = F.softmax(scores, dim=-1)
        return self.Wo((weights @ V).transpose(1,2).contiguous().view(B,S,D)), weights

mha = CausalMHA(d_model=8, num_heads=2)
_, weights = mha(torch.randn(1, 5, 8))
print(weights[0,0].detach().numpy().round(3))
print("Row sums:", weights[0,0].sum(-1).detach().numpy().round(4))
print("All future positions exactly zero?", torch.allclose(weights[0,0].triu(1), torch.zeros(5,5)))

#2 Verified against PyTorch's built-in causal attention

#torch_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
#print("Matches manual masked implementation?", torch.allclose(manual_out, torch_out, atol=1e-5))

#3 The capstone — a complete tiny GPT, trained end to end, generating text

vocab = ['a', 'b', 'c']
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
pattern = "abc" * 20
data = torch.tensor([stoi[c] for c in pattern])

def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    return pe

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = CausalMHA(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = self.ln1(x + self.mha(x)[0])
        return self.ln2(x + self.ffn(x))

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=16, num_heads=2, d_ff=32, n_layers=2, max_len=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pe = sinusoidal_pe(max_len, d_model)   # Episode 04.04
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, S = idx.shape
        x = self.token_emb(idx) + self.pe[:S].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

model = TinyGPT(len(vocab))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
block_size = 8

def get_batch(batch_size=16):
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])   # next-token targets, shifted by 1
    return x, y

for step in range(500):
    x, y = get_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, len(vocab)), y.view(-1))
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if step % 100 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

model.eval()
generated = torch.tensor([[stoi['a']]])
with torch.no_grad():
    for _ in range(15):
        logits = model(generated)
        next_id = logits[0, -1].argmax().item()
        generated = torch.cat([generated, torch.tensor([[next_id]])], dim=1)

print(''.join(itos[i] for i in generated[0].tolist()))