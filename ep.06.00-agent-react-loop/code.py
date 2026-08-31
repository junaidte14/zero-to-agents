import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. Core Model Definition (from EP.04.05)
# ==========================================

def causal_mask(S):
    return torch.triu(torch.ones(S, S), diagonal=1).bool()

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
        return self.Wo((weights @ V).transpose(1, 2).contiguous().view(B, S, D)), weights

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
        self.pe = sinusoidal_pe(max_len, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, S = idx.shape
        x = self.token_emb(idx) + self.pe[:S].unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(x)

# Set seed for reproducibility
torch.manual_seed(42)

# ==========================================
# #1 Vocabulary & Trace Setup
# ==========================================

# Digits 0-4 are mapped directly to token IDs 0-4
DIGITS = [0, 1, 2, 3, 4]
Q, A, O, Fi, EOS = 5, 6, 7, 8, 9
vocab_size = 10

def add_tool(a, b): 
    return a + b

def make_trace(d1, d2):
    r = add_tool(d1, d2)
    # Trace: [Q, d1, d2, A, d1, d2, O, r, Fi, r, EOS]
    return torch.tensor([Q, d1, d2, A, d1, d2, O, r, Fi, r, EOS])

# ==========================================
# #2 Training Setup & Execution
# ==========================================

# Data Splits
all_pairs = [(d1, d2) for d1 in range(3) for d2 in range(3)] # pairs yielding sum <= 4
train_pairs = all_pairs[:7]
test_pairs = all_pairs[7:]

def build_batch(pairs):
    traces = [make_trace(d1, d2) for d1, d2 in pairs]
    batch = torch.stack(traces)
    inputs = batch[:, :-1]   # X: tokens 0..T-2
    targets = batch[:, 1:]   # Y: tokens 1..T-1
    return inputs, targets

inputs, targets = build_batch(train_pairs)

# Mask the prompt tokens (predicting the initial random inputs Q, d1, d2)
masked_targets = targets.clone()
masked_targets[:, :2] = -100

model = TinyGPT(vocab_size=vocab_size, d_model=32, num_heads=2, d_ff=64, n_layers=2)
opt = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for step in range(3000):
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), masked_targets.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

# ==========================================
# #3 The Real Agent Loop
# ==========================================

def run_agent_loop(model, d1, d2):
    model.eval()
    with torch.no_grad():
        # Start prompt with Q, d1, d2
        seq = torch.tensor([[Q, d1, d2]])
        
        # Step A: Model proposes action arguments after generating 'A'
        logits = model(seq)
        seq = torch.cat([seq, torch.tensor([[A]])], dim=1)
        
        # Generate the two action argument digits
        for _ in range(2):
            logits = model(seq)
            next_id = logits[0, -1].argmax().item()
            seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
            
        # Step B: Agent encounters 'O' marker, calls REAL Python tool, and appends observation
        seq = torch.cat([seq, torch.tensor([[O]])], dim=1)
        tool_result = add_tool(seq[0, 4].item(), seq[0, 5].item())
        seq = torch.cat([seq, torch.tensor([[tool_result]])], dim=1)
        
        # Step C: Resume text generation to output Final Answer (Fi) and result
        for _ in range(3):
            logits = model(seq)
            next_id = logits[0, -1].argmax().item()
            seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
            if next_id == EOS:
                break
                
    return seq[0].tolist()

# ==========================================
# #4 Testing on Unseen Pairs
# ==========================================

correct = 0
for d1, d2 in test_pairs:
    result_seq = run_agent_loop(model, d1, d2)
    f_idx = result_seq.index(Fi)
    predicted = result_seq[f_idx + 1]
    is_correct = predicted == (d1 + d2)
    correct += int(is_correct)

print(f"{correct}/{len(test_pairs)} correct on unseen tool-use tasks")