import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for reproducibility
torch.manual_seed(42)

# ==========================================
# 0. Core Model Definition (from Ep 04.05)
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
    def __init__(self, vocab_size, d_model=32, num_heads=2, d_ff=64, n_layers=2, max_len=32):
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

# ==========================================
# #1 Extended Vocabulary and Traces
# ==========================================

# Digits 0-9 map directly to IDs 0-9
# Control and tool tokens follow:
QADD, QDBL, QCHAIN = 10, 11, 12
TADD, TDBL = 13, 14
A, O, Fi, EOS, PAD = 15, 16, 17, 18, 19
vocab_size = 20

def add_tool(a, b): return a + b
def double_tool(a): return min(a * 2, 9)

def trace_add(d1, d2):
    r = add_tool(d1, d2)
    return [QADD, d1, d2, A, TADD, d1, d2, O, r, Fi, r, EOS]

def trace_double(d1):
    r = double_tool(d1)
    return [QDBL, d1, A, TDBL, d1, O, r, Fi, r, EOS]

def trace_chain(d1, d2, d3):
    r1 = add_tool(d1, d2)
    r2 = add_tool(r1, d3)
    return [QCHAIN, d1, d2, d3, A, TADD, d1, d2, O, r1, A, TADD, r1, d3, O, r2, Fi, r2, EOS]

# Dataset Generation
train_add = [(d1, d2) for d1 in range(4) for d2 in range(4) if d1 + d2 <= 9][:10]
test_add  = [(d1, d2) for d1 in range(4) for d2 in range(4) if d1 + d2 <= 9][10:]

train_dbl = [0, 1, 2]
test_dbl  = [3, 4]

train_chain = [(d1, d2, d3) for d1 in range(3) for d2 in range(3) for d3 in range(3) if d1 + d2 + d3 <= 9][:12]
test_chain  = [(d1, d2, d3) for d1 in range(3) for d2 in range(3) for d3 in range(3) if d1 + d2 + d3 <= 9][12:]

def get_all_train_traces():
    traces = []
    for d1, d2 in train_add: traces.append(trace_add(d1, d2))
    for d1 in train_dbl: traces.append(trace_double(d1))
    for d1, d2, d3 in train_chain: traces.append(trace_chain(d1, d2, d3))
    return traces

# Batch creation with padding & loss masking
train_traces = get_all_train_traces()
max_len = max(len(t) for t in train_traces)

padded_inputs, padded_targets = [], []
for t in train_traces:
    inp = t[:-1] + [PAD] * (max_len - len(t))
    tgt = t[1:] + [PAD] * (max_len - len(t))
    
    # Mask initial prompt predictions
    prompt_len = 3 if t[0] in (QADD, QDBL) else 4
    for i in range(prompt_len - 1):
        tgt[i] = -100
    for i in range(len(tgt)):
        if tgt[i] == PAD:
            tgt[i] = -100
            
    padded_inputs.append(inp)
    padded_targets.append(tgt)

inputs = torch.tensor(padded_inputs)
masked_targets = torch.tensor(padded_targets)

# ==========================================
# Training
# ==========================================

model = TinyGPT(vocab_size=vocab_size, d_model=32, num_heads=2, d_ff=64, n_layers=2, max_len=32)
opt = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()
for step in range(3500):
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), masked_targets.reshape(-1), ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(f"step {step}: loss={loss.item():.4f}")

# ==========================================
# #2 The Extended Dynamic Agent Loop
# ==========================================

def run_loop(model, prompt_tokens, real_tool_fn_sequence, max_new=16):
    model.eval()
    with torch.no_grad():
        seq = torch.tensor([prompt_tokens])
        tool_idx, generated = 0, 0
        while generated < max_new:
            logits = model(seq)
            next_id = logits[0, -1].argmax().item()
            seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
            generated += 1
            if next_id == O and tool_idx < len(real_tool_fn_sequence):
                real_result = real_tool_fn_sequence[tool_idx]()
                seq = torch.cat([seq, torch.tensor([[real_result]])], dim=1)
                tool_idx += 1
                generated += 1
            if next_id == EOS:
                break
    return seq[0].tolist()

# ==========================================
# #3 Testing & Results
# ==========================================

# 1. ADD Tool Evaluation
correct_add = 0
for d1, d2 in test_add:
    res = run_loop(model, [QADD, d1, d2], [lambda d1=d1, d2=d2: add_tool(d1, d2)])
    if Fi in res and res[res.index(Fi) + 1] == add_tool(d1, d2):
        correct_add += 1

# 2. DOUBLE Tool Evaluation
correct_dbl = 0
for d1 in test_dbl:
    res = run_loop(model, [QDBL, d1], [lambda d1=d1: double_tool(d1)])
    if Fi in res and res[res.index(Fi) + 1] == double_tool(d1):
        correct_dbl += 1

# 3. CHAIN Evaluation (2-step execution using sequential outputs)
correct_chain = 0
for d1, d2, d3 in test_chain:
    # State tracking closure to supply the dynamic result of step 1 into step 2
    r1_box = []
    tool_fns = [
        lambda d1=d1, d2=d2: (r1_box.append(add_tool(d1, d2)), r1_box[-1])[1],
        lambda d3=d3: add_tool(r1_box[0], d3)
    ]
    res = run_loop(model, [QCHAIN, d1, d2, d3], tool_fns)
    expected = add_tool(add_tool(d1, d2), d3)
    if Fi in res and res[res.index(Fi) + 1] == expected:
        correct_chain += 1

print(f"ADD tool, unseen pairs: {correct_add}/{len(test_add)}")
print(f"DOUBLE tool, unseen value: {correct_dbl}/{len(test_dbl)}")
print(f"CHAIN (2-step), unseen triples: {correct_chain}/{len(test_chain)}")