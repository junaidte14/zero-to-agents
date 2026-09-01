import math
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set random seed for consistent transformer weight initialization
torch.manual_seed(42)

# ==========================================
# 0. Core Vector Math & Data Setup
# ==========================================

past_turns = [
    "user asked how to open a new bank account for savings",
    "user asked about hiking trails along the river bank",
    "user asked about shipping times to canada",
    "user asked to cancel a subscription plan",
]

query = "user needs help with their bank"
stopwords = {"user", "asked", "their", "to", "about", "needs", "help", "with", "how", "a", "for"}

print("new query being tested: user needs help with their bank")

# Vocabulary indexing
all_text = " ".join(past_turns + [query]).lower().split()
vocab = sorted(list(set(all_text)))
stoi = {w: i for i, i_w in enumerate(vocab) for w, i in [(i_w, i)]}

def cosine(v1, v2):
    """Computes cosine similarity between two 1D PyTorch tensors or lists."""
    if isinstance(v1, list):
        v1 = torch.tensor(v1, dtype=torch.float32)
    if isinstance(v2, list):
        v2 = torch.tensor(v2, dtype=torch.float32)
    
    norm1 = torch.norm(v1)
    norm2 = torch.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return (torch.dot(v1, v2) / (norm1 * norm2)).item()

# ==========================================
# 1. Static vs Contextual Model Setup
# ==========================================

d_model = 16
vocab_size = len(vocab)

# Static Embeddings: Standard lookup table (Word2Vec/GloVe equivalent)
static_embedding = nn.Embedding(vocab_size, d_model)

# Contextual Model: Single-layer Self-Attention Encoder (Transformer equivalent)
class MiniContextualEncoder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.mha = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, idx):
        x = self.emb(idx)  # (1, Seq_Len, d_model)
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + attn_out)

contextual_model = MiniContextualEncoder(vocab_size, d_model)
contextual_model.eval()

# ==========================================
# Vectorization Functions
# ==========================================

def static_turn_vec(text, stop_words):
    """Computes mean static embedding across non-stopword tokens."""
    tokens = [w for w in text.lower().split() if w not in stop_words and w in stoi]
    if not tokens:
        return torch.zeros(d_model)
    indices = torch.tensor([stoi[w] for w in tokens])
    return static_embedding(indices).mean(dim=0).detach()

def contextual_turn_vec(text, stop_words):
    """
    Computes contextualized sequence embeddings via self-attention,
    then pools non-stopword token representations.
    """
    words = text.lower().split()
    indices = torch.tensor([[stoi[w] for w in words if w in stoi]])
    
    with torch.no_grad():
        ctx_embs = contextual_model(indices)[0]  # (Seq_Len, d_model)
    
    # Filter out positions belonging to stopwords
    valid_indices = [i for i, w in enumerate(words) if w not in stop_words]
    if not valid_indices:
        return ctx_embs.mean(dim=0).detach()
        
    return ctx_embs[valid_indices].mean(dim=0).detach()

# ==========================================
# #1 Static Retrieval
# ==========================================

print("=== #1 Static Retrieval (Weakened but Correct) ===")
qv_static = static_turn_vec(query, stopwords)
static_results = []

for t in past_turns:
    s = cosine(qv_static, static_turn_vec(t, stopwords))
    static_results.append((t, s))

static_results.sort(key=lambda x: -x[1])
for t, s in static_results:
    print(f"  sim={s:.3f}  {t}")

margin_static = static_results[0][1] - static_results[1][1]
print(f"\nMargin, STATIC: {margin_static:.3f}")

print("\n" + "="*60 + "\n")

# ==========================================
# #2 Contextual Retrieval
# ==========================================

print("=== #2 Contextual Retrieval (Anisotropy & Context Bleed) ===")
qv_ctx = contextual_turn_vec(query, stopwords)
ctx_results = []

for t in past_turns:
    s = cosine(qv_ctx, contextual_turn_vec(t, stopwords))
    ctx_results.append((t, s))

ctx_results.sort(key=lambda x: -x[1])
for t, s in ctx_results:
    print(f"  sim={s:.3f}  {t}")

margin_ctx = ctx_results[0][1] - ctx_results[1][1]
print(f"\nMargin, CONTEXTUAL: {margin_ctx:.3f} -- pointing at the WRONG answer entirely")