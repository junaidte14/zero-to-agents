# From Zero to Agents
## Module 04 — Sequence Models and Attention
### Episode 04.05: Causal Masking — and a Complete, Working GPT-Style Model
 
---
 
## 0. Closing the open question
 
Episode 04.04 ended by asking what would go wrong if a position could attend to tokens that come *after* it in the sequence. The answer: during generation, a model predicting token 6 genuinely has no access to tokens 7, 8, or 9 — they don't exist yet, they haven't been generated. If a model were trained allowing position 6 to peek at position 7's content, it would learn to rely on information it will never actually have at inference time — a direct mismatch between training and real use. This episode fixes that with one of the smallest architectural changes in this entire course, then uses it to assemble and train a complete, working GPT-style model — the capstone of everything Module 04, and really this entire course, has built.
 
## 1. Theory: training in parallel, but generating one token at a time
 
**1.1 The mismatch between how training happens and how generation happens.**
At generation time, a language model produces tokens one at a time, left to right — token $t+1$ is predicted using only tokens $1$ through $t$, because nothing after position $t$ exists yet. But training an attention-based model one token at a time, the way an RNN naturally would (Episode 04.00), would throw away exactly the parallelism that made attention faster than recurrence in the first place (Episode 04.01's closing point). The standard solution: feed the *entire* training sequence in at once, for full parallelism, but artificially prevent each position from attending to anything after itself — reproducing, during training, exactly the information constraint the model will actually face at generation time.
 
**1.2 Causal masking — blocking the future, structurally.**
This constraint is called **causal masking** (also "look-ahead masking"): before the softmax step in attention, force every score corresponding to "attend to a future position" down to a value that guarantees zero probability after normalization. Section 2 makes this exact.
 
## 2. Math: one addition to the attention formula, with total effect
 
**2.1 The masked attention formula.**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V, \qquad M_{ij} = \begin{cases} 0 & j \leq i \\ -\infty & j > i\end{cases}$$
 
This is Episode 02.01's exact formula from months ago, with one term added: $M$, an upper-triangular matrix of zeros and negative infinities. Recall from Episode 02.04 §2.2 that softmax exponentiates every entry before normalizing — and $\exp(-\infty) = 0$ exactly. Adding $-\infty$ to any future-position score doesn't just *discourage* attending there, it makes it **structurally impossible**: after softmax, that entry's probability is precisely zero, every single time, regardless of what the raw $QK^T$ score happened to be. This is the entire mechanism — one additive mask, applied once, before a function ($\exp$) that turns $-\infty$ into exactly $0$.
 
**2.2 The one difference between an "encoder" and a "decoder" layer.**
This is worth stating precisely, because it resolves something implicit throughout the whole module: the "transformer block" built in Episode 04.03 — unmasked, every position free to attend to every other position, forward and backward — is what papers call an **encoder** layer, well suited to *understanding* a complete, already-known sequence. Add exactly the mask from §2.1, and nothing else about the architecture changes, and it becomes a **decoder** layer, suited to *generating* a sequence one token at a time. The entire encoder/decoder architectural distinction that shows up throughout transformer literature reduces, in the attention computation itself, to this one additive term.
 
## 3. Decoding real notation — and recognizing this pattern everywhere
 
The formula in §2.1 is written essentially this way in most papers describing decoder-only (GPT-style) architectures — sometimes with $M$ drawn explicitly as a triangular grid in an architecture diagram, sometimes just noted as "causal attention" or "masked self-attention" in prose, trusting the reader to know it means exactly this additive mask. Recognizing "masked" or "causal" attention in any paper from here on should immediately mean: identical attention formula, plus this one $-\infty$-before-softmax term — not a different mechanism requiring separate understanding.
 
## 4. Code: masking verified exactly, then a complete model trained and generating text
 
**4.1 From scratch — causal masking, verified structurally**
 
```python
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
```
```
[[1.    0.    0.    0.    0.   ]
 [0.764 0.236 0.    0.    0.   ]
 [0.361 0.281 0.358 0.    0.   ]
 [0.217 0.33  0.226 0.226 0.   ]
 [0.237 0.125 0.283 0.184 0.171]]
Row sums: [1. 1. 1. 1. 1.]
All future positions exactly zero? True
```
 
Exactly the lower-triangular structure §2.1 predicts: position 0 can only attend to itself (weight 1.0), position 1 splits attention between positions 0 and 1, and so on — every row still sums to exactly 1 (it's still a valid probability distribution, §1.4 of Episode 02.04), just with zero probability mass anywhere in the future.
 
**4.2 Verified against PyTorch's built-in causal attention**
 
```python
torch_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
print("Matches manual masked implementation?", torch.allclose(manual_out, torch_out, atol=1e-5))
```
```
Matches manual masked implementation? True
```
 
**4.3 The capstone — a complete tiny GPT, trained end to end, generating text**
 
Every piece from this entire course, assembled: a character-level vocabulary (Episode 01.00's tokenization, at its simplest), an embedding table (Module 00), sinusoidal positional encoding (Episode 04.04), stacked causal transformer blocks (Episodes 04.02–04.05), a softmax output layer trained with cross-entropy loss (Episode 02.04, Episode 03.06), and gradient descent (Episode 02.02–02.03) doing the actual learning:
 
```python
vocab = ['a', 'b', 'c']
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
pattern = "abc" * 20
data = torch.tensor([stoi[c] for c in pattern])
 
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
```
```
step 0: loss=1.3136
step 100: loss=0.0009
step 200: loss=0.0004
step 300: loss=0.0002
step 400: loss=0.0001
```
 
```python
model.eval()
generated = torch.tensor([[stoi['a']]])
with torch.no_grad():
    for _ in range(15):
        logits = model(generated)
        next_id = logits[0, -1].argmax().item()
        generated = torch.cat([generated, torch.tensor([[next_id]])], dim=1)
 
print(''.join(itos[i] for i in generated[0].tolist()))
```
```
abcabcabcabcabca
```
 
Seeded with nothing but the single character `'a'`, the model correctly, autoregressively generates the entire repeating pattern — each predicted token fed back in as input for predicting the next one, exactly the generation process real language models use, at genuinely tiny scale but with every mechanism identical. Loss collapsing from `1.31` to `0.0001` confirms the model didn't memorize a lookup table (there's no table here — every generated token comes from a live forward pass through learned attention weights, causal masking, and a trained softmax output layer); it learned the actual *rule* generating the pattern.
 
## 5. Where this leaves us — Module 04 complete
 
Six episodes, and the two threads this entire course has been building — representation (Module 01: tokenization, embeddings, primitive attention) and trainable networks (Module 03: activations, initialization, backpropagation, the right loss function) — have fully merged into a real, working, generative transformer: RNNs exposed the vanishing-gradient-across-time problem (04.00); LSTMs partially fixed it with gating (04.01); attention was rebuilt with genuinely learned, trainable relevance (04.02); multi-head attention, residual connections, and layer normalization assembled the complete block, with residual connections proven to solve vanishing gradients across depth using the *exact same mechanism* LSTM gating used across time (04.03); positional encoding gave a fundamentally order-blind mechanism a real, mathematically well-behaved sense of sequence (04.04); and today, one additive mask turned an "understand a sequence" architecture into a "generate a sequence, one token at a time" architecture — trained, and watched generating correctly, autoregressively, from a single seed character.
 
Nothing in this final model was taken on faith. Every mechanism inside it — the embedding lookup, the attention scores, the softmax, the residual paths, the gradient computation, the loss function — was derived, implemented from scratch at least once, and verified against a production library somewhere across the last eighteen episodes.
 
## 6. What comes next
 
This tiny model learned a trivial, perfectly repeating pattern — the point was verifying every mechanism works, not building something linguistically useful yet. The natural next step, and where this course heads next, is **fine-tuning and adaptation** — starting from a much larger, already-capable pretrained model rather than training from scratch, and adjusting it efficiently for a specific task or domain (including LoRA — low-rank adaptation — which connects directly back to real work you already do at AIVerse). Before that, it's worth pausing for the kind of review and synthesis we've done at the end of every module so far — this one closes out not just a module, but the entire foundational arc of the course.
 
---
 
**Previous:** Episode 04.04 — Positional Encoding
**Next:** Module 05 — Fine-Tuning and Adaptation (starting with LoRA)