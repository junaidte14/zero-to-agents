# From Zero to Agents
## Module 04 — Sequence Models and Attention
### Episode 04.03: Multi-Head Attention and the Transformer Block
 
---
 
## 0. Closing the open question
 
Episode 04.02 ended by asking what a network gains from several independently-trained attention heads over one larger single head. The core limitation of a single head: one softmax produces exactly one weighted-average blend per query — if two genuinely different relevance criteria both matter simultaneously (say, "who's the subject" and "what's the nearest modifier"), a single attention distribution has to compromise between them, blending two different signals into one blurry average, the same fundamental problem one-hot vectors had trying to represent two things with one number back in Episode 00.02. Multiple heads sidestep this by giving each criterion its *own* independent projection and its own independent softmax — no compromise required. This episode builds that, and then assembles the complete architectural block — attention, residual connections, and layer normalization — that gets stacked to build a full transformer.
 
## 1. Theory: three pieces, each fixing a specific problem
 
**1.1 Multi-head attention — parallel, independently-specialized attention.**
Split the work across $h$ separate "heads," each with its own learned $W_Q^i, W_K^i, W_V^i$ projecting down into a smaller subspace (typically $d_k = d_{\text{model}}/h$), run Episode 04.02's full attention mechanism independently in each subspace, then combine all $h$ heads' outputs back into one vector. Each head is free to specialize in whatever notion of relevance turns out to be useful, discovered independently through its own training — nothing forces two heads to learn the same thing, and nothing prevents them from learning to track completely different relationships in parallel.
 
**1.2 Residual connections — the direct architectural sibling of Episode 04.01's cell-state trick.**
A **residual connection** (or "skip connection") means the output of a sublayer isn't just $\text{Sublayer}(x)$, it's $x + \text{Sublayer}(x)$ — the original input is added directly back on, unchanged, alongside whatever the sublayer computed. This might look like a minor implementation detail. It isn't — Section 2 shows it's mathematically the *identical fix* Episode 04.01 used for LSTMs, now applied to arbitrarily deep feedforward stacks rather than long time sequences: a guaranteed, near-identity path for the gradient to flow backward through, independent of how deep the stack gets.
 
**1.3 Layer normalization — a runtime fix for the activation-scale problem from Episode 03.03.**
Episode 03.03 fixed unstable activation scale across depth by carefully choosing the *initial* weight variance. **Layer normalization** attacks the same underlying problem — activations growing or shrinking unpredictably across layers — with a runtime mechanism instead: at every layer, for *each token independently*, rescale that token's entire feature vector to have zero mean and unit variance, then apply a small learned rescaling ($\gamma$) and shift ($\beta$) so the network can still express whatever scale is actually useful. This is worth distinguishing precisely from **batch normalization** (a related, earlier technique): batch norm normalizes across the *batch* dimension — over many examples' values for one feature — while layer norm normalizes across the *feature* dimension — over one example's own values for every feature. This distinction is frequently a source of real confusion, and layer norm's per-example independence (no dependency on what else happens to be in the batch) is a large part of why it, not batch norm, is standard in transformer architectures.
 
## 2. Math: the identity-path argument, made precise, and layer norm formalized
 
**2.1 Why residual connections fix vanishing gradients — the direct calculation.**
For $y = x + F(x)$ (a sublayer $F$ with a residual connection), the derivative with respect to the input, using ordinary calculus rules for a sum:
 
$$\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$$
 
where $I$ is the identity matrix (the derivative of the "$+x$" term, unchanged, contributes exactly 1 along the diagonal, 0 elsewhere). **No matter how small or badly-behaved $\partial F/\partial x$ is** — even if the sublayer's own gradient has nearly vanished, exactly the failure mode from Episode 03.02 — the *total* derivative still contains that $I$ term, guaranteeing at least an identity-strength path for the gradient to flow through. Stack $N$ residual layers, and the gradient reaching the very first layer is a sum of many terms, at least one of which (the product of all the identity terms) survives completely undiminished regardless of depth. Compare this directly against Episode 04.01 §2.2's $\partial \mathbf{c}_t/\partial\mathbf{c}_{t-1}\approx \mathbf{f}_t$ — both mechanisms solve their respective vanishing-gradient problem (across *time* for LSTM's gate, across *depth* here) with the identical underlying trick: **provide an additive, near-identity path the gradient can take instead of being forced through a repeated multiply-and-squash chain.** This is not a coincidence of naming — it's the same mathematical idea, discovered independently for two different problems (LSTM: 1997; residual networks: He et al., 2015; applied inside transformers: Vaswani et al., 2017), unified here as one principle.
 
**2.2 Layer normalization, precisely.**
For a single token's activation vector $\mathbf{x} \in \mathbb{R}^{d}$ (one row — one position — of the full sequence):
 
$$\text{LN}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta, \qquad \mu = \frac{1}{d}\sum_{i=1}^d x_i, \quad \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2$$
 
$\mu$ and $\sigma^2$ are computed *across that one token's own $d$ features* — not across other tokens, not across the batch. $\epsilon$ is a tiny constant preventing division by zero if $\sigma^2$ happens to be near zero. $\gamma$ and $\beta$ (both learned, per-feature vectors) let the network undo the forced zero-mean-unit-variance normalization if a different scale actually turns out to be useful for a specific feature — normalization isn't a hard constraint on what the network can represent, just a well-behaved default it starts from at every layer.
 
## 3. Decoding real notation — the transformer encoder block, from the paper
 
Vaswani et al. (2017) describe one encoder block as two sublayers, each wrapped in a residual connection and followed by layer normalization:
 
$$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{MultiHead}(\mathbf{x})), \qquad \mathbf{x}'' = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$
 
$\text{FFN}$ (feed-forward network) is simply Episode 03.01's ordinary multi-layer perceptron — typically two linear layers with a nonlinearity between them — applied identically, independently, to *each token's* vector (the same small network, reused at every sequence position, exactly the weight-sharing idea from Episode 04.00 §1.2, just without any recurrence between positions this time). One notational detail worth flagging for reading other papers: the ordering above — normalize *after* the residual add — is called **"post-norm,"** the original paper's choice; many later architectures instead normalize the sublayer's input *before* it's processed (**"pre-norm"**, $\mathbf{x}+\text{Sublayer}(\text{LayerNorm}(\mathbf{x}))$) because it tends to make very deep stacks easier to train. Seeing either ordering in a paper's architecture diagram is describing the identical set of components (§1.1–§1.3), just sequenced slightly differently — recognizing this rather than treating it as an unfamiliar variant is most of the skill.
 
## 4. Code: every component built, verified, and the residual claim measured directly
 
**4.1 Multi-head attention, from scratch**
 
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
 
class MultiHeadAttention(nn.Module):
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
        weights = F.softmax(scores, dim=-1)
        heads_out = (weights @ V).transpose(1, 2).contiguous().view(B, S, D)
        return self.Wo(heads_out)
 
mha = MultiHeadAttention(d_model=8, num_heads=2)
out = mha(torch.randn(3, 5, 8))
print("Shape preserved?", out.shape == torch.Size([3, 5, 8]))   # True
```
 
**4.2 Layer normalization, verified against PyTorch**
 
```python
def manual_layernorm(x, gamma, beta, eps=1e-5):
    mu = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    return gamma * (x - mu) / torch.sqrt(var + eps) + beta
 
x = torch.randn(2, 5, 8) * 10 + 3     # deliberately large scale to make normalization visible
out_manual = manual_layernorm(x, torch.ones(8), torch.zeros(8))
print("Original per-token std:", x[0,0].std().item())            # 9.83
print("Post-LN per-token mean:", out_manual[0,0].mean().item())   # ~0
print("Post-LN per-token std:", out_manual[0,0].std(unbiased=False).item())  # 1.0
```
```
Original per-token std: 9.83
Post-LN per-token mean: -4.47e-08
Post-LN per-token std: 1.0
```
 
Confirmed against `torch.nn.LayerNorm` directly as well — identical output.
 
**4.3 The full transformer block, assembled**
 
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.ln2 = nn.LayerNorm(d_model)
 
    def forward(self, x):
        x = self.ln1(x + self.mha(x))   # residual + post-norm, per §3
        x = self.ln2(x + self.ffn(x))
        return x
 
block = TransformerBlock(d_model=16, num_heads=4, d_ff=32)
X = torch.randn(2, 6, 16)
out = block(X)
print("Output shape matches input?", out.shape == X.shape)   # True -- critical for stacking many blocks
```
```
Output shape matches input? True
```
 
Shape preservation isn't incidental — it's exactly what lets many transformer blocks be stacked one after another, each one's output feeding directly into the next as input, with no dimension mismatches anywhere.
 
**4.4 The residual claim, measured directly across real depth**
 
```python
class SubLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin, self.act = nn.Linear(d, d), nn.Tanh()
    def forward(self, x): return self.act(self.lin(x))
 
def measure(n_layers, residual):
    torch.manual_seed(0)
    layers = nn.ModuleList([SubLayer(16) for _ in range(n_layers)])
    x = torch.randn(1, 16, requires_grad=True)
    h = x
    for layer in layers:
        out = layer(h)
        h = h + out if residual else out
    h.sum().backward()
    return x.grad.norm().item()
 
for n in [5, 10, 20, 40]:
    print(f"depth={n:3d}  no residual: {measure(n, False):.3e}   with residual: {measure(n, True):.3e}")
```
```
depth=  5  no residual: 5.062e-02   with residual: 4.583e+00
depth= 10  no residual: 8.551e-03   with residual: 8.902e+00
depth= 20  no residual: 7.893e-06   with residual: 1.238e+01
depth= 40  no residual: 7.833e-12   with residual: 2.694e+01
```
 
By depth 40, the no-residual stack's gradient has shrunk to $7.8\times10^{-12}$ — Episode 03.02's vanishing gradient problem, replaying exactly, now across a plain deep stack rather than a specifically sigmoid-activated one. The residual version's gradient, at the *identical* depth, is a healthy $27$ — not just larger, but stable and usable, exactly as §2.1's $I + \partial F/\partial x$ argument predicted. Depth stopped being the enemy of gradient flow the moment an additive identity path was added — the same fix, mathematically, that Episode 04.01 used to rescue gradients across time.
 
## 5. Where this leaves us
 
Every piece of a real transformer block has now been built, verified against PyTorch component by component, and — critically — the *reason* each piece exists has been derived rather than asserted: multi-head attention solves the "one softmax can't represent two relevance criteria" limitation; residual connections solve vanishing gradients across depth using the identical mechanism LSTMs used across time; layer normalization solves the activation-scale problem Episode 03.03 first diagnosed, via a runtime mechanism instead of careful initialization alone. Stack several of these blocks — shape-preserving, so stacking is trivial — and the result is architecturally a real transformer encoder.
 
## 6. Module 04 checkpoint, and what's left
 
Four episodes into Module 04: RNNs exposed the vanishing-gradient-across-time problem (04.00); LSTMs partially fixed it with gating, measured honestly rather than overclaimed (04.01); attention was rebuilt with genuinely learned, trainable Q/K/V projections and proven capable of learning relevance signals static embeddings never could (04.02); and today, the complete architectural block — multi-head attention, residual connections, layer norm, feedforward — assembled and verified piece by piece.
 
## 7. Before Episode 04.04
 
> Every transformer block built today processes a sequence with **no notion of order at all** — attention computes pairwise relationships between positions using only their content, never their position in the sequence. Swap the order of two input tokens, and every $QK^T$ score, every attention weight, every output stays mathematically identical — the block genuinely cannot tell "the cat sat" from "sat the cat." Given everything built in this module about how information flows through a network, what would need to be added to the *input* itself — not the architecture — to let a shape-preserving, order-blind mechanism like this actually distinguish sequence order?
 
That's the on-ramp into Episode 04.04 — positional encoding, the deceptively simple piece that makes everything built so far actually usable for language.
 
---
 
**Previous:** Episode 04.02 — Trainable Attention: Learned Queries, Keys, and Values
**Next:** Episode 04.04 — Positional Encoding: Giving Attention a Sense of Order