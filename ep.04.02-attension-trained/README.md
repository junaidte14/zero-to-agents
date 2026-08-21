# From Zero to Agents
## Module 04 — Sequence Models and Attention
### Episode 04.02: Trainable Attention — Learned Queries, Keys, and Values
 
---
 
## 0. Closing the open question
 
Episode 04.01 ended by asking what advantage a mechanism with no sequential chain at all — where any two positions are connected by one direct computation — might have over even a well-gated recurrent architecture. Episode 02.01 already built the mechanical answer: $QK^T$ computes every pairwise relationship in one matrix multiplication, no timestep-by-timestep chain, no accumulating gradient shrinkage across a sequence. But that earlier version had a real limitation, flagged explicitly at the time and left unresolved until now: $Q$, $K$, and $V$ were just the *same static embeddings*, reused three times. Today they become genuinely **learned** — and the difference this makes is large enough to demonstrate directly, not just assert.
 
## 1. Theory: why reusing static embeddings as Q, K, and V isn't enough
 
**1.1 What "relevant" means is task-dependent, not fixed by embedding similarity.**
Episode 02.01's attention used raw embedding similarity to decide what a word should attend to — which worked for the "bank" example specifically because *semantic* similarity happened to be the right signal for disambiguating word sense. But plenty of real relationships a model needs to track have nothing to do with semantic similarity at all: which pronoun refers to which noun, which word is the grammatical subject of which verb, which token was specially marked as important for a specific task. A fixed embedding can't be relevant to *all* of these different notions of "what matters here" simultaneously — it was built for one purpose (predicting nearby words, back in Module 00) and reused for structurally unrelated tasks by necessity, not by design.
 
**1.2 The fix — three separate learned transformations.**
Instead of using the embedding directly as query, key, and value, pass it through three separate **learned linear transformations** first (exactly Episode 03.01's linear layer, applied three times with three independent weight matrices):
 
$$Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V$$
 
Each of $W_Q, W_K, W_V$ is trained via ordinary backpropagation to reshape the raw embedding into whatever representation is actually useful for *this specific task's* notion of relevance — a completely different notion than raw embedding similarity, discovered entirely from data rather than designed by hand. Section 4 demonstrates this concretely with a task where embedding similarity provides *zero* useful signal, and only a learned transformation can solve it.
 
**1.3 Multi-head attention — a forward pointer.**
Real transformers don't use a single set of $Q,K,V$ projections — they use several **in parallel** (Vaswani et al., 2017 call this **multi-head attention**), each with its own independently-learned $W_Q, W_K, W_V$. The intuition: different heads can specialize in tracking different kinds of relationships simultaneously — one head might learn something closer to syntactic dependency, another something closer to coreference, without either being told to. We won't build the full multi-head mechanism today (that's Episode 04.03's job, once single-head attention is solid) — but it's worth knowing the single learned projection built here is exactly one "head" of that larger mechanism.
 
## 2. Math: what changes, and an honest note on what we won't hand-derive
 
**2.1 The formula is unchanged; what feeds it is not.**
The attention formula itself is identical to Episode 02.01 §2.4:
 
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
 
The only difference: $Q$, $K$, $V$ are now $XW_Q$, $XW_K$, $XW_V$ — outputs of trainable layers — rather than $X$ itself, three times over.
 
**2.2 An honest scoping note.**
Training $W_Q, W_K, W_V$ requires backpropagating the loss gradient *through* the softmax and matrix multiplications in the attention formula itself — genuinely involved calculus (the softmax Jacobian in particular is not a simple elementwise derivative like the activation functions in Episode 03.04). We're deliberately not hand-deriving that backward pass symbolically in this episode. This isn't a shortcut avoiding rigor — it's an accurate reflection of how the field actually operates: virtually nobody hand-derives full transformer backward passes by hand in practice; Episode 03.04 already proved, rigorously, that autograd computes exactly the same gradients a hand-derivation would produce, for an entirely different architecture. That proof transfers here by the same modular logic from Episode 03.04 §1.3 — every differentiable operation, attention included, plugs into the same automatic differentiation machinery. Section 4 trains real weights via `autograd`, with full confidence (backed by Episode 03.04's verification) that it's doing the same thing correct manual differentiation would do.
 
## 3. Decoding real notation — multi-head attention, verbatim from the paper
 
Vaswani et al. (2017) write multi-head attention as:
 
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O, \qquad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
 
Decoded, using everything built so far: each $\text{head}_i$ is one complete, independent attention computation from §2.1, with its *own* learned projection matrices $W_i^Q, W_i^K, W_i^V$ — literally $h$ separate copies of the single-head mechanism built in this episode, run in parallel on the same input $X$. Their outputs get concatenated side by side (stacked into one wider matrix, not summed or averaged), then passed through one more learned linear transformation $W^O$ to blend the different heads' findings back into a single output of the original width. Recognizing this equation now: it is not new machinery, it is $h$ literal copies of §2.1's formula, plus concatenation, plus one more linear layer — the entire complexity is in the *count*, not in any single head's mechanism being more complex than what this episode already builds.
 
## 4. Code: training real Q/K/V weights on a task raw embeddings cannot solve
 
**4.1 A task specifically designed so embedding similarity gives zero useful signal**
 
Each example: a sequence of random content vectors, one of which has a special "flag" dimension set to 1 (marking "the important one," at a random position each time). The model must learn to output that flagged position's content. Since the content vectors are freshly random every example, there is *no* stable embedding-similarity signal to exploit — the only reliable signal is the flag dimension, and using it correctly is something the model has to *learn*, not something a pre-existing static embedding could already encode.
 
```python
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
```
 
**4.2 Attention before training — as expected, unfocused**
 
```python
torch.manual_seed(0)
untrained = LearnedAttention(d_model, d_k)
X, targets, flag_pos = make_batch(3)
with torch.no_grad():
    _, weights_before = untrained(X)
for i in range(3):
    print(f"example {i}: flagged pos={flag_pos[i].item()}  weights={weights_before[i].numpy().round(3)}")
```
```
example 0: flagged pos=0  weights=[0.217 0.209 0.189 0.184 0.2  ]
example 1: flagged pos=4  weights=[0.185 0.242 0.212 0.185 0.176]
example 2: flagged pos=1  weights=[0.17  0.202 0.186 0.25  0.192]
```
 
With random initial weights, attention is close to uniform across positions, with no relationship to where the flag actually is — exactly what you'd expect before any training has happened.
 
**4.3 Training, and attention afterward**
 
```python
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
```
```
step    0: loss=0.9827
step  400: loss=0.0000
step  800: loss=0.0000
step 1200: loss=0.0000
step 1600: loss=0.0000
 
example 0: flagged pos=2  top-attended pos=2  match=True
example 1: flagged pos=0  top-attended pos=0  match=True
example 2: flagged pos=3  top-attended pos=3  match=True
... (all 10 examples match)
```
 
Loss collapses to essentially zero, and on ten fresh, never-seen examples, the trained model attends to the *exactly correct* flagged position, every single time. Compare this directly against §4.2's untrained, near-uniform weights: nothing in the raw content vectors gave any hint about which position mattered — the network *discovered*, purely through gradient descent adjusting $W_Q$ and $W_K$, to base its attention scores on the one dimension that actually carried the relevant signal. This is precisely what Episode 02.01's static-embedding attention could never do — it had no mechanism to learn a task-specific notion of relevance at all, because nothing about it was trainable.
 
**4.4 Confirming this is exactly what `nn.MultiheadAttention` implements internally**
 
```python
mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
X = torch.randn(3, 5, 8)
query = torch.randn(3, 1, 8)
output, attn_weights = mha(query, X, X)
print("Output shape:", output.shape)              # torch.Size([3, 1, 8])
print("Attention weights shape:", attn_weights.shape)  # torch.Size([3, 1, 5])
print("Total learnable parameters:", sum(p.numel() for p in mha.parameters()))  # 288
```
 
`nn.MultiheadAttention` — the actual production layer used inside real transformer implementations — holds exactly the learnable $W_Q, W_K, W_V$ (and an output projection $W^O$, per §3) our manual implementation built by hand, just wrapped for convenience and generalized to multiple heads.
 
## 5. Where this leaves us
 
Attention is no longer a fixed, hand-computed mechanism reusing embeddings that were trained for an unrelated purpose — it's a fully trainable component whose entire job is to *learn* what "relevant" means for whatever task it's embedded in, verified directly on a task where the correct notion of relevance couldn't have been known in advance. Combined with Episode 04.00–04.01's finding that attention has no sequential chain to cause gradient decay across positions, this is the actual mechanism that displaced recurrent architectures as the default for sequence modeling.
 
## 6. Before Episode 04.03
 
> Section 1.3 named multi-head attention but didn't build it — several independent copies of today's mechanism, run in parallel, then combined. What do you think a network gains from having several *independently initialized and independently trained* attention heads working on the same input, compared to just one larger single-head attention layer with a proportionally bigger $d_k$? (Consider: does a single head have any way to simultaneously prioritize two genuinely different notions of relevance at once?)
 
That's the on-ramp into Episode 04.03 — building full multi-head attention from scratch, and finally assembling a complete transformer block.
 
---
 
**Previous:** Episode 04.01 — LSTMs, GRUs, and the Gating Mechanism
**Next:** Episode 04.03 — Multi-Head Attention and the Transformer Block