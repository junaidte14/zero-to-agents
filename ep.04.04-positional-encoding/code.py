#Code: the permutation-blindness proven, the fix verified, the rotation property confirmed

#1 Proving the permutation-blindness directly

import torch
#1 Multi-head attention, from scratch

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
torch.manual_seed(1)
A, B, C = torch.randn(1,8), torch.randn(1,8), torch.randn(1,8)

seq_ABC = torch.stack([A,B,C], dim=1)
seq_CBA = torch.stack([C,B,A], dim=1)   # same tokens, fully reversed order

with torch.no_grad():
    out_ABC = mha(seq_ABC)
    out_CBA = mha(seq_CBA)

print("Is reversed-input output just the original output, reversed?",
      torch.allclose(out_CBA[0], out_ABC[0].flip(0), atol=1e-5))

#2 Generating sinusoidal encoding, and confirming it breaks the symmetry

def sinusoidal_pe(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(seq_len).unsqueeze(1).float()
    div_term = 10000 ** (torch.arange(0, d_model, 2).float() / d_model)
    pe[:, 0::2] = torch.sin(position / div_term)
    pe[:, 1::2] = torch.cos(position / div_term)
    return pe

pe = sinusoidal_pe(3, 8).unsqueeze(0)
with torch.no_grad():
    out_ABC_pe = mha(seq_ABC + pe)
    out_CBA_pe = mha(seq_CBA + pe)

print("With positional encoding, is it still just a reversal?",
      torch.allclose(out_CBA_pe[0], out_ABC_pe[0].flip(0), atol=1e-5))


#3 Confirming the relative-position rotation property from §2.2

import numpy as np

d_model, i, k = 8, 1, 3   # test dimension pair i=1, offset k=3
freq = 1 / (10000 ** (2*i/d_model))

def rotation_matrix(k, freq):
    theta = k * freq
    return np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])

R = rotation_matrix(k, freq)
for pos in [0, 2, 5, 7]:
    pe_pos = np.array([np.sin(pos*freq), np.cos(pos*freq)])
    pe_pos_plus_k = np.array([np.sin((pos+k)*freq), np.cos((pos+k)*freq)])
    predicted = R @ pe_pos
    print(f"pos={pos}: actual={np.round(pe_pos_plus_k,4)}  R@PE(pos)={np.round(predicted,4)}  match={np.allclose(pe_pos_plus_k, predicted)}")

