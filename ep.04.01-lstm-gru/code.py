#Code: building an LSTM cell, verifying it, then measuring the actual improvement

#1 From scratch — a full LSTM cell forward pass

import numpy as np
import torch
import torch.nn as nn

def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh(z): return np.tanh(z)

class ManualLSTMCell:
    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        concat_size = input_size + hidden_size
        scale = np.sqrt(1 / concat_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.Wf = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.Wi = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.Wg = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        self.Wo = rng.normal(0, 1, (hidden_size, concat_size)) * scale
        
        self.bf = np.ones((hidden_size, 1))    # forget gate bias -- init to 1
        self.bi = np.zeros((hidden_size, 1))
        self.bg = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def step(self, x_t, h_prev, c_prev):
        concat = np.vstack([h_prev, x_t])
        f_t = sigmoid(self.Wf @ concat + self.bf)
        i_t = sigmoid(self.Wi @ concat + self.bi)
        g_t = tanh(self.Wg @ concat + self.bg)
        o_t = sigmoid(self.Wo @ concat + self.bo)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)
        return h_t, c_t, f_t

# --- Verification Setup ---
input_size = 3
hidden_size = 4
seed = 42

# 1. Initialize manual cell
manual_cell = ManualLSTMCell(input_size, hidden_size, seed=seed)

# 2. Initialize PyTorch cell with identical weights
torch_cell = nn.LSTMCell(input_size, hidden_size)

# Extract Wh (hidden weights) and Wx (input weights) from manual matrices
# Note: concat = np.vstack([h_prev, x_t]), so columns 0:hidden_size are Wh, and the rest are Wx
h_sz = hidden_size
Wi_h, Wi_x = manual_cell.Wi[:, :h_sz], manual_cell.Wi[:, h_sz:]
Wf_h, Wf_x = manual_cell.Wf[:, :h_sz], manual_cell.Wf[:, h_sz:]
Wg_h, Wg_x = manual_cell.Wg[:, :h_sz], manual_cell.Wg[:, h_sz:]
Wo_h, Wo_x = manual_cell.Wo[:, :h_sz], manual_cell.Wo[:, h_sz:]

# PyTorch gates order: i, f, g, o
with torch.no_grad():
    torch_cell.weight_ih.copy_(torch.tensor(np.vstack([Wi_x, Wf_x, Wg_x, Wo_x]), dtype=torch.float32))
    torch_cell.weight_hh.copy_(torch.tensor(np.vstack([Wi_h, Wf_h, Wg_h, Wo_h]), dtype=torch.float32))
    torch_cell.bias_ih.copy_(torch.tensor(np.vstack([manual_cell.bi, manual_cell.bf, manual_cell.bg, manual_cell.bo]).flatten(), dtype=torch.float32))
    torch_cell.bias_hh.zero_() # Keep all biases in bias_ih to perfectly match

# 3. Create dummy sequence (3 steps, batch_size=1)
rng = np.random.default_rng(seed)
xs = [rng.normal(0, 1, (input_size, 1)) for _ in range(3)]

# Initialize states
h_manual = np.zeros((hidden_size, 1))
c_manual = np.zeros((hidden_size, 1))

h_torch = torch.zeros(1, hidden_size)
c_torch = torch.zeros(1, hidden_size)

# 4. Run step-by-step comparison
for step in range(3):
    x_t_np = xs[step]
    x_t_th = torch.tensor(x_t_np.T, dtype=torch.float32) # Shape (1, input_size)
    
    # Manual step
    h_manual, c_manual, _ = manual_cell.step(x_t_np, h_manual, c_manual)
    
    # PyTorch step
    h_torch, c_torch = torch_cell(x_t_th, (h_torch, c_torch))
    
    # Format arrays for clean display
    h_man_flat = h_manual.flatten()
    h_trch_flat = h_torch.detach().numpy().flatten()
    match = np.allclose(h_man_flat, h_trch_flat, atol=1e-6)
    
    # Print formatted output matching your template
    fmt = {'float_kind': lambda x: f"{x: .4f}"}
    man_str = np.array2string(h_man_flat, precision=4, formatter=fmt)
    trch_str = np.array2string(h_trch_flat, precision=4, formatter=fmt)
    
    print(f"step {step}: manual h={man_str}")
    print(f"        torch  h={trch_str}   match? {match}")


#3 The actual payoff — measuring gradient preservation, LSTM vs. vanilla RNN

import torch.nn as nn
import torch

def run_rnn(T, hidden_size=20):
    cell = nn.RNNCell(5, hidden_size, nonlinearity='tanh')
    x_seq = [torch.randn(1, 5, requires_grad=True) for _ in range(T)]
    h = torch.zeros(1, hidden_size)
    for t in range(T): h = cell(x_seq[t], h)
    h.sum().backward()
    return [x.grad.norm().item() for x in x_seq]

def run_lstm(T, hidden_size=20):
    cell = nn.LSTMCell(5, hidden_size)
    x_seq = [torch.randn(1, 5, requires_grad=True) for _ in range(T)]
    h, c = torch.zeros(1, hidden_size), torch.zeros(1, hidden_size)
    for t in range(T): h, c = cell(x_seq[t], (h, c))
    h.sum().backward()
    return [x.grad.norm().item() for x in x_seq]

for T in [15, 30, 50]:
    torch.manual_seed(0); rnn_grads = run_rnn(T)
    torch.manual_seed(0); lstm_grads = run_lstm(T)
    print(f"T={T}: RNN ratio(early/late)={rnn_grads[0]/rnn_grads[-1]:.3e}   LSTM ratio={lstm_grads[0]/lstm_grads[-1]:.3e}")

