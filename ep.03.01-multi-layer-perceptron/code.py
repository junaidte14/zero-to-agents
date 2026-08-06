#hand-built solution, the linear-collapse proof, and a real trained network

#From scratch — the hand-constructed XOR network, verified

import numpy as np

def step(z): return np.where(z >= 0, 1, 0)

W1 = np.array([[1, 1], [-1, -1]])   # row 0: OR weights, row 1: NAND weights
b1 = np.array([-0.5, 1.5])
W2 = np.array([1, 1])                # AND weights
b2 = -1.5

def forward(x):
    h = step(W1 @ x + b1)
    return h, step(W2 @ h + b2)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0,1,1,0])

for x, yt in zip(X, y_xor):
    h, y = forward(x)
    print(f"{x}  hidden=OR:{h[0]},NAND:{h[1]}  predicted={y}  true={yt}")

#Proving the linear-collapse claim from §2.2, numerically

np.random.seed(0)
W1r, b1r = np.random.randn(3,2), np.random.randn(3)
W2r, b2r = np.random.randn(2,3), np.random.randn(2)

def two_layer_linear(x):
    h = W1r @ x + b1r           # NOTE: no activation function here
    return W2r @ h + b2r

W_combined = W2r @ W1r
b_combined = W2r @ b1r + b2r
def one_layer_equiv(x): return W_combined @ x + b_combined

x_test = np.array([0.5, -1.2])
print("Two linear layers:", two_layer_linear(x_test))
print("Equivalent single layer:", one_layer_equiv(x_test))


#Using real training — learning XOR via gradient descent, no hand-set weights

import torch
import torch.nn as nn

torch.manual_seed(42)
X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
print(X)
y = torch.tensor([[0.],[1.],[1.],[0.]])
print(y)

model = nn.Sequential(nn.Linear(2,4), nn.Sigmoid(), nn.Linear(4,1), nn.Sigmoid())
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
loss_fn = nn.BCELoss()

for epoch in range(3000):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss={loss.item():.4f}")

with torch.no_grad():
    for xi, yi, p in zip(X, y, model(X)):
        print(f"{xi.tolist()} -> predicted={p.item():.3f}  true={yi.item()}")