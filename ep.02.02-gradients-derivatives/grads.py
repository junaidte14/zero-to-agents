import numpy as np

def f(x): return x**2
def f_prime_analytical(x): return 2*x
def f_prime_numerical(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)   # central difference

for x in [0, 1, 2, -3, 5.5]:
    print(f"x={x:5}: analytical={f_prime_analytical(x):8.4f}  numerical={f_prime_numerical(f, x):8.4f}")


x1, x2, y = 2.0, 3.0, 10.0

def L(w1, w2): return (w1*x1 + w2*x2 - y)**2
def dL_dw1(w1, w2): return 2*(w1*x1 + w2*x2 - y) * x1
def dL_dw2(w1, w2): return 2*(w1*x1 + w2*x2 - y) * x2

w1, w2 = 1.0, 1.0
grad = np.array([dL_dw1(w1, w2), dL_dw2(w1, w2)])
print(f"At w1={w1}, w2={w2}: L={L(w1, w2)}, gradient={grad}")

lr = 0.01
w1_new, w2_new = w1 - lr * grad[0], w2 - lr * grad[1]
print(f"After one gradient-descent step: w1={w1_new:.4f}, w2={w2_new:.4f}")
print(f"L before: {L(w1, w2):.4f}   L after: {L(w1_new, w2_new):.4f}")


import torch

w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)

L = (w1 * x1 + w2 * x2 - y) ** 2
L.backward()   # applies the chain rule automatically, layer by layer

print("L =", L.item())
print("dL/dw1 (autograd):", w1.grad.item())
print("dL/dw2 (autograd):", w2.grad.item())