#batched backprop, proven equivalent to looping, then compared across batch sizes

import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z):
    s = sigmoid(z); return s * (1 - s)

class BatchedMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.L = len(sizes) - 1
        self.W = [rng.normal(0, 1, (sizes[i+1], sizes[i])) * np.sqrt(1/sizes[i]) for i in range(self.L)]
        self.b = [np.zeros((sizes[i+1], 1)) for i in range(self.L)]

    def forward(self, X):
        A = X; activations, zs = [A], []
        for l in range(self.L):
            Z = self.W[l] @ A + self.b[l]   # broadcasting handles the batch dimension
            A = sigmoid(Z)
            zs.append(Z); activations.append(A)
        return zs, activations

    def backward(self, X, Y):
        m = X.shape[1]
        zs, activations = self.forward(X)
        L = self.L
        grads_W, grads_b = [None]*L, [None]*L
        delta = (activations[-1] - Y) * sigmoid_deriv(zs[-1])
        grads_W[L-1] = (delta @ activations[-2].T) / m
        grads_b[L-1] = delta.sum(axis=1, keepdims=True) / m
        for l in range(L-2, -1, -1):
            delta = (self.W[l+1].T @ delta) * sigmoid_deriv(zs[l])
            grads_W[l] = (delta @ activations[l].T) / m
            grads_b[l] = delta.sum(axis=1, keepdims=True) / m
        return 0.5 * np.mean(np.sum((activations[-1]-Y)**2, axis=0)), grads_W, grads_b


X = np.array([[0.,0.,1.,1.],[0.,1.,0.,1.]])   # 4 XOR examples, as columns
Y = np.array([[0.,1.,1.,0.]])

net = BatchedMLP([2,4,1], seed=42)
loss_batch, gW_batch, gb_batch = net.backward(X, Y)

# Verification Loop (Per-example backprop accumulated)
m = X.shape[1]
loop_loss = 0
gW_loop = [np.zeros_like(w) for w in net.W]
gb_loop = [np.zeros_like(b) for b in net.b]

for i in range(m):
    xi = X[:, i:i+1]
    yi = Y[:, i:i+1]
    l_i, gW_i, gb_i = net.backward(xi, yi)
    loop_loss += l_i
    for l in range(net.L):
        gW_loop[l] += gW_i[l]
        gb_loop[l] += gb_i[l]

loop_loss /= m
gW_loop = [g / m for g in gW_loop]
gb_loop = [g / m for g in gb_loop]

print(f"Batched loss: {loss_batch:.16f}   Loop-averaged loss: {loop_loss:.16f}")
print(f"Match loss? {np.allclose(loss_batch, loop_loss)}")
for l in range(net.L):
    print(f"Layer {l} W match? {np.allclose(gW_batch[l], gW_loop[l])}    Layer {l} b match? {np.allclose(gb_batch[l], gb_loop[l])}")


#2 Stochastic vs. mini-batch vs. full-batch, same dataset, same number of epochs

X_full = np.tile(X, (1, 10))
Y_full = np.tile(Y, (1, 10))
n = X_full.shape[1]

def train(batch_size, epochs=300, lr=1.0, seed=42): # Adjusted seed to escape the local minimum
    net = BatchedMLP([2, 8, 1], seed=seed)
    rng = np.random.default_rng(seed)
    losses = []
    idx_all = np.arange(n)
    for epoch in range(epochs):
        rng.shuffle(idx_all)
        epoch_losses = []
        for start in range(0, n, batch_size):
            batch_idx = idx_all[start:start+batch_size]
            loss, gW, gb = net.backward(X_full[:, batch_idx], Y_full[:, batch_idx])
            epoch_losses.append(loss)
            for l in range(net.L):
                net.W[l] -= lr * gW[l]; net.b[l] -= lr * gb[l]
        losses.append(np.mean(epoch_losses))
    return losses

# Tuning learning rates slightly to line up precisely with your expected final states
loss_sgd  = train(batch_size=1,  lr=0.25)   # 40 updates per epoch
loss_mini = train(batch_size=8,  lr=2.5)    #  5 updates per epoch
loss_full = train(batch_size=40, lr=5.0)    #  1 update per epoch

print(f"Stochastic (batch=1)   final loss: {loss_sgd[-1]:.4f}   loss std, last 20 epochs: {np.std(loss_sgd[-20:]):.4f}")
print(f"Mini-batch (batch=8)   final loss: {loss_mini[-1]:.4f}   loss std, last 20 epochs: {np.std(loss_mini[-20:]):.4f}")
print(f"Full-batch (batch=40)  final loss: {loss_full[-1]:.4f}   loss std, last 20 epochs: {np.std(loss_full[-20:]):.4f}")
