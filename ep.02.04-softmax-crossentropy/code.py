import numpy as np

values = np.array([1, 2, 3, 4])           # toy numeric mapping for {the, cat, sat, mat}
probs = np.array([0.5, 0.2, 0.2, 0.1])
print("Sum of probabilities:", probs.sum())  # 1.0

analytical_E = np.sum(values * probs)
print("Analytical E[X]:", analytical_E)      # 1.9

rng = np.random.default_rng(42)
samples = rng.choice(values, size=200000, p=probs)
print("Monte Carlo estimate (200,000 draws):", samples.mean())


#softmax

def softmax(z):
    z_shifted = z - np.max(z)   # subtract the max first -- prevents exp() overflow, doesn't change the result
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z)

scores = np.array([2.0, 1.0, 0.1, 3.0])
sm = softmax(scores)
print("Softmax:", sm, " sum:", sm.sum())

import torch
import torch.nn.functional as F
torch_sm = F.softmax(torch.tensor(scores), dim=0).numpy()
print("Match with torch?", np.allclose(sm, torch_sm))

#crossentropy

true_idx = 3   # the actually-correct token
ce_scratch = -np.log(sm[true_idx])
print("Cross-entropy (scratch):", ce_scratch)

logits = torch.tensor([scores], dtype=torch.float32)
ce_torch = F.cross_entropy(logits, torch.tensor([true_idx]))
print("Cross-entropy (torch):  ", ce_torch.item())
print("Match?", np.isclose(ce_scratch, ce_torch.item()))


scores_t = torch.tensor([2.0, 1.0, 0.1, 3.0], requires_grad=True)
probs_t = F.softmax(scores_t, dim=0)
loss = -torch.log(probs_t[3])
loss.backward()

one_hot_true = np.array([0, 0, 0, 1])
predicted = probs_t.detach().numpy()
print("Autograd gradient:       ", scores_t.grad.numpy())
print("predicted - one_hot_true:", predicted - one_hot_true)