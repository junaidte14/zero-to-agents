#Code: the masking mechanism, verified exactly, then an honest look at a toy training comparison

#1 Verifying ignore_index does exactly what §2.1 claims — down to the gradient

import torch
import torch.nn.functional as F

vocab_size = 10
seq = torch.tensor([1, 2, 3, 9, 4, 5])   # 9 = a separator token between prompt and response
inputs, targets = seq[:-1], seq[1:]       # standard next-token shift

IGNORE_INDEX = -100
masked_targets = targets.clone()
masked_targets[:3] = IGNORE_INDEX          # positions 0,1,2 are "predict more prompt" -- excluded

logits = torch.randn(5, vocab_size, requires_grad=True)
loss_unmasked = F.cross_entropy(logits, targets)
loss_masked = F.cross_entropy(logits, masked_targets, ignore_index=IGNORE_INDEX)
manual_check = F.cross_entropy(logits[3:], targets[3:])   # compute by hand on JUST the response positions

print(f"Unmasked loss (all 5 positions): {loss_unmasked.item():.4f}")
print(f"Masked loss (ignore_index):      {loss_masked.item():.4f}")
print(f"Manual CE on response positions only: {manual_check.item():.4f}")
print("Masked loss matches manual computation exactly?", torch.allclose(loss_masked, manual_check))

loss_masked.backward()
print("\nGradient norm per position:", logits.grad.norm(dim=1).numpy().round(4))
print("Prompt positions have EXACTLY zero gradient?", torch.allclose(logits.grad[:3].norm(dim=1), torch.zeros(3)))

#2 An honest attempt at measuring the training-outcome benefit — and an honest report of what happened

