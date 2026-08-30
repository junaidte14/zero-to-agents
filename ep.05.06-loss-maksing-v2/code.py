#Code: the properly-scoped experiment, and what it actually shows

#previous episode code

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

#1 The task and setup

# vocab: digits 0-9, SEP=10. Sequence: [d1,d2,d3, SEP, r1,r2,r3] where r = reverse(d1,d2,d3)
# 200 training sequences, 100 held-out test sequences, drawn from 300 unique random 3-digit combos

#print("Final train loss masked:", loss_m, " unmasked:", loss_u)
#print("Masked   -- train acc:", evaluate(model_masked, train_seqs), " test acc:", evaluate(model_masked, test_seqs))
#print("Unmasked -- train acc:", evaluate(model_unmasked, train_seqs), " test acc:", evaluate(model_unmasked, test_seqs))

#Final train loss masked: 0.0000675   unmasked: 0.5021

#Masked   -- train acc: (200, 200)  test acc: (100, 100)
#Unmasked -- train acc: (200, 200)  test acc: (97, 100)