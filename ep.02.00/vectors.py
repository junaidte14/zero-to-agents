import numpy as np

# Multiple examples of vectors from different domains
displacement = np.array([3, 2])                  # physical: 3 east, 2 north
house = np.array([1450, 3, 320000])               # sqft, bedrooms, price -- mixed units, flagged in §1.2
king_embedding = np.array([-1.4, 0.7, 2.1, 0.3])   # a toy embedding, Module 00-style

print("Displacement dimension:", displacement.shape[0])
print("House feature dimension:", house.shape[0])
print("Embedding dimension:", king_embedding.shape[0])

# Vector addition and scalar multiplication
u, w = np.array([1, 3]), np.array([3, -1])
print("\nu + w =", u + w)          # [4, 2]
print("2.5 * u =", 2.5 * u)         # [2.5, 7.5]

# Dot product: algebraic vs. geometric, proven equal
a, b = np.array([2, 0]), np.array([1, 1])
dot = np.sum(a * b)
norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
cos_theta = dot / (norm_a * norm_b)
theta_deg = np.degrees(np.arccos(cos_theta))
print(f"\ndot(a,b) = {dot}")
print(f"|a||b|cos(theta) = {norm_a * norm_b * cos_theta:.3f}  <- matches exactly, angle = {theta_deg:.1f} deg")

# Norms: L1, L2, and general Lp
v = np.array([3, -4])
def lp_norm(v, p):
    return np.sum(np.abs(v) ** p) ** (1 / p)

for p in [1, 2, 3, 10]:
    print(f"L{p} norm of {v} = {lp_norm(v, p):.4f}")

# Unit vector -- exactly what cosine similarity has been dividing by since Episode 00.02
unit_v = v / np.linalg.norm(v)
print(f"\nUnit vector: {unit_v}, its own L2 norm = {np.linalg.norm(unit_v):.4f}")  # always 1.0