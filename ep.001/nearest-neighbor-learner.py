# A tiny nearest-neighbor "learner": given a few labeled examples,
# it generalizes to inputs it has never seen, using structure (distance),
# not memorized exact matches.

import math

training_examples = [
    # (x, y) -> label:  1 if inside the unit circle, else 0
    ((0.1, 0.1), 1),
    ((0.9, 0.9), 0),
    ((0.2, -0.1), 1),
    ((1.2, 0.3), 0),
    ((-0.3, 0.2), 1),
]

def euclidean_distance(p, q):
    return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def nearest_neighbor_classify(point, examples):
    closest = min(examples, key=lambda ex: euclidean_distance(point, ex[0]))
    return closest[1]

# Test on points NEVER seen during "training":
for test_point in [(0.05, -0.05), (1.5, 1.5), (0.4, 0.4)]:
    label = nearest_neighbor_classify(test_point, training_examples)
    print(f"{test_point} -> predicted inside-circle = {label}")