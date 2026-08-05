import numpy as np

def f(x): return x**2
def grad(x): return 2*x

def gradient_descent(x0, lr, steps):
    x = x0
    trajectory = [x]
    for _ in range(steps):
        x = x - lr * grad(x)
        trajectory.append(x)
    return trajectory

for lr in [0.05, 0.4, 0.9, 1.1]:
    traj = gradient_descent(10.0, lr, 10)
    closed_form = [10.0 * (1 - 2 * lr) ** t for t in range(11)]
    print(f"lr={lr}  (1-2*lr)={1-2*lr:.2f}  matches closed form? {np.allclose(traj, closed_form)}")
    print("  ", [round(v, 2) for v in traj])