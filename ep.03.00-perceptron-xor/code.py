#Ep:03.00 building a perceptron, watching it succeed and fail exactly as proven

#from scratch

import numpy as np

def step(z):
    return 1 if z >= 0 else 0

class Perceptron:
    def __init__(self, n_inputs, lr=0.1):
        self.w = np.zeros(n_inputs)
        self.b = 0.0
        self.lr = lr

    def predict(self, x):
        return step(np.dot(self.w, x) + self.b)

    def train(self, X, y, epochs=20):
        history = []
        for _ in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.lr * (yi - self.predict(xi))   # exactly Rosenblatt's rule, §3
                self.w += update * xi
                self.b += update
                errors += int(update != 0.0)
            history.append(errors)
            if errors == 0:
                break
        return history

X = np.array([[0,0],[0,1],[1,0],[1,1]])

p_and = Perceptron(2)
print("AND errors per epoch:", p_and.train(X, np.array([0,0,0,1]), epochs=20))
print("AND learned weights:", p_and.w, "bias:", p_and.b)

p_xor = Perceptron(2)
print("\nXOR errors per epoch:", p_xor.train(X, np.array([0,1,1,0]), epochs=50))

#Using a library — sklearn, confirming the identical result

from sklearn.linear_model import Perceptron

clf_and = Perceptron(max_iter=1000).fit(X, [0,0,0,1])
print("AND accuracy:", clf_and.score(X, [0,0,0,1]))   # 1.0

clf_xor = Perceptron(max_iter=1000).fit(X, [0,1,1,0])
print("XOR accuracy:", clf_xor.score(X, [0,1,1,0]))   # 0.5 -- chance level