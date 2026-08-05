# From Zero to Agents
## Module 02 — Mathematical Foundations
### Episode 02.02: Derivatives and Gradients — The Mechanism Behind "Learning"
 
---
 
## 0. Closing the open question
 
Episode 02.01 ended by asking: if training a network means adjusting a matrix's entries to make its output progressively less wrong, what tool tells you *which direction* to adjust each entry, and *by how much*? The answer is the **derivative** — and its generalization to many-input functions, the **gradient**. This episode builds both from scratch, and by the end, "gradient descent" — the actual mechanism behind every model this course will train from here forward — stops being a phrase and becomes something you've computed by hand and watched work.
 
## 1. Theory: rate of change, and why it tells you which way to move
 
**1.1 The single-variable derivative — slope of the tangent line.**
For a function $f(x)$, the derivative $f'(x)$ answers one question: *if I nudge $x$ by a tiny amount, how much does $f(x)$ change, and in which direction?* Geometrically, it's the slope of the line that just touches the curve at that exact point (the tangent line). A positive derivative means $f$ is increasing there; negative means decreasing; the derivative's *magnitude* tells you how steeply.
 
**1.2 Two concrete examples before any formula.**
- **Physical**: if $x(t)$ is an object's position over time, $x'(t)$ is its velocity — literally "how fast, and in which direction, is position changing right now." This is the example calculus was originally invented to formalize (Newton, 17th century).
- **A simple curve**: $f(x) = x^2$. At $x=2$, the curve is rising steeply; at $x=0$, it's momentarily flat (the bottom of the bowl); at $x=-3$, it's falling steeply as $x$ increases toward zero. Notice the shape of this function — a bowl with a single minimum — is exactly the shape a squared-error loss function has, which is not a coincidence we'll let go unremarked in §1.4.
**1.3 Partial derivatives — when a function has more than one input.**
A matrix (Episode 02.01) has many entries, and a network's loss depends on *all* of them simultaneously. The **partial derivative** $\frac{\partial f}{\partial x_i}$ answers the single-variable question, but with everything else held fixed: *if I nudge only $x_i$, leaving every other input exactly where it is, how does $f$ change?* Compute one partial derivative per input, and you know how the function responds to a small nudge in each individual direction, independently.
 
**1.4 The gradient — all the partial derivatives, collected into one vector, and why it points "uphill."**
Stack every partial derivative into a single vector, and you get the **gradient**, $\nabla f$ — and here's the part that makes this whole episode worth it: the gradient vector points in the direction of **steepest ascent** — the single direction, among all possible directions you could nudge the inputs, that increases $f$ the *fastest*. Immediate, load-bearing consequence: the **negative** gradient points in the direction of steepest *descent*. If $f$ is a loss function (something we want to minimize — recall §1.2's bowl shape), repeatedly taking small steps in the direction of the negative gradient is a direct, principled way to make the loss smaller, a tiny bit at a time. This is **gradient descent**, the training algorithm behind essentially every model in this course from Module 03 onward — and it's nothing more exotic than "the derivative tells you which way is downhill; walk that way."
 
**1.5 The chain rule — derivatives of functions built from other functions.**
Real networks are functions composed of other functions, layer after layer. The **chain rule** is what makes differentiating a composition tractable: if $L$ depends on $y$, and $y$ depends on $w$, then:
 
$$\frac{dL}{dw} = \frac{dL}{dy} \cdot \frac{dy}{dw}$$
 
— multiply the "local" rates of change along the chain. This single rule, applied repeatedly through every layer of a network, back to front, *is* backpropagation (Rumelhart, Hinton & Williams, 1986) — the algorithm that trains virtually every neural network in existence, including the transformer we're building toward. Nothing about backpropagation is more exotic than this equation, applied many times in sequence; §3 makes that concrete.
 
## 2. Math: stated precisely, with a genuinely ML-relevant worked example
 
**2.1 The derivative, defined via a limit.**
 
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$
 
Read it as: take a tiny step $h$ from $x$, see how much $f$ changed, divide by the size of the step (to get a *rate*, not just a raw change), then ask what that ratio approaches as the step shrinks toward zero. For $f(x) = x^2$, this limit works out (via algebra we won't rederive here) to the familiar power rule: $f'(x) = 2x$.
 
**2.2 A worked example that's actually a tiny piece of linear regression.**
Consider a single-example squared-error loss for a 2-feature linear model — this is not a toy unrelated to the rest of the course; it's the literal loss function used to train linear regression, and structurally identical to what trains every layer of every network we'll build later:
 
$$L(w_1, w_2) = (w_1 x_1 + w_2 x_2 - y)^2$$
 
Here $x_1, x_2$ are fixed input features, $y$ is the fixed true target value, and $w_1, w_2$ are the *weights* — the things we're allowed to adjust to make the prediction $w_1x_1+w_2x_2$ closer to $y$. Applying the chain rule from §1.5 (outer function: "square it"; inner function: "the prediction error"):
 
$$\frac{\partial L}{\partial w_1} = 2(w_1x_1 + w_2x_2 - y) \cdot x_1, \qquad \frac{\partial L}{\partial w_2} = 2(w_1x_1 + w_2x_2 - y) \cdot x_2$$
 
The gradient is $\nabla L = \left(\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}\right)$ — and per §1.4, taking a small step in the *negative* of this vector should make $L$ smaller. Section 4 verifies this numerically, not just symbolically.
 
## 3. Decoding a real equation — backpropagation's actual notation
 
Papers and textbooks describing backpropagation write the chain rule almost exactly as in §1.5, typically as:
 
$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial w}$$
 
where $z$ is some intermediate quantity a weight $w$ affects on its way to influencing the final loss $L$ (in our worked example, $z$ would be the prediction $w_1x_1+w_2x_2$). Read cold, in a paper, with everything above in hand: this says "the effect of $w$ on the final loss equals the effect of $w$ on the intermediate quantity $z$, multiplied by the effect of $z$ on the loss" — exactly the chain of dominoes intuition from §1.5, just written for a network with an explicit intermediate step named. When a paper shows a chain of several such terms multiplied together — $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial z_3}\cdot\frac{\partial z_3}{\partial z_2}\cdot\frac{\partial z_2}{\partial z_1}\cdot\frac{\partial z_1}{\partial w}$ — that's the identical rule applied across more layers; nothing new is happening, just more dominoes in the chain. This is precisely the computation Module 03 will formalize as backpropagation proper, and precisely what PyTorch's `autograd` (§4.2) is doing automatically, layer by layer, every time a real model trains.
 
## 4. Code: analytical derivatives verified numerically, then verified again by autograd
 
**4.1 From scratch — analytical vs. numerical derivatives, and the gradient-descent step itself**
 
```python
import numpy as np
 
def f(x): return x**2
def f_prime_analytical(x): return 2*x
def f_prime_numerical(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)   # central difference
 
for x in [0, 1, 2, -3, 5.5]:
    print(f"x={x:5}: analytical={f_prime_analytical(x):8.4f}  numerical={f_prime_numerical(f, x):8.4f}")
```
```
x=    0: analytical=  0.0000  numerical=  0.0000
x=    1: analytical=  2.0000  numerical=  2.0000
x=    2: analytical=  4.0000  numerical=  4.0000
x=   -3: analytical= -6.0000  numerical= -6.0000
x=  5.5: analytical= 11.0000  numerical= 11.0000
```
 
Exact agreement at every point, confirming §2.1's power rule numerically, not just symbolically — a useful general habit: numerical (finite-difference) derivatives are a reliable way to sanity-check an analytical derivative you've worked out by hand.
 
```python
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
```
```
At w1=1.0, w2=1.0: L=25.0, gradient=[-20. -30.]
After one gradient-descent step: w1=1.2000, w2=1.3000
L before: 25.0000   L after: 13.6900
```
 
This is the entire mechanism of learning, made completely concrete: at $w_1=w_2=1$, the loss is $25.0$. The gradient $(-20, -30)$ says "moving $w_1$ and $w_2$ in the *positive* direction decreases the loss fastest" (note the negative sign — increasing the weights here reduces error, given these particular $x_1,x_2,y$). Taking one small step *against* the gradient's sign — exactly what gradient descent does — drops the loss from $25.0$ to $13.69$, in a single step, with no search or guessing involved.
 
**4.2 Using a library — the same gradient, computed automatically by PyTorch's autograd**
 
```python
import torch
 
w1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
 
L = (w1 * x1 + w2 * x2 - y) ** 2
L.backward()   # applies the chain rule automatically, layer by layer
 
print("L =", L.item())
print("dL/dw1 (autograd):", w1.grad.item())
print("dL/dw2 (autograd):", w2.grad.item())
```
```
L = 25.0
dL/dw1 (autograd): -20.0
dL/dw2 (autograd): -30.0
```
 
Exact match to the hand-derived gradient in §4.1 — `-20.0` and `-30.0`, precisely. `.backward()` is PyTorch performing the chain rule from §1.5/§3 automatically, without us ever writing `dL_dw1` or `dL_dw2` by hand. This is autograd's entire job: given any computation built from differentiable pieces, automatically apply the chain rule across every step to produce exact gradients — and it's the single piece of machinery that makes training realistic transformer-scale models with millions or billions of parameters actually feasible, since nobody is hand-deriving gradients at that scale.
 
## 5. Where this leaves us
 
We now have the actual mechanism behind "a model learns": compute how the loss responds to a tiny nudge in every trainable number (the gradient), then repeatedly step in the direction that makes the loss smaller. Verified three separate ways in this one episode — analytically, numerically via finite differences, and automatically via autograd — all agreeing exactly. This is no longer an abstraction; Section 4.1 is a complete, working, if tiny, training step.
 
## 6. Before Episode 02.03
 
> Section 4.1 took exactly one gradient-descent step and the loss dropped from 25.0 to 13.69. Real training takes thousands or millions of such steps. Two things are worth sitting with: first, what happens if you keep applying this same update rule over and over — does the loss reliably keep dropping, forever, or could it overshoot, oscillate, or get stuck? Second — the gradient told us a *direction*; the learning rate (`0.01` in our code) controlled *how big a step* to take in that direction. What might go wrong if that step size were much larger, or much smaller?
 
That's the on-ramp into Episode 02.03 — a closer look at gradient descent as an iterative process, and the mathematics of probability and expectation we'll need before Module 03 can properly formalize a loss function in the first place.
 
---
 
**Previous:** Episode 02.01 — Matrices and the Geometry of Linear Transformations
**Next:** Episode 02.03 — Iterating Gradient Descent, and an Introduction to Probability