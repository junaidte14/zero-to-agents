# From Zero to Agents
## Module 00 — Introduction
### Episode 00.01: What Is Intelligence?

---

## 0. Where we're starting from

Last episode ended with a question. Here's a definition worth pressure-testing (a very common first instinct — someone may say):

> A system is intelligent if it can analyze requirements and produce a solution that matches them.

This is a *specification-satisfaction* definition. Let's break it with three examples before building something sturdier.

**Counterexample A — the calculator.** Input `2+2`, requirement "return the correct sum," output `4`, every single time, with zero error. By the definition above, a four-function calculator is intelligent. Almost nobody's intuition agrees.

**Counterexample B — the thermostat.** Requirement: "hold the room at 21°C." It senses temperature, compares to setpoint, actuates the heater, continuously, correctly. Same problem.

**Counterexample C — the opposite failure.** A manager tells an employee: "make the customer happy." No spec. No defined "solution shape." The employee has to figure out *what the requirement even is* — read the situation, infer unstated needs, decide on an approach nobody wrote down — before any "solving" starts. This feels like a *more* intelligent act than A or B, and yet it doesn't fit the definition at all, because there was no clear requirement to analyze in the first place.

So the specification-satisfaction definition is simultaneously **too generous** (lets in things with obviously zero intelligence — a calculator, a thermostat, a lookup table) and **too strict** (excludes the thing that feels most central to intelligence — operating well under ambiguity, novelty, and incomplete information).

This is not a flaw specific to your phrasing — it's the exact same trap that early AI research fell into for decades, and it's worth seeing why.

## 1. Theory: the definitions that have been tried, and why each one leaks

**1.1 The "achieves goals" definition (McCarthy, 1955-ish; classical AI).**
John McCarthy, who coined the term "artificial intelligence," defined it roughly as *the computational part of the ability to achieve goals in the world.* This is closer, but "achieves goals" still doesn't distinguish a thermostat (which achieves its goal beautifully) from something we'd call intelligent. Goal-achievement is necessary but not sufficient.

**1.2 The behavioral test (Turing, 1950).**
Alan Turing sidestepped the definition problem entirely: instead of asking "what is intelligence," he asked "can we build a test where we can't tell the difference?" The famous Imitation Game — a human judge, texting with a hidden human and a hidden machine, tries to tell which is which. If the judge can't reliably tell, the machine passes.

This is clever precisely because it refuses to define intelligence internally — it defines it *externally*, by indistinguishability of behavior. But it has a well-known leak: a system can pass by being a very good mimic of surface patterns (this is essentially the "stochastic parrot" critique aimed at LLMs decades later) without any of the underlying capability we actually care about — generalizing to problems it hasn't memorized answers for.

**1.3 The rational-agent definition (Russell & Norvig — the standard modern textbook framing).**
An intelligent agent is one that **selects actions to maximize its expected performance measure, given its percept history and prior knowledge.** This is a big improvement: it separates the *goal* (performance measure) from *how well you achieve it under uncertainty* (expected value given imperfect information), and it's the definition most of modern AI — including the RL formulation of agentic systems you already build professionally — is actually built on.

But notice what's still missing: a lookup table that's had every possible situation pre-solved by a human and hard-coded also "maximizes expected performance." It still feels like cheating. Which leads to the piece almost every casual definition (including your original one) skips entirely:

**1.4 The missing ingredient: generalization under novelty.**
This is the crux, and it's the part François Chollet formalizes explicitly in *"On the Measure of Intelligence"* (2019) — a paper worth knowing exists even before you read it in full. His core move: stop measuring **skill** at a task, and start measuring **skill-acquisition efficiency** — how well a system performs on tasks it was *not specifically prepared for*, relative to how much prior knowledge and experience it was given.

A chess engine that crushes every human is enormously *skilled*, but it required either massive prior encoding (rules, heuristics) or massive experience (self-play at industrial scale) to get there, and it generalizes to nothing outside chess. A toddler who's seen a handful of examples of "stacking" can apply the concept of stacking to objects she's never touched before, with almost no prior data. By skill alone, the chess engine wins. By skill *per unit of prior knowledge and experience spent acquiring it* — Chollet's actual proposed measure — the toddler is doing something far more intelligent.

This reframes your counterexamples perfectly:
- The **calculator** and **thermostat** have zero generalization — they only ever do the one narrow thing they were built for, and that's fine because we didn't need or want more.
- The **employee facing an ambiguous ask** is doing high generalization — applying loosely related prior knowledge (what "happy customers" have looked like before) to a completely new, unspecified situation.

**1.5 A second useful lens: intelligence as compression/prediction.**
A separate but complementary tradition (Solomonoff, later formalized by Marcus Hutter and Shane Legg) frames intelligence as the ability to find the *shortest/simplest explanation* that predicts what happens next — essentially, compression. A system that can predict a sequence well is implicitly modeling the underlying structure that generated it, which is why "predict the next token" turned out to be such a surprisingly powerful training objective for the LLMs your agentic systems are built on. This isn't a coincidence — it's the same idea we'll meet again, formally, when we get to language modeling in Module 01.

## 2. Math: formalizing "maximize expected performance under uncertainty"

Let's put Russell & Norvig's rational-agent definition into notation, because this exact structure — agent, environment, percept, action, reward — is the skeleton every RL formulation and every agentic system (including the ones you build at AIVerse) is built on top of. You will see this again, unchanged in spirit, when we reach agentic workflows in the later modules.

Define:
- An **environment** with internal state $s \in S$ (unknown to the agent in general).
- An agent that receives a **percept** $o_t$ at each timestep $t$ (its observation of the world — not necessarily the full state).
- The agent selects an **action** $a_t \in A$ according to a **policy** $\pi$: a function (possibly stochastic) mapping percept history to an action, $a_t = \pi(o_1, o_2, \ldots, o_t)$.
- A **performance measure** (reward) $r_t = R(s_t, a_t)$, scoring how good that action was in that state.

The rational agent is defined as the one that chooses the policy maximizing **expected cumulative reward**:

$$\pi^* = \arg\max_{\pi} \ \mathbb{E}\left[ \sum_{t=0}^{T} \gamma^t \, r_t \ \middle| \ \pi \right]$$

Read this piece by piece, because every symbol earns its place:
- $\arg\max_\pi$ — "search over all possible policies (ways of choosing actions), and pick the one that..."
- $\mathbb{E}[\cdot]$ — "...maximizes the *expected* value..." — expected, not guaranteed, because the environment is uncertain; the agent doesn't know $s_t$ perfectly and can't predict the future with certainty.
- $\sum_{t=0}^{T} \gamma^t r_t$ — "...of the total reward collected over time," where $\gamma \in [0,1]$ is a **discount factor** that controls how much the agent values a reward now versus the same reward later (this single symbol is doing a lot of work — we'll come back to it directly when we hit reinforcement learning).

This is the formal skeleton. What it deliberately does **not** specify is *how* $\pi$ is found — by a hard-coded rule table, by search, by a learned neural network, by an LLM reasoning step by step. That's exactly the right silence: the definition of rational behavior is separate from the mechanism that produces it, and almost the entire rest of this course is "what are the mechanisms, and why do some generalize far better than others."

Chollet's generalization-efficiency framing adds one more piece worth writing down informally (the full formalism is heavier than we need on Episode 1):

$$\text{Intelligence} \approx \frac{\text{Skill on novel tasks}}{\text{Prior knowledge} + \text{Experience spent}}$$

Not a rigorous equation yet — a ratio to keep in your head. Skilled-but-narrow systems (the calculator, the chess engine, a model that's simply memorized its training set) score low because the denominator is huge relative to how novel the "skill" really is. We'll meet a properly formalized version of this when it becomes useful.

## 3. Code: telling "solves the spec" apart from "generalizes beyond it"

Two tiny systems. Both will "satisfy requirements" on their training cases. Only one will handle a case it never saw.

**3.1 From scratch — a pure lookup-table "solver" (zero generalization)**

```python
# A system that satisfies every requirement it was given...
# by memorizing the exact answer. No understanding, no generalization.

requirements_to_solutions = {
    "2+2": 4,
    "3+5": 8,
    "10-4": 6,
}

def lookup_solver(requirement: str):
    if requirement in requirements_to_solutions:
        return requirements_to_solutions[requirement]
    return None  # anything it wasn't explicitly given, it cannot solve

print(lookup_solver("2+2"))   # 4  -- looks "intelligent"
print(lookup_solver("7+1"))   # None -- never seen this exact requirement, total failure
```

This system passes your original definition perfectly on every case it's ever tested against *by its own author* — because its author only tests it on things already in the table. That's the trap: specification-satisfaction looks like intelligence right up until you test on something novel.

**3.2 From scratch — a system that generalizes from a handful of examples**

```python
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
```

This system was never told the answer for `(0.4, 0.4)`. It generalizes from structural similarity to examples it *was* given. It will sometimes be wrong (nearest-neighbor is a deliberately weak learner — that's coming in Module 03) — but the *kind* of thing it's doing is qualitatively different from the lookup table, and that difference is the entire chapter.

**3.3 Using a library — the same idea, properly implemented**

```python
from sklearn.neighbors import KNeighborsClassifier

X_train = [[0.1, 0.1], [0.9, 0.9], [0.2, -0.1], [1.2, 0.3], [-0.3, 0.2]]
y_train = [1, 0, 1, 0, 1]

clf = KNeighborsClassifier(n_neighbors=1)
clf.fit(X_train, y_train)

print(clf.predict([[0.4, 0.4]]))  # generalizes, same principle, production-grade implementation
```

Same mechanism as 3.2, properly vectorized, handling edge cases and higher dimensions we didn't bother with by hand. This pairing — build the mechanism raw, then see the library do the same thing — is the pattern every episode from here on will follow.

## 4. Where this leaves us

A working definition to carry forward, built from what actually survived the counterexamples:

> **Intelligence is the capacity to select actions that perform well against a goal, under uncertainty, generalizing to situations beyond what was explicitly prepared for — and doing so efficiently relative to the prior knowledge and experience available.**

Hold onto three components specifically, because each becomes its own multi-module arc later in this course:
- **"Select actions under uncertainty"** → the rational-agent/RL formalism from §2, which resurfaces directly when we build agents.
- **"Generalizing beyond what was prepared for"** → the reason *learning* (Module 03 onward) beats hard-coded rules, and the exact axis on which modern LLMs are judged.
- **"Efficiently relative to prior knowledge/experience"** → why "bigger model, more data" isn't automatically "more intelligent," and why sample-efficient learning is an open research problem, not a solved one — directly relevant to the fine-tuning work you already do at AIVerse.

## 5. Before Episode 00.02

Sit with this one, informally:

> Section 3 showed a system that memorizes (lookup table) and a system that generalizes (nearest-neighbor). But *both* of them, ultimately, only ever operate on **numbers** — coordinates, labels. Real requirements arrive as language: "make the customer happy," "summarize this document," "book me a flight." Before we can build anything that reasons about *that*, we need to answer a much more basic question: **how does a word become a number in the first place, and what gets lost — or gained — in that translation?**

That's Episode 00.02, and it's the actual on-ramp into Module 01.

---

**Previous:** Episode 00.00 — Course Introduction and Methodology
**Next:** Episode 00.02 — From Words to Numbers: The Representation Problem

## 6. Reference Papers/Books:
- https://courses.cs.umbc.edu/471/papers/turing.pdf
- http://lib.ysu.am/disciplines_bk/efdd4d1d4c2087fe1cbe03d9ced67f34.pdf
- https://arxiv.org/pdf/1911.01547
- https://www.vetta.org/documents/Machine_Super_Intelligence.pdf