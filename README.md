# From Zero to Agents
## Module 00 — Introduction
### Episode 00.00: Why This Course Exists, and How We're Going to Run It
 
---
 
## 1. What this course actually is
 
This is not a "learn AI in 30 days" course. It is a research apprenticeship in written form.
 
The premise is simple: you cannot build reliable agentic systems on a foundation you don't fully own. Frameworks like LangChain, LlamaIndex, or a raw agent loop calling an LLM API will teach you to *assemble* things. They won't teach you why a transformer attends the way it does, why your embeddings collapse when you fine-tune wrong, or why your agent hallucinates a tool call. Those failures live below the abstraction layer that most tutorials operate at.
 
So we are starting from zero — deliberately, even where it feels slow — and walking all the way up to the frontier: agentic workflows, the systems you already build professionally. The test for whether this course is working is not "did I finish a module" — it's "can I derive this from first principles if the library disappeared tomorrow."
 
## 2. Who "we" are in this course
 
Two roles, one conversation:
 
- **You** bring the questions, the intuition checks, the "wait, why does that work" moments, and the discipline to actually do the exercises instead of skimming them.
- **I** act as research supervisor — not a lecturer reciting facts, but someone who explains, probes your understanding, corrects misconceptions, and pushes you toward the next open question the way a PhD advisor would in a weekly meeting.
This matters for how episodes are written: expect questions directed at you inside the text, not just at the end.
 
## 3. The three-lens method
 
Every concept in this course, without exception, gets explained three ways before we move on:
 
1. **Theoretically** — the intuition. What is this idea trying to capture about the world or about computation? What problem existed before this idea, and what broke without it?
2. **Mathematically** — the formalism. The actual notation, the derivation, the proof sketch where relevant. Not hand-waved — worked through.
3. **Programmatically** — twice:
   - **From scratch in core Python** (no NumPy/PyTorch shortcuts where the point is to see the mechanism) — so the math has nowhere to hide.
   - **Using the standard library/framework** (PyTorch, etc.) — so you can see exactly what the abstraction is doing *for* you, and trust it because you've built the primitive version yourself.
If any one of these three legs is missing for a concept, that episode isn't done.
 
## 4. Structure: modules and episodes
 
- A **module** is a coherent unit of the field (e.g., "Introduction," "Text Representation," "Linear Algebra Foundations," "Neural Networks," "Attention & Transformers," "Agentic Systems").
- An **episode** is one discrete concept within that module, small enough to actually finish in a sitting, with worked examples and code.
- Numbering: `Module 0X / Episode 0X.YY`. This episode is **00.00**.
- Every episode produces:
  1. A markdown file for the [GitHub repo](https://github.com/junaidte14/zero-to-agents).
  2. An adapted, SEO-optimized post for the blog ([blog.codoplex.com](https://blog.codoplex.com)).
  3. Eventually, a chapter in the compiled ebook.

## 5. The starting-from-zero rule
 
You have production ML/LLM engineering experience already — AIVerse's inference routing, LoRA fine-tuning, multi-tenant agentic infra. It would be easy to skip ahead.
 
We're not going to. The explicit goal of this course is to rebuild the foundation deliberately, so that the parts of your knowledge that are currently "I know this works" become "I know why this works, and I could rebuild it." Where your existing background lets us move faster through an episode, we will — but we're not skipping steps, we're moving through them efficiently.
 
## 6. The road ahead — provisional map
 
This will get refined as we go, but here's the shape of the mountain:
 
- **Module 00 — Introduction** *(you are here)*: methodology, what intelligence even means, framing the whole course.
- **Module 01 — Representing Language as Numbers**: from "what is a word to a computer" through tokenization, one-hot encoding, embeddings.
- **Module 02 — Mathematical Foundations**: vectors, matrices, linear transformations, derivatives, gradients, probability — exactly as much as we need, built up from what Module 01 already required.
- **Module 03 — Neural Networks from First Principles**: perceptrons → MLPs → backprop by hand → backprop in PyTorch.
- **Module 04 onward**: sequence models, attention, the transformer architecture, training/fine-tuning (including LoRA, tying directly back to your AIVerse work), and finally agentic workflows — planning, tool use, memory, multi-agent systems.
The exact episode breakdown inside each module will be decided as we finish the one before it — no point over-planning Module 04 before Module 01 exists.
 
## 7. What Episode 01 will require of you
 
Before we move to Ep. 00.01 ("What is Intelligence?"), sit with one question, informally, no need to write anything polished:
 
> If you had to explain to a non-technical friend what makes a system "intelligent" — not "smart," not "fast," but *intelligent* — what would you say, and where does that definition start to break down?
 
Bring your answer (even a rough one) to the next episode. We'll use it as the entry point into a surprisingly deep rabbit hole.
 
---
 
**Next:** Episode 00.01 — What Is Intelligence? (framing the question the entire course tries to answer)

---

## 🤝 Contributing

Contributions, suggestions, corrections, and discussions are welcome.

If you find an issue or would like to improve the course material, feel free to open an issue or submit a pull request.

---

## ⭐ Support the Project

If you find this repository useful:

- ⭐ Star the repository
- 🍴 Fork it
- 📢 Share it with others interested in AI

Your support helps make high-quality AI education freely accessible.

---

## 📄 License

This project is released under the MIT License unless otherwise specified.

---

*Happy Learning!*