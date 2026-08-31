# From Zero to Agents
## Module 06 — Agents
### Episode 06.00: From Language Model to Agent — the ReAct Loop
 
---
 
## 0. Where we're starting from — and where this course finally arrives at its own title
 
Every model built since Module 04 does one thing: given text so far, predict what comes next. That's a language model. An **agent** does something meaningfully different — it observes an environment, decides on actions (which may include using tools), and incorporates the results of those actions into further decisions, repeated until a task is done. This episode builds the bridge between the two, and the genuinely satisfying part: the bridge requires **no new model architecture at all** — it's built entirely from things this course already has.
 
## 1. Theory: an agent is a loop, not a new kind of network
 
**1.1 The rational-agent formalism, finally made concrete.**
Episode 00.01 §2 formalized a rational agent as a policy $\pi$ selecting actions from percept history to maximize expected reward: $\pi^*(o_1,\ldots,o_t) \to a_t$. That equation sat unused, mostly abstract, for the rest of the course. Here it becomes literal: $\pi$ is the exact causal transformer built in Module 04, the "percepts" are the growing text context (everything generated and observed so far), and an "action" is either a tool call or a final answer — both are just tokens the model generates, exactly the same mechanism Episode 04.05's tiny GPT used to generate `"abcabc..."`.
 
**1.2 ReAct — interleaving reasoning and acting in one generation stream.**
Yao et al. (2022), in *"ReAct: Synergizing Reasoning and Acting in Language Models,"* proposed a specific pattern: have the model generate **Thought** (reasoning about what to do), **Action** (a tool call), and — critically — receive an **Observation** (the tool's real result) back into its own context before continuing, repeating this Thought→Action→Observation cycle until it emits a final answer. The reasoning step lets the model plan and interpret; the acting step grounds that reasoning in real information instead of the model's own (potentially wrong) internal guesses.
 
**1.3 The crucial mechanical detail — this is an inference-time loop, not a training algorithm.**
Nothing about ReAct changes how the underlying model is trained. The "loop" is ordinary code, external to the model entirely: generate tokens until a stop condition (an action is complete), parse out what tool was requested, **actually execute that tool in real code**, format its real result as text, append it to the context, and resume generation from there. The model never generates the observation itself — it's injected from outside, by the real tool call, which is the entire point: the model's own output for "what did the tool return" would just be a guess; the real tool's output is fact. Section 4 builds and runs this loop exactly, with a genuine Python function standing in for the tool.
 
## 2. Math: the loop formalized precisely, as an extension of Episode 00.01's equation
 
**2.1 State, action, and transition, precisely.**
Let $s_t$ be the full text context so far (every prior thought, action, and observation, concatenated). The model proposes an action by sampling from its own next-token distribution — literally Episode 04.05's autoregressive generation — until a complete action is produced: $a_t \sim \pi_{\text{LLM}}(\cdot \mid s_t)$. If $a_t$ is a tool call, the environment transition is:
 
$$s_{t+1} = s_t \,\Vert\, a_t \,\Vert\, \text{Observation}(\text{execute}(a_t))$$
 
($\Vert$ denoting concatenation) — the real tool executes, its real output gets appended, and generation resumes from this new, longer context. If $a_t$ is instead a final answer, the loop terminates. This is Episode 00.01's rational-agent recursion, with $\pi$ instantiated as an actual trained causal transformer and the "environment" instantiated as real callable functions.
 
**2.2 Where the agent's competence actually comes from.**
Worth being precise about this: nothing in the basic ReAct loop trains $\pi$ using the reward signal directly — there's no reinforcement learning happening in this pattern by default. Whatever ability the model has to produce sensible thoughts and well-formed actions comes entirely from what shaped its next-token distribution *before* this loop ever runs: pretraining (Module 04) and any fine-tuning (Module 05) that specifically taught it to produce this kind of reasoning-and-acting text. Section 4 makes this concrete: our tiny model is explicitly *trained* — using Episode 05.05's exact masked cross-entropy machinery — to produce well-formed ReAct-style traces, before the loop is ever run on it.
 
## 3. Decoding real notation — traces, and the practical evolution into tool-calling APIs
 
Papers and technical write-ups describing ReAct-style agents typically show a **trace** — an example transcript with labeled `Thought:`, `Action:`, `Observation:` lines — as the primary way of communicating what the pattern looks like, more than a dense equation (consistent with §2.1 being a fairly direct formalization of something that's easiest to just show an example of). Real production systems — including the kind of LLM inference routing and agentic workflow work this course has referenced before — have mostly moved from free-text `Action: toolname[args]` parsing (fragile — a slightly malformed string breaks the loop) to **structured function-calling / tool-calling APIs**, where the model outputs a structured object (commonly JSON) specifying the tool and arguments directly, validated by a schema rather than parsed from prose. The underlying loop — generate a proposed action, execute a real tool, inject the real result, resume — is identical either way; only the *format* of the action, and how reliably it can be parsed, has changed.
 
## 4. Code: training a real (tiny) agent, and running the real loop
 
**4.1 A minimal ReAct-style vocabulary and trace format**
 
Ten symbols: digits 0–4, plus `Q` (question start), `A` (action/tool-call marker), `O` (observation marker), `F` (final-answer marker), `EOS`. A full training trace for asking the agent to add two numbers: `[Q, d1, d2, A, d1, d2, O, result, F, result, EOS]` — the model needs to learn to propose the right action, and then, having seen the real observation, copy it forward as the final answer.
 
```python
def add_tool(a, b): return a + b   # a REAL Python function -- the tool, not something the model generates
 
def make_trace(d1, d2):
    r = add_tool(d1, d2)
    return torch.tensor([Q, d1, d2, A, d1, d2, O, r, Fi, r, EOS])
```
 
**4.2 Training — with loss masking, exactly as Episode 05.05–05.06 established**
 
The `Q, d1, d2` question portion is randomly generated per example, exactly the same "fundamentally unpredictable prompt" situation Episode 05.06 diagnosed — so the same fix applies directly:
 
```python
inputs, targets = build_batch(train_pairs)   # 20 (d1,d2) pairs, tiny causal transformer from Ep04.05
masked_targets = targets.clone()
masked_targets[:, :2] = -100   # mask predicting the random question digits, per Episode 05.05
for step in range(3000):
    logits = model(inputs)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), masked_targets.reshape(-1), ignore_index=-100)
    opt.zero_grad(); loss.backward(); opt.step()
```
```
Final training loss, UNMASKED: 0.2996   (plateaus, exactly the entropy-floor pattern from Episode 05.06)
Final training loss, MASKED:   0.0000118
```
 
Masking drives training loss down by four orders of magnitude — the same mechanism, the same reason, verified again in a genuinely different (agentic) context than where it was first discovered.
 
**4.3 The real agent loop — generation interrupted by a real tool call**
 
```python
def run_agent_loop(model, d1, d2):
    seq = torch.tensor([[Q, d1, d2, A, d1, d2, O]])   # up through the proposed action
    tool_result = add_tool(d1, d2)                      # the REAL tool executes -- not the model
    seq = torch.cat([seq, torch.tensor([[tool_result]])], dim=1)   # its REAL result is injected
    for _ in range(3):                                   # resume generation from the real observation
        logits = model(seq)
        next_id = logits[0, -1].argmax().item()
        seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
        if next_id == EOS:
            break
    return seq[0].tolist()
```
 
**4.4 Testing on genuinely unseen digit pairs — never encountered during training**
 
```python
correct = 0
for d1, d2 in test_pairs:   # 5 pairs excluded entirely from training
    result_seq = run_agent_loop(model, d1, d2)
    f_idx = result_seq.index(Fi)
    predicted = result_seq[f_idx + 1]
    correct += int(predicted == d1 + d2)
print(f"{correct}/{len(test_pairs)} correct on unseen tool-use tasks")
```
```
4/5 correct on unseen tool-use tasks
```
 
Reported honestly: **4 out of 5**, not a perfect score, and worth being precise about what this small model actually learned. It wasn't taught arithmetic — it never needed to be, because the real answer was injected by the real tool at every step, training and inference alike. What it *did* need to learn is the more modest, more honestly-scoped skill of correctly copying the injected observation forward into the final-answer slot, on genuinely novel digit combinations it never saw paired together during training — and it does this correctly four times out of five, a real, if imperfect, example of exactly the mechanism (generate → real tool executes → real result injected → resume) that underlies every production tool-using agent, including ones vastly larger and more capable than this toy.
 
## 5. Where this leaves us
 
Nothing in this episode required a new architecture, a new training algorithm, or a new mathematical tool. Episode 00.01's rational-agent equation, dormant since the first module, is now literally instantiated; Episode 04.05's autoregressive generation loop is the mechanism proposing actions; Episode 05.05's loss masking is what made training the trace format tractable at all. "Agent" turns out to be less a new kind of model and more a specific way of *using* everything this course has already built — a loop around a language model, with the crucial discipline of never letting the model's own guess substitute for a real tool's real answer.
 
## 6. Before the next episode
 
> Section 4's agent used exactly one tool, always available, with no possibility of failure. Real environments have multiple tools to choose between, tools that can fail or return unexpected results, and tasks that may require several tool calls chained together rather than just one. What do you think breaks first as this toy setup is scaled toward that reality — the model's ability to *choose* the right tool among several, its ability to *recover* from a failed or unexpected observation, or something about the loop itself?
 
That's the on-ramp into the next episode of Module 06 — multi-tool agents, and what happens when the environment doesn't cooperate.
 
---
 
**Previous:** Module 05, Episode 05.08 — A Complete LoRA Pipeline (Module 05 wrap)
**Next:** Episode 06.01 — Multi-Tool Agents and Handling Failure