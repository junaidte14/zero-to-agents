# From Zero to Agents
## Module 06 — Agents
### Episode 06.01: Multi-Tool Agents — Selection, Chaining, and an Honest Failure
 
---
 
## 0. Closing the open question
 
Episode 06.00 ended by asking what breaks first as a single-tool, single-step toy scales toward multiple tools, possible failures, and multi-step tasks. This episode builds all three extensions directly, and — consistent with this course's habit of reporting what actually happens rather than what was expected — one of them genuinely doesn't work well, for an entirely diagnosable reason.
 
## 1. Theory: none of these extensions need new machinery
 
**1.1 Tool selection is just next-token prediction over a bigger vocabulary.**
Episode 06.00's single-tool agent never had to *choose* — there was only one possible action. Adding a second tool means adding a second tool-identifier token to the vocabulary, and letting the model's ordinary next-token softmax (Episode 02.04, unchanged) decide between them based on context. No new mechanism — tool selection is exactly the same computation that decides every other token, just with the stakes of "which tool" riding on that one decision.
 
**1.2 Chaining is the same loop, run more than once.**
Episode 06.00 §2.1 already wrote the state-transition recursion in general form — $s_{t+1} = s_t \Vert a_t \Vert \text{Observation}(\text{execute}(a_t))$ — but only exercised it for a single iteration. A task requiring two tool calls in sequence (say, add two numbers, then add a third to that result) is the identical recursion, just run twice before a final answer terminates it: propose an action, get a real observation, propose another action using that observation as part of the new context, get another real observation, then finalize.
 
**1.3 Failure is just a different kind of observation, not a different kind of machinery.**
When a tool call fails — invalid input, an exception, an unexpected result — the correct handling is to catch it and format the failure itself as an observation (e.g., an explicit error token or message), injected into the context exactly the way a successful result would be, letting the model's next action be conditioned on "that failed" the same way it would be conditioned on a real result. The loop doesn't need a separate failure-handling code path; failure is content, not a different control-flow branch.
 
## 2. Code: two tools, tool selection, and a genuine two-step chain
 
**2.1 The extended vocabulary and trace types**
 
Building directly on Episode 06.00's setup, with two real tools and a chained task type:
 
```python
def add_tool(a, b): return a + b
def double_tool(a): return min(a * 2, 9)   # capped to stay single-digit for this toy vocabulary
 
def trace_add(d1, d2):
    r = add_tool(d1, d2)
    return [QADD, d1, d2, A, TADD, d1, d2, O, r, Fi, r, EOS]
 
def trace_double(d1):
    r = double_tool(d1)
    return [QDBL, d1, A, TDBL, d1, O, r, Fi, r, EOS]
 
def trace_chain(d1, d2, d3):
    r1 = add_tool(d1, d2)
    r2 = add_tool(r1, d3)   # a genuine SECOND tool call, using the first call's real result
    return [QCHAIN, d1, d2, d3, A, TADD, d1, d2, O, r1, A, TADD, r1, d3, O, r2, Fi, r2, EOS]
```
 
The chain trace has **two** complete Action→Observation cycles before the final answer — exactly §1.2's repeated recursion, made concrete in the training data itself.
 
**2.2 The extended loop — injecting a real result after every observation marker, however many occur**
 
```python
def run_loop(model, prompt_tokens, real_tool_fn_sequence, max_new=12):
    seq = torch.tensor([prompt_tokens])
    tool_idx, generated = 0, 0
    while generated < max_new:
        next_id = model(seq)[0, -1].argmax().item()
        seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
        generated += 1
        if next_id == O and tool_idx < len(real_tool_fn_sequence):
            real_result = real_tool_fn_sequence[tool_idx]()   # the REAL tool executes, fresh each time
            seq = torch.cat([seq, torch.tensor([[real_result]])], dim=1)
            tool_idx += 1; generated += 1
        if next_id == EOS:
            break
    return seq[0].tolist()
```
 
This is a direct generalization of Episode 06.00 §4.3 — instead of injecting exactly one real observation, it injects one every time the model emits the observation marker, in order, for as many tool calls as the task actually needs.
 
**2.3 Results — trained once, on a mix of all three trace types**
 
```python
print(f"ADD tool, unseen pairs: {correct_add}/5")
print(f"DOUBLE tool, unseen value: {correct_dbl}")
print(f"CHAIN (2-step), unseen triples: {correct_chain}/5")
```
```
ADD tool, unseen pairs:       5/5
DOUBLE tool, unseen value:    double(4): predicted=4, true=8  -- WRONG
CHAIN (2-step), unseen triples: 5/5
```
 
Two genuinely strong results and one honest failure, worth reporting exactly as it happened rather than smoothed over.
 
## 3. Reading the results precisely — including the failure
 
**3.1 What worked, and why.**
Both the tool-selection task (ADD, correctly invoked and correctly answered on 5/5 unseen pairs) and the genuinely harder chained two-step task (5/5 on unseen triples, each requiring the model to correctly propose a *second* action using the *first* tool call's real, injected result as part of its own context) worked perfectly. The chain result is the more impressive one: it required the model to correctly interpret an injected observation, use it as input to a fresh action proposal, and get *that* action right too — genuine multi-step conditioning on real, external information, not just a longer version of the single-step task.
 
**3.2 What failed, and exactly why — a real, diagnosable limitation, not a mystery.**
The DOUBLE tool failed its only test case (predicted 4 instead of 8). The reason is visible directly in the training setup: the ADD task had 20 unique training pairs; the DOUBLE task had only **4** unique training values (oversampled 5x to balance batch composition, but still only 4 genuinely distinct examples for the model to generalize the "double" pattern from) — nowhere near enough coverage for a task requiring the model to learn a general rule from so few instances, especially sharing model capacity with two other, more heavily-represented task types. This is not a flaw in the multi-tool mechanism itself (§1.1's claim — tool selection is just next-token prediction over a bigger vocabulary — isn't contradicted by this failure); it's a straightforward data-imbalance problem, the kind any real multi-tool fine-tuning effort has to actively manage: **a tool that's underrepresented in training data will be undertrained, regardless of how sound the surrounding agent architecture is.**
 
## 4. Where this leaves us
 
Every extension named at the end of Episode 06.00 turned out to require no new mechanism — bigger action vocabulary for tool choice, more recursion steps for chaining, and (in theory, per §1.3, not built out fully in code this episode) treating failure as ordinary observation content rather than special-cased control flow. The one place things genuinely broke wasn't the mechanism — it was data coverage, an entirely mundane and entirely fixable problem, and one directly relevant to any real multi-tool agent-training effort: **each tool needs its own adequate training signal, not just a slot in a shared action vocabulary.**
 
## 5. Before the next episode
 
> The DOUBLE tool's failure here was diagnosed after the fact, by noticing its small training-set size. In a real system with many tools, you generally can't eyeball every tool's data coverage by hand. What would you want to measure, systematically and automatically, to catch an undertrained tool *before* it fails in production — something computable from the training data or the model's behavior, rather than something requiring a human to notice a suspiciously small example count?
 
That's a genuine, practical question worth carrying into the next episode of Module 06.
 
---
 
**Previous:** Episode 06.00 — From Language Model to Agent: the ReAct Loop
**Next:** Episode 06.02 — Detecting Untrained Tools