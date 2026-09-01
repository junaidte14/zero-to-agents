#Code: testing reflection on a case this module already understands deeply

#1 The test — DOUBLE, the exact tool Episode 06.02–06.03 diagnosed as undertrained

""" d1 = 4   # Episode 06.01's actual failure case: double(4) should be 8, model predicted 4

baseline = run_with_prefix([], [QDBL, d1, A, TDBL, d1], [lambda: double_tool(d1)])
print("Baseline final answer:", extract_final(baseline), "(true = 8)")

Baseline final answer: 4  (true = 8)
 """

#2 Adding a reflection-style prefix, and testing whether it changes anything

""" reflection_prefix = [ERR, d1, O, 4]   # a stand-in for "previous attempt on input 4 wrongly gave 4"
with_reflection = run_with_prefix(reflection_prefix, [QDBL, d1, A, TDBL, d1], [lambda: double_tool(d1)])
print("With-reflection final answer:", extract_final(with_reflection), "(true = 8)")

With-reflection final answer: 4  (true = 8)

No change whatsoever. The reflection prefix had exactly zero corrective effect — the model produced the identical wrong answer, with or without it. """