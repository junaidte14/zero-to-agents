""" #2 The result — partial improvement, not a fix

for d1 in dbl_vals:
    r = run_loop(model, [QDBL, d1, A, TDBL, d1], [lambda d1=d1: double_tool(d1)])
    pred = extract_final(r)
    print(f"  double({d1}): pred={pred} true={double_tool(d1)} ok={pred==double_tool(d1)}")


double(0): pred=0 true=0 ok=True  (train)
double(1): pred=2 true=2 ok=True  (train)
double(2): pred=4 true=4 ok=True  (train)
double(3): pred=6 true=6 ok=True  (train)
double(4): pred=6 true=8 ok=False (held-out)

#2 Confirming everything else in the system still works.

print(f"ADD (unseen): {correct_add}/5")
print(f"CHAIN (unseen): {correct_chain}/5")

ADD (unseen): 5/5
CHAIN (unseen): 5/5

Both tools with adequately large domains continue to generalize perfectly under the exact same training regime — confirming this episode's diagnosis is specific to DOUBLE's domain size, not a symptom of some broader problem with the training setup, architecture, or masking approach. """