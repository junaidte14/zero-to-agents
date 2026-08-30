#Code: the exact same two models from Episode 05.06, evaluated two different ways

#1 Perplexity, computed over the full sequence vs. response-only

""" def perplexity(model, seqs, response_only=True):
    inputs, targets = build_batch(seqs)
    with torch.no_grad():
        logits = model(inputs)
        if response_only:
            logits_eval = logits[:, 3:, :].reshape(-1, vocab_size)
            targets_eval = targets[:, 3:].reshape(-1)
        else:
            logits_eval = logits.reshape(-1, vocab_size)
            targets_eval = targets.reshape(-1)
        ce = F.cross_entropy(logits_eval, targets_eval)
    return math.exp(ce.item()), ce.item()

for name, model in [("masked", model_masked), ("unmasked", model_unmasked)]:
    ppl_resp, _ = perplexity(model, test_seqs, response_only=True)
    ppl_full, _ = perplexity(model, test_seqs, response_only=False)
    print(f"{name:9s} -- response-only PPL: {ppl_resp:.3f}   full-sequence PPL: {ppl_full:.3f}") """


""" masked    -- response-only PPL: 1.000   full-sequence PPL: 148.402
unmasked  -- response-only PPL: 1.042   full-sequence PPL: 9.242

(reference: uniform-random-over-10-digits perplexity = 10.000) """