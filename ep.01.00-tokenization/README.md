# From Zero to Agents
## Module 01 — Representing Language as Numbers
### Episode 01.00: Tokenization — Why "Words" Aren't the Right Unit
 
---
 
## 0. Where we're starting from
 
Every episode of Module 00 quietly relied on `.split()` — text broken into whole words at whitespace, no questions asked. It was a deliberate simplification, flagged twice and never resolved. It's time to resolve it, because the gap between "split on spaces" and what production systems actually do is bigger than it looks, and everything in the rest of this course — embeddings, attention, the transformer's input layer — sits directly on top of whatever we build today.
 
## 1. Theory: why whole-word splitting breaks
 
**1.1 The out-of-vocabulary (OOV) problem.**
A word-level vocabulary is built from a fixed training corpus. Any word not in that corpus — a misspelling, a brand-new coinage, a proper noun, a typo — has no representation at all. The typical fallback is a single `[UNK]` (unknown) token that swallows every such case, which means the model loses all information about what the word actually was. This isn't a rare edge case in production: internet text is full of typos, slang, invented product names, code snippets, and words that didn't exist when the vocabulary was built.
 
**1.2 Word-level vocabularies are also enormous.**
English alone has hundreds of thousands of words once you count inflections (run, runs, running, ran), and morphologically rich languages (Turkish, Finnish, German compounds) can generate effectively unbounded numbers of valid word forms by combining pieces. A word-level vocabulary either misses most of them (OOV problem again) or grows unmanageably large — remember from Episode 00.02/00.04: vocabulary size $V$ directly sets the size of the one-hot space and everything built downstream of it.
 
**1.3 The other extreme — character-level — has its own cost.**
You could sidestep OOV entirely by treating every individual character as a token. No word is ever "unknown," because every word is just a sequence of characters, and the character vocabulary is tiny (dozens of symbols). But this trades one problem for another: sequences become very long (a 5-letter word becomes 5 tokens instead of 1), and — more importantly for everything we'll build starting in Module 03 — the model now has to learn how characters combine into meaningful units *from scratch*, at every layer, rather than getting word-like structure for free. In practice this makes learning meaningfully harder and slower.
 
**1.4 The middle ground: subword tokenization.**
The actual answer used by essentially every modern LLM is to split text into **subword units** — pieces smaller than whole words but usually larger than single characters, chosen so that common words stay whole (or close to it) and rare/unseen words decompose into familiar, meaningful pieces instead of hitting `[UNK]`. "Tokenization" gets split into `token` + `ization`, say — neither an arbitrary character sequence nor a single, potentially-unseen whole word. This is a genuine engineering sweet spot, and **Byte-Pair Encoding (BPE)** is the specific algorithm that made it practical.
 
**1.5 Where BPE actually comes from.**
BPE wasn't invented for NLP. Philip Gage introduced it in 1994 as a general-purpose **data compression** algorithm — repeatedly replace the most frequent pair of adjacent bytes in a file with a new byte that isn't used elsewhere, shrinking the file. Sennrich, Haddow, and Birch adapted the exact same mechanism to text tokenization in 2016 (*"Neural Machine Translation of Rare Words with Subword Units"*), with one twist: instead of compressing a file, you're *building a vocabulary* — the "replacements" you learn along the way become your subword tokens. This is the tokenizer behind GPT-2 onward, in a refined byte-level form we'll get to in §3.
 
**1.6 A close cousin — WordPiece.**
BERT and its relatives use a variant called **WordPiece**, which follows the same iterative-merge structure as BPE but picks which pair to merge differently: instead of picking the single most *frequent* adjacent pair, it picks the pair whose merge most increases the likelihood of the training corpus — roughly, the pair that's most *surprising* to see split apart, not just most common. The mechanics are close enough that understanding BPE thoroughly (which we're about to do) gets you WordPiece for free, conceptually.
 
## 2. Math: formalizing the BPE training algorithm
 
**2.1 Initialization.**
Given a training corpus, first split every word into its individual characters, and append a special end-of-word marker (we'll use `</w>`) so the algorithm can tell "er" at the end of a word from "er" in the middle of one — these are meaningfully different subword contexts (compare "lower" and "era"). The initial vocabulary $V_0$ is just the set of unique characters plus this marker.
 
**2.2 The merge step, formalized.**
At each iteration, compute the frequency of every adjacent symbol pair across the entire (symbol-segmented) corpus:
 
$$\text{freq}(a, b) = \sum_{w \,\in\, \text{corpus}} \text{count}(w) \cdot \big[\text{number of times } a \text{ is immediately followed by } b \text{ in } w\big]$$
 
where $\text{count}(w)$ is how often word $w$ appears in the corpus (a common word contributes its pairs more times than a rare one). Select the single pair $(a^*, b^*) = \arg\max_{(a,b)} \text{freq}(a,b)$, merge every occurrence of it into one new symbol $a^*b^*$, and add that symbol to the vocabulary. Repeat for a fixed number of merges $k$, chosen in advance as a hyperparameter — this is exactly why real tokenizers advertise a fixed vocabulary size (GPT-2: ~50,000; GPT-4-era models: ~100,000+): $|V| = |V_0| + k$.
 
**2.3 A structural note: this is greedy, not globally optimal.**
At every step, BPE picks the single best merge *right now*, with no lookahead to whether a different choice earlier would have led to a better vocabulary $k$ steps later. This is a **greedy algorithm** — cheap to run, no guarantee of the best possible $k$-merge vocabulary in any formal sense. WordPiece's likelihood-based selection criterion, mentioned in §1.6, is one attempt to make the greedy choice slightly smarter at each step; neither approach solves the fundamental greediness. This is a pattern worth flagging now because we will meet it again, formally, once we reach gradient-based training in Module 03 — "greedy and locally reasonable, not globally guaranteed optimal" describes an enormous amount of what actually works in deep learning.
 
## 3. Code: building BPE, then watching a real tokenizer make the same trade-offs
 
**3.1 From scratch — training BPE on a toy corpus**
 
```python
from collections import defaultdict
 
corpus = {"low": 5, "lower": 2, "lowest": 4, "newer": 6, "wider": 3, "new": 2}
 
def word_to_symbols(word):
    return tuple(word) + ("</w>",)
 
vocab = {word_to_symbols(w): freq for w, freq in corpus.items()}
 
def get_pair_freqs(vocab):
    pairs = defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs
 
def merge_vocab(pair, vocab):
    new_vocab = {}
    for symbols, freq in vocab.items():
        new_symbols, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_vocab[tuple(new_symbols)] = freq
    return new_vocab
 
merges = []
for step in range(10):
    pairs = get_pair_freqs(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    merges.append(best)
    print(f"Merge {step + 1}: {best} (freq={pairs[best]})")
```
 
```
Merge 1: ('w', 'e') (freq=12)
Merge 2: ('l', 'o') (freq=11)
Merge 3: ('r', '</w>') (freq=11)
Merge 4: ('we', 'r</w>') (freq=8)
Merge 5: ('n', 'e') (freq=8)
Merge 6: ('w', '</w>') (freq=7)
Merge 7: ('ne', 'wer</w>') (freq=6)
Merge 8: ('lo', 'w</w>') (freq=5)
Merge 9: ('lo', 'we') (freq=4)
Merge 10: ('lowe', 's') (freq=4)
```
 
`'w'+'e'` merges first because "wer" and "we" patterns recur across "lower," "lowest," "newer," "wider" — the algorithm found the shared structure purely from frequency, with no notion of English morphology at all.
 
Now encode words the training corpus never contained, by applying the learned merges **in the order they were learned**:
 
```python
def encode(word, merges):
    symbols = list(word_to_symbols(word))
    for pair in merges:
        i, new_symbols = 0, []
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
                new_symbols.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols
    return symbols
 
for w in ["lowering", "newest", "wow"]:
    print(f"{w!r} -> {encode(w, merges)}")
```
 
```
'lowering' -> ['lowe', 'r', 'i', 'n', 'g', '</w>']
'newest' -> ['ne', 'we', 's', 't', '</w>']
'wow' -> ['w', 'o', 'w</w>']
```
 
None of these three words appeared in training. None hit an `[UNK]` token. "Lowering" decomposed into the learned piece `lowe` plus individual characters for the part the training data genuinely had no basis to predict — exactly the graceful degradation subword tokenization is designed to produce, in contrast to whole-word tokenization's all-or-nothing `[UNK]`.
 
**3.2 Using a library — training a real BPE tokenizer on the same corpus**
 
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
 
with open("toy_corpus.txt", "w") as f:
    f.write(" ".join(" ".join([w] * freq) for w, freq in corpus.items()))
 
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["toy_corpus.txt"], BpeTrainer(vocab_size=40, min_frequency=1, special_tokens=["[UNK]"]))
 
for w in ["lowering", "newest", "wow"]:
    print(f"{w!r} -> {tokenizer.encode(w).tokens}")
```
 
```
'lowering' -> ['lower', 'i', 'n', '[UNK]']
'newest' -> ['ne', 'west']
'wow' -> ['w', 'o', 'w']
```
 
Look closely at `'lowering'`: the `'g'` came out as `[UNK]`. Not a bug — our toy corpus (`low`, `lower`, `lowest`, `newer`, `wider`, `new`) never contains the letter "g" anywhere, so it never entered the base character vocabulary in the first place. **Character-level BPE can still have an OOV problem** — one level down, at the level of characters instead of words — if the base alphabet itself is incomplete. A production system trained on a real corpus wouldn't miss "g," obviously, but the same structural risk exists for anything genuinely absent from training: emoji, rare scripts, novel symbols.
 
**3.3 The production fix: byte-level BPE.**
This is exactly the problem GPT-2 (Radford et al., 2019) solved by changing what the *base* vocabulary is built from. Instead of starting from Unicode characters (an open-ended set — new characters, emoji, and scripts keep getting added), start from raw **bytes**: every possible byte value, 0 through 255, forms the base vocabulary before any merging happens. Since literally any text — any character, in any language, any emoji — can be represented as some sequence of bytes, a byte-level base vocabulary has no possible OOV case, by construction, the same way one-hot's per-word axis guaranteed coverage back in Episode 00.02, but without needing a gigantic vocabulary to do it. This is the exact mechanism behind the tokenizer used in GPT-2 through GPT-4-era models — the same algorithm we built by hand in §3.1, just with a base alphabet that can never be caught missing a symbol.
 
## 4. Where this leaves us
 
We now have a principled way to turn arbitrary text — including text the tokenizer has never seen a single example of — into a bounded, manageable vocabulary of subword tokens, with no `[UNK]` catastrophe and no unbounded vocabulary explosion. This is the actual first step of every LLM's input pipeline, done before a single number reaches an embedding table.
 
Which raises the natural next question, given everything Module 00 built:
 
## 5. Before Episode 01.01
 
> We spent Module 00 building word embeddings — one dense vector per whole word. Now words are being split into subword pieces like `lowe` + `r` + `i` + `n` + `g`. Does the embedding machinery from Module 00 just apply directly to these subword tokens instead of whole words — treat `lowe` the way we treated `king` — or does splitting words into pieces break something about how those embeddings were supposed to work? Think specifically about the co-occurrence/context idea from Episode 00.02: what does "context" even mean for a token that's a word-fragment rather than a whole word?
 
That question is where Episode 01.01 picks up.
 
---
 
**Previous:** Episode 00.04 — Module 00 Wrap-Up
**Next:** Episode 01.01 — Subword Embeddings in Practice