from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

corpus = {"low": 5, "lower": 2, "lowest": 4, "newer": 6, "wider": 3, "new": 2}

with open("toy_corpus.txt", "w") as f:
    f.write(" ".join(" ".join([w] * freq) for w, freq in corpus.items()))

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer.train(["toy_corpus.txt"], BpeTrainer(vocab_size=40, min_frequency=1, special_tokens=["[UNK]"]))

for w in ["lowering", "newest", "wow"]:
    print(f"{w!r} -> {tokenizer.encode(w).tokens}")