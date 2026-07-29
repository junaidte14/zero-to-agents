from gensim.models import FastText, Word2Vec
import random

# Reusing the same synthetic royals/commoners corpus generator from Episode 00.03
random.seed(42)
royals, royal_actions, royal_objects = ["king", "queen"], ["rules", "wears", "commands"], ["kingdom", "crown", "throne", "army"]
commoners, commoner_actions, commoner_objects = ["man", "woman"], ["eats", "drinks", "buys"], ["apple", "orange", "water", "bread"]
sentences = []
for _ in range(500):
    if random.random() < 0.5:
        s, a, o = random.choice(royals), random.choice(royal_actions), random.choice(royal_objects)
    else:
        s, a, o = random.choice(commoners), random.choice(commoner_actions), random.choice(commoner_objects)
    sentences.append([s, a, o])

ft_model = FastText(sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100, seed=42, min_n=2, max_n=4)
w2v_model = Word2Vec(sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100, seed=42)

oov_word = "kingly"
print("'kingly' in FastText's known-word list?", oov_word in ft_model.wv.key_to_index)  # False
print(ft_model.wv.most_similar(oov_word, topn=3))