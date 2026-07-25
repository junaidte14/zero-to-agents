from gensim.models import Word2Vec
import random

random.seed(42)
royals = ["king", "queen"]; royal_actions = ["rules", "wears", "commands"]; royal_objects = ["kingdom", "crown", "throne", "army"]
commoners = ["man", "woman"]; commoner_actions = ["eats", "drinks", "buys"]; commoner_objects = ["apple", "orange", "water", "bread"]

sentences = []
for _ in range(500):
    if random.random() < 0.5:
        s, a, o = random.choice(royals), random.choice(royal_actions), random.choice(royal_objects)
    else:
        s, a, o = random.choice(commoners), random.choice(commoner_actions), random.choice(commoner_objects)
    sentences.append([s, a, o])

#print(sentences)

model = Word2Vec(sentences, vector_size=20, window=2, min_count=1, sg=1, epochs=100, seed=42)

print(model.wv.most_similar("king", topn=3))
# [('queen', 0.994), ('kingdom', 0.986), ('crown', 0.985)]
print(model.wv.similarity("king", "apple"))
# 0.74 -- still positive (small synthetic vocabulary), but clearly the weakest relationship