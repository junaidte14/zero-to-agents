import math

vocabulary = ["king", "queen", "man", "woman", "apple", "orange"]
vocab_size = len(vocabulary)
word_to_index = {w: i for i, w in enumerate(vocabulary)}
print(word_to_index)

def one_hot(word):
    vec = [0] * vocab_size
    vec[word_to_index[word]] = 1
    print(word)
    print(vec)
    return vec

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

king, queen, apple = one_hot("king"), one_hot("queen"), one_hot("apple")
print("cos(king, queen) =", cosine_similarity(king, queen))  # 0.0
print("cos(king, apple) =", cosine_similarity(king, apple))  # 0.0 -- identical to the line above!