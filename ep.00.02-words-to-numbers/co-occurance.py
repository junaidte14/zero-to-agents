import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

corpus = [
    "king rules kingdom",
    "queen rules kingdom",
    "man eats apple",
    "woman eats orange",
    "king eats apple",
    "queen eats orange",
    "man drinks water",
    "woman drinks water",
]

# Step 1: Use CountVectorizer as the Tokenizer & Vocabulary Builder
vectorizer = CountVectorizer()
vectorizer.fit(corpus)

vocab = vectorizer.get_feature_names_out()
print(vocab)
word_to_id = vectorizer.vocabulary_
print(word_to_id)
n_words = len(vocab)

# Convert corpus sentences into lists of integer IDs using CountVectorizer's analyzer
tokenizer = vectorizer.build_analyzer()
tokenized_corpus = [[word_to_id[word] for word in tokenizer(doc) if word in word_to_id] for doc in corpus]
print(tokenized_corpus)

# Step 2: Build Co-occurrence Matrix using a Sliding Window over the IDs
window_size = 2
cooc_matrix = np.zeros((n_words, n_words), dtype=int)

for sentence in tokenized_corpus:
    for i, target_id in enumerate(sentence):
        start = max(0, i - window_size)
        end = min(len(sentence), i + window_size + 1)
        for j in range(start, end):
            if i != j:
                context_id = sentence[j]
                cooc_matrix[target_id, context_id] += 1

# Step 3: Compute Word Cosine Similarity using Scikit-Learn
word_sim = cosine_similarity(cooc_matrix)

# Quick Lookups
king_id, queen_id, apple_id = word_to_id['king'], word_to_id['queen'], word_to_id['apple']
print(f"cos(king, queen) = {word_sim[king_id, queen_id]:.3f}")  # 0.750
print(f"cos(king, apple) = {word_sim[king_id, apple_id]:.3f}")  # 0.447