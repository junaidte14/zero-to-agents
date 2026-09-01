import math
from collections import Counter

# ==========================================
# 1. Setup Data & Memory Store
# ==========================================

past_turns = [
    "user asked about refund policy for damaged items",
    "user asked how to reset their account password",
    "user asked about shipping times to canada",
    "user asked to cancel a subscription plan",
    "user mentioned their favorite color is blue",
]

new_query = "user wants to know about shipping times"

# Define stopwords to be filtered out in section #3
stopwords = {"user", "asked", "their", "to", "about", "wants", "know"}

# Create a master vocabulary from all turns and query
all_text = " ".join(past_turns + [new_query]).lower().split()
vocab = sorted(list(set(all_text)))

# ==========================================
# 2. Embedding Machinery & Vector Operations
# ==========================================

def tokenize(text, stop_words=None):
    """Tokenize text and optionally filter out stopwords."""
    tokens = text.lower().split()
    if stop_words:
        tokens = [t for t in tokens if t not in stop_words]
    return tokens

def turn_vector(text, stop_words=None):
    """
    Constructs a term-frequency vector across the vocabulary,
    emulating Module 00's vector embedding representation.
    """
    tokens = tokenize(text, stop_words)
    counts = Counter(tokens)
    return [counts[word] for word in vocab]

def cosine(v1, v2):
    """Computes cosine similarity between two dense vectors."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

print("New Query Being Tested: user wants to know about shipping times")
# ==========================================
# #2 Retrieval without any adjustment
# ==========================================

print("=== #2 Retrieval without any adjustment ===")
qv = turn_vector(new_query)

ranked_unadjusted = sorted(
    [(t, cosine(qv, turn_vector(t))) for t in past_turns],
    key=lambda x: -x[1]
)

for t, s in ranked_unadjusted:
    print(f"  sim={s:.3f}  {t}")

print("\n" + "="*50 + "\n")

# ==========================================
# #3 The fix Episode 00.04 didn't get to apply, applied now
# ==========================================

print("=== #3 Retrieval with Stopword Filtering Fix ===")
qv_f = turn_vector(new_query, stopwords)

ranked_adjusted = sorted(
    [(t, cosine(qv_f, turn_vector(t, stopwords))) for t in past_turns],
    key=lambda x: -x[1]
)

for t, s in ranked_adjusted:
    print(f"  sim={s:.3f}  {t}")