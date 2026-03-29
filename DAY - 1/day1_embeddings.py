# FILE: day1_embeddings.py
# PURPOSE: Learn what embeddings are by building and comparing them
# AUTHOR: Joel Siby
# DATE: 28 / 03 / 2026

# import deps 
from sentence_transformers import SentenceTransformer, util
import numpy as np

# load the embedding model 'all-MiniLM-L6-v2'
# - It's small (22MB) - runs on any laptop
# - It outputs 384 numbers per sentence - easy to work with
# - It's fast - good for learning
# - It's well-tested - millions of downloads
print("Loading the embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("model loaded successfully!")

# CREATE  FIRST EMBEDDING
sentence1 = "The cat sits on the mat"
# The encode() function does the conversion: text -> vector
# The output is a numpy array (list of numbers)
vector1 = model.encode(sentence1)

# Let's examine what we got
print(f"\n--- FIRST SENTENCE ---")
print(f"Sentence: {sentence1}")
print(f"Vector type: {type(vector1)}")
print(f"Vector length: {len(vector1)} numbers")
print(f"First 10 numbers: {vector1[:10]}")
print(f"These numbers represent the 'meaning' of the sentence1")

sentence2 = "The cat is sleeping on the rug"
vector2 = model.encode(sentence2)

# Let's examine what we got
print(f"\n--- SECOND SENTENCE ---")
print(f"Sentence: {sentence2}")
print(f"Vector type: {type(vector2)}")
print(f"Vector length: {len(vector2)} numbers")
print(f"First 10 numbers: {vector2[:10]}")
print(f"These numbers represent the 'meaning' of the sentence2")

sentence3 = "The car design is so beautifull"
vector3 = model.encode(sentence3)

# Let's examine what we got
print(f"\n--- THIRD SENTENCE ---")
print(f"Sentence: {sentence3}")
print(f"Vector type: {type(vector3)}")
print(f"Vector length: {len(vector3)} numbers")
print(f"First 10 numbers: {vector3[:10]}")
print(f"These numbers represent the 'meaning' of the sentence3")


# Similarity Scoring - Cosine Similarity

# 0.85 – 1.00 → almost same meaning  
# 0.70 – 0.85 → clearly similar  
# 0.50 – 0.70 → somewhat related  
# 0.30 – 0.50 → weak relation  
# 0.00 – 0.30 → unrelated  
# < 0       → opposite / very different (rare in text)

# Cosine similarity works best for embeddings because it measures direction (meaning), not magnitude.

# Cosine similarity measures how similar two vectors are
# Formula: dot(A,B) / (norm(A) * norm(B))
# Result: 1.0 = identical meaning, 0.0 = completely different
#         -1.0 = opposite meaning (rare with these models)

print(f"\n--- SIMILARITY RESULTS ---")

score1_2 = util.cos_sim(vector1, vector2)
print(f"(Both about cats - should be HIGH)")
print(score1_2.item())


score1_3 = util.cos_sim(vector1, vector3)
print(f"(Cat vs Car - should be LOW)")
print(score1_3.item())


print(f"\n--- WHAT THIS MEANS ---")
print(f"1. Sentences become lists of {len(vector1)} numbers")
print(f"2. Similar sentences have similar numbers")
print(f"3. Cosine similarity measures how close the numbers are")
print(f"4. This is how AI 'understands' text")