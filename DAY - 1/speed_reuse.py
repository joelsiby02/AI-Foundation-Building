# this notebook helps you understand model load speed
# Compare with day1_embedding.py model load


# Keeps program alive
# Model stays in memory!!
# You can test repeatedly



# """"""
# # import deps
# from sentence_transformers import SentenceTransformer, util
# import numpy as np

# # import model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # create custom sentences
# sentence1  = "hot"
# sentence2 = "Cold" 

# print(sentence1)
# print(sentence2)

# # create vector encoding for the sentences
# vector1 = model.encode(sentence1)
# vector2 = model.encode(sentence2)


# # check for cosin similarity
# cos_sin1_2 = util.cos_sim(vector1, vector2)
# print(cos_sin1_2.item())

# """"


# In case if you dont need to process the model once but execute n number of tasks with loading again
# Run once initlially and reuse fast
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

while True:
    s1 = input("Sentence 1: ")
    s2 = input("Sentence 2: ")

    v1 = model.encode(s1)
    v2 = model.encode(s2)

    print(util.cos_sim(v1, v2).item())