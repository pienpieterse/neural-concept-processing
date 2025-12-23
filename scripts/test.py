# import pickle

# with open("results/december/model_correlations.pkl", "rb") as f:
#     data = pickle.load(f)

# print(type(data))   # should be dict
# print(data.keys())


import os

path = "data/LLM embeddings/used LLM embeddings"

for name in os.listdir(path):
    print(name)
