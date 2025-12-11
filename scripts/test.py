import pickle

with open("results/word2vec/model_correlations.pkl", "rb") as f:
    data = pickle.load(f)

print(data)