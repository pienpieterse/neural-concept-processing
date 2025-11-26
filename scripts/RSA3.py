import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from utils import thirdRSA
import os

def main():
    run_start_indices = [0, 267, 492, 812, 1137, 1373]

    directory = [
        "data/LLM embeddings/used LLM embeddings"
    ]

    models = {}

    for base_dir in directory:
        for file in os.listdir(base_dir):
            if file.endswith(".txt") and ("embedding+" in file or "layers" in file):
                file_path = os.path.join(base_dir, file)
                print(f"Processing: {file_path}")

                try:
                    data = thirdRSA.load_and_split_trimmed(file_path, run_start_indices)
                    model_label = os.path.splitext(file)[0]
                    models[model_label] = data
                    print(f"Loaded: {model_label}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    try:
        models["binder_abstractness"] = thirdRSA.load_and_split_trimmed(
            "data/semantic models/binder_abstractness_conv.1D",
            run_start_indices
        )

        models["binder_concreteness"] = thirdRSA.load_and_split_trimmed(
            "data/semantic models/binder_concreteness_conv.1D",
            run_start_indices
        )

        print("Semantic models loaded.")

    except Exception as e:
        print(f"Error loading semantic models: {e}")

    stimuli_keys = [
        'binder_concreteness',
        'binder_abstractness',
    ]

    stimuli_models = {key: models[key] for key in stimuli_keys}

    foundationmodel_keys = [
        os.path.splitext(f)[0]
        for f in os.listdir(directory[0])
        if f.endswith(".txt")
    ]

    print(foundationmodel_keys)

    foundationmodel_models = {key: models[key] for key in foundationmodel_keys}

    surprisal = thirdRSA.load_and_split_trimmed("data/lowlevel models/convolved_surprisal", run_start_indices)

    thirdRSA.compute_model_to_model_similarities_timewise_RSA(stimuli_models, foundationmodel_models, surprisal, res=True, save_path = "results/semantic/results_semantic.pkl")


if __name__ == "__main__":
    main()