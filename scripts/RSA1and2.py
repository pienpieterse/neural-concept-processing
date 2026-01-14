import nibabel as nib
import numpy as np

from nilearn.maskers import NiftiMasker
from nilearn.image import resample_to_img, math_img

from scipy.stats import spearmanr, rankdata

from sklearn.linear_model import LinearRegression
import os
import pickle
from collections import defaultdict

import os
import pickle
from scipy.stats import spearmanr

from utils import tools, RSA

def main():
    ctrlAV = [12, 13, 14, 15, 16, 17, 18, 19, 22, 32]
    ctrlA = [3, 4, 5, 6, 7, 8, 9, 10, 11, 27]
    blind = [33, 35, 36, 38, 39, 41, 42, 43, 53]

    groups = {'blind': blind
            ,'ctrlA': ctrlA
            ,'ctrlAV': ctrlAV}
    
    language_mask = nib.load("data/ROI masks/allParcels-language-SN220.nii")
    visual_mask = nib.load("data/ROI masks/both_vision-areas-full_mask.nii")

    # Create dictionary
    roi_masks = {
        **{i: language_mask for i in range(2, 6)},  # ROIs 2–5 all map to language_mask (where ROI 1 and 2 are taken together)
        # "body": body_mask,
        # "object": object_mask,
        # "scene": scene_mask,
        # "face": face_mask
        "visual": visual_mask
    }

    run_start_indices =[0, 267, 492, 812, 1137, 1373]

    base_dirs = [
    "data/LLM embeddings/QWEN-omni"
    ]

    # Dictionary to store all L1 similarity scores of the models
    models = {}

    for base_dir in base_dirs:
        for file in os.listdir(base_dir):
            if file.endswith((".txt", ".1D")):
                file_path = os.path.join(base_dir, file)
                model_label = os.path.splitext(file)[0]

                print(f"Processing: {file_path}")

                try:
                    data = tools.load_and_split_trimmed(file_path, run_start_indices)
                    print(f"Loaded with load_and_split_trimmed: {model_label}")
                except Exception as e1:
                    print(f"Primary loader failed for {model_label}: {type(e1).__name__}")
                    try:
                        data = tools.read_model(file_path)
                        print(f"Loaded with read_model: {model_label}")
                    except Exception as e2:
                        print(f"Fallback loader also failed for {model_label}: {type(e2).__name__}")
                        continue  # skip this file entirely

                models[model_label] = data


    # models["binder_abstractness"] = tools.load_and_split_trimmed(
    #     "data/semantic models/binder-abstractness-contextualized_conv.1D",
    #     run_start_indices
    # )

    # models["binder_concreteness"] = tools.load_and_split_trimmed(
    #     "data/semantic models/binder-concreteness-contextualized_conv.1D",
    #     run_start_indices
    # )

    # models["word2vec"] = tools.load_and_split_trimmed(
    #     "data/lowlevel models/Original setti models/highlevel_word2vec_72pcs_conv.1D",
    #     run_start_indices
    # )

    #path to where the results are stored
    storing_results = "results/january/qwen-omni"

    RSA.RSA_fmri(groups, roi_masks, models, storing_results, recompute_model_correlations=True)



if __name__ == "__main__":
    main()