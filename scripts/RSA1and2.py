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
    
    language_mask = nib.load("/Volumes/SSD-1TB/UU - thesis/ROIs/allParcels-language-SN220.nii")
    scene_mask = nib.load("/Volumes/SSD-1TB/UU - thesis/ROIs/ventralparcels/cvs_scene_parcels/fROIs-fwhm_5-0.0001.nii")
    body_mask = nib.load("/Volumes/SSD-1TB/UU - thesis/ROIs/ventralparcels/cvs_body_parcels/fROIs-fwhm_5-0.0001.nii")
    object_mask = nib.load("/Volumes/SSD-1TB/UU - thesis/ROIs/ventralparcels/cvs_object_parcels/fROIs-fwhm_5-0.0001.nii")
    face_mask = nib.load("/Volumes/SSD-1TB/UU - thesis/ROIs/ventralparcels/cvs_face_parcels/fROIs-fwhm_5-0.0001.nii")

    # Create dictionary
    roi_masks = {
        **{i: language_mask for i in range(2, 6)},  # ROIs 2–5 all map to language_mask (where ROI 1 and 2 are taken together)
        "body": body_mask,
        "object": object_mask,
        "scene": scene_mask,
        "face": face_mask
    }

    run_start_indices =[0, 267, 492, 812, 1137, 1373]

    base_dirs = [
    "data/LLM embeddings/used LLM embeddings"
    ]

    # Dictionary to store all L1 similarity scores of the models
    models = {}

    # --- Process the renamed LLM model files ---
    for base_dir in base_dirs:
        for file in os.listdir(base_dir):
            if file.endswith(".txt") and ("embedding+" in file or "layers" in file):
                file_path = os.path.join(base_dir, file)
                print(f"Processing: {file_path}")

                try:
                    data = tools.load_and_split_trimmed(file_path, run_start_indices)
                    model_label = os.path.splitext(file)[0]
                    models[model_label] = data
                    print(f"Loaded: {model_label}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    models["binder_abstractness"] = tools.load_and_split_trimmed(
        "/Volumes/SSD-1TB/UU - thesis/fMRI data/conv_models_1614/binder_abstractness_conv.1D",
        run_start_indices
    )

    models["binder_concreteness"] = tools.load_and_split_trimmed(
        "/Volumes/SSD-1TB/UU - thesis/fMRI data/conv_models_1614/binder_concreteness_conv.1D",
        run_start_indices
    )

    storing_results = "results"

    RSA.RSA_fmri(groups, roi_masks, models, storing_results, recompute_model_correlations=True)



if __name__ == "__main__":
    main()