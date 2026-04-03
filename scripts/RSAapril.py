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

from utils import RSA_noresidualization

def split_runs(data, indices, spliti):
    """
    Splits the data into multiple runs based on the indices and the specified dimension.
    
    Parameters:
    - data: The input data to be split (can be a 4D array).
    - indices: List of indices marking where the runs start.
    - spliti: The axis (dimension) to split the data along. For fMRI data, spliti would typically be 3 for time.
    
    Returns:
    - runs: A list containing the data split by runs.
    """
    runs = []
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i + 1]
        
        # Dynamically slice the data along the specified dimension (spliti)
        # Using slicing: data[... , start:end] where `spliti` defines which axis is sliced
        slices = [slice(None)] * data.ndim  # Create a list of slices to cover all dimensions
        slices[spliti] = slice(start, end)  # Update the slice for the specified dimension
        
        run_data = data[tuple(slices)]  # Apply the slice to the data
        runs.append(run_data)

    # The last run should end at the last time point
    start = indices[-1]
    slices = [slice(None)] * data.ndim  # Again, create a list of slices for all dimensions
    slices[spliti] = slice(start, None)  # Slice from the last start index to the end
    run_data = data[tuple(slices)]
    runs.append(run_data)

    return runs

def load_and_split_trimmed(path_or_array, run_start_indices, trim_start=3, trim_end=3, is_path=True):
    """
    Load a model from file or use the provided array,
    split it into runs, and trim edges (extra 1 at start for first run).
    """
    data = np.loadtxt(path_or_array, delimiter=",") if is_path else path_or_array
    runs = split_runs(data, run_start_indices, 0)
    trimmed_runs = [runs[0][trim_start + 1 : -trim_end]] + [r[trim_start:-trim_end] for r in runs[1:]]
    return trimmed_runs

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
        "visual": visual_mask
    }

    run_start_indices =[0, 267, 492, 812, 1137, 1373]

    models = {}
    base_path = "data/LLM embeddings/other used LLM embeddings"

    for file in os.listdir(base_path):
        if file.endswith(".1D") or file.endswith(".txt"):
            model_name = os.path.splitext(file)[0]
            full_path = os.path.join(base_path, file)

            models[model_name] = load_and_split_trimmed(
                full_path,
                run_start_indices
            )

    storing_results = "results/april"

    RSA_noresidualization.RSA_fmri(groups, roi_masks, models, storing_results, recompute_model_correlations=True)



if __name__ == "__main__":
    main()