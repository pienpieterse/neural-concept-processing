import nibabel as nib
import numpy as np

from nilearn.input_data import NiftiMasker
from nilearn.image import resample_to_img, math_img

from scipy.stats import spearmanr, rankdata

from sklearn.linear_model import LinearRegression
import os
import pickle
from collections import defaultdict

import os
import pickle
from scipy.stats import spearmanr

def cross_validated_residualize(fmri_list, confound_list):
    """
    Residualizes each run using confound model trained on other runs.
    Returns a new list of residualized runs.
    """
    residualized_runs = []
    for i in range(len(fmri_list)):
        # Prepare training confounds and fMRI
        train_confounds = np.vstack([confound_list[j] for j in range(len(fmri_list)) if j != i])
        train_fmri = np.vstack([fmri_list[j] for j in range(len(fmri_list)) if j != i])

        # Fit model
        confounds_aug = np.hstack([train_confounds, np.ones((train_confounds.shape[0], 1))])
        betas, _, _, _ = np.linalg.lstsq(confounds_aug, train_fmri, rcond=None)

        # Apply to test confounds
        test_confounds = confound_list[i]
        test_confounds_aug = np.hstack([test_confounds, np.ones((test_confounds.shape[0], 1))])
        predicted = test_confounds_aug @ betas
        residuals = fmri_list[i] - predicted
        residualized_runs.append(residuals)

    return residualized_runs

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

def mask_fmri_data(fmri_img, original_mask, roi=None):
    if roi=="body" or roi=="object" or roi=="scene" or roi=="face":
        mask_data = original_mask.get_fdata()

        # Determine dynamic threshold (≥ 60% of subjects)
        max_subjects_plus_1 = np.max(mask_data)
        estimated_n_subjects = int(max_subjects_plus_1 - 1)
        threshold = int(np.ceil(0.6 * estimated_n_subjects)) + 1  # +1 for the +1 encoding

        # Create binary mask
        binary_mask_data = (mask_data >= threshold).astype(np.uint8)
        voxel_count = np.count_nonzero(binary_mask_data)

        if voxel_count == 0:
            raise ValueError(f"No voxels meet the 60% threshold ({threshold}). Mask is empty.")

        # Create binary mask NIfTI
        binary_mask_img = nib.Nifti1Image(binary_mask_data, affine=original_mask.affine, header=original_mask.header)

        # Resample to fMRI image
        mask_resampled = resample_to_img(binary_mask_img, fmri_img, interpolation="nearest", force_resample=True, copy_header=True)

        # Apply the mask
        masker = NiftiMasker(mask_img=mask_resampled)
        masked_data = masker.fit_transform(fmri_img)
    elif roi==2:
        mask_img = math_img("np.logical_or(img == 1, img == 2)", img=original_mask)
        mask_resampled = resample_to_img(mask_img, fmri_img, interpolation="nearest", force_resample=True, copy_header=True)

        # Apply mask
        masker = NiftiMasker(mask_img=mask_resampled)
        masked_data = masker.fit_transform(fmri_img)

    else:
        mask_img = math_img(f"img == {roi}", img=original_mask)
        mask_resampled = resample_to_img(mask_img, fmri_img, interpolation="nearest", force_resample=True, copy_header=True)

        # Apply mask
        masker = NiftiMasker(mask_img=mask_resampled)
        masked_data = masker.fit_transform(fmri_img)


    return masked_data

def return_ranks_voxels(fmri_list, model_list):
    """
    Rank the correlations of the voxels to the model using vectorized Spearman correlation
    or negative absolute distance for 1D models.

    Args:
    - fmri_list: List of 5 NumPy arrays (each run's fMRI data of shape [timepoints, voxels])
    - model_list: List of 5 NumPy arrays (each run's model time series, shape [timepoints, features] or [timepoints])

    Returns:
    - top_voxel_indices: Numpy array of the indices of the top 10% highest-ranking voxels.
    """
    num_runs = len(fmri_list)
    num_voxels = fmri_list[0].shape[1]
    voxel_ranks = np.zeros((num_runs, num_voxels))

    for i in range(num_runs):
        train_fmri = fmri_list[i]       # Shape: (T, V)
        train_model = model_list[i]     # Shape: (T, F) or (T,)

        if train_model.ndim == 1:
            # 1D case — use negative absolute distance
            model_ts = rankdata(train_model) - np.mean(rankdata(train_model))  # rank and zero-mean
            fmri_ranked = np.apply_along_axis(rankdata, 0, train_fmri) - np.mean(train_fmri, axis=0)

            # Negative absolute difference per voxel
            avg_neg_abs_dist = -np.mean(np.abs(fmri_ranked - model_ts[:, None]), axis=0)
            voxel_ranks[i, :] = rankdata(-avg_neg_abs_dist, method='average')

        else:
            # 2D case — Spearman correlation
            ranked_fmri = np.apply_along_axis(rankdata, 0, train_fmri)
            ranked_model = np.apply_along_axis(rankdata, 0, train_model)

            ranked_fmri -= ranked_fmri.mean(axis=0)
            ranked_model -= ranked_model.mean(axis=0)

            fmri_std = np.linalg.norm(ranked_fmri, axis=0)
            model_std = np.linalg.norm(ranked_model, axis=0)

            valid_voxels = fmri_std > 0
            valid_features = model_std > 0

            if not np.any(valid_voxels) or not np.any(valid_features):
                print(f"Warning: No valid voxels or model features in run {i}")
                voxel_ranks[i, :] = np.zeros(num_voxels)
                continue

            cov = ranked_fmri[:, valid_voxels].T @ ranked_model[:, valid_features]
            fmri_std = fmri_std[valid_voxels]
            model_std = model_std[valid_features]

            correlation_matrix = cov / (fmri_std[:, None] * model_std[None, :])

            if correlation_matrix.size == 0:
                print(f"Warning: Empty correlation matrix in run {i}")
                avg_corrs = np.zeros(valid_voxels.sum())
            else:
                avg_corrs = np.nanmean(correlation_matrix, axis=1)

            # Create full-size average correlation vector
            full_avg_corrs = np.full(num_voxels, -np.inf)
            full_avg_corrs[np.where(valid_voxels)] = avg_corrs

            voxel_ranks[i, :] = rankdata(-full_avg_corrs, method='average')

    average_ranks = np.mean(voxel_ranks, axis=0)
    top_voxel_count = int(0.1 * num_voxels)
    top_voxel_indices = np.argsort(average_ranks)[:top_voxel_count]

    return top_voxel_indices


def load_and_split_trimmed(path_or_array, run_start_indices, trim_start=3, trim_end=3, is_path=True):
    """
    Load a model from file or use the provided array,
    split it into runs, and trim edges (extra 1 at start for first run).
    """
    data = np.loadtxt(path_or_array, delimiter=",") if is_path else path_or_array
    runs = split_runs(data, run_start_indices, 0)
    trimmed_runs = [runs[0][trim_start + 1 : -trim_end]] + [r[trim_start:-trim_end] for r in runs[1:]]
    return trimmed_runs


def first_order_similarity(fmri):
    """
    Compute upper-triangle similarity of fMRI data (timepoints x voxels)
    using Spearman correlation.
    
    Returns flattened upper-triangle vector (length T*(T-1)/2)
    """
    T = fmri.shape[0]
    sim_vector = []
    for t1 in range(T):
        for t2 in range(t1 + 1, T):
            corr = spearmanr(fmri[t1, :], fmri[t2, :]).correlation
            sim_vector.append(corr)
    return np.array(sim_vector)


def model_correlations(models, storing_results="results", recompute=False):
    """
    Compute or load first-order similarity matrices for each model and run.
    Handles both 1D (vector) and 2D (matrix) models.
    """
    model_corr_path = os.path.join(storing_results, "model_correlations.pkl")

    if os.path.exists(model_corr_path) and not recompute:
        print("Loading existing model correlations")
        with open(model_corr_path, 'rb') as f:
            model_corrs = pickle.load(f)
        return model_corrs

    print("Computing model correlations for...")
    model_corrs = {}
    for model_name, model_list in models.items():
        print(f"\t{model_name}")
        per_run = []
        for run_data in model_list:
            if run_data.ndim == 1:
                # 1D model: negative absolute distance
                T = len(run_data)
                sim_vector = [-abs(run_data[t1] - run_data[t2]) 
                              for t1 in range(T) for t2 in range(t1+1, T)]
                per_run.append(np.array(sim_vector))
            else:
                # 2D model: use first_order_similarity
                per_run.append(first_order_similarity(run_data))
        model_corrs[model_name] = per_run

    os.makedirs(storing_results, exist_ok=True)
    with open(model_corr_path, 'wb') as f:
        pickle.dump(model_corrs, f)

    return model_corrs

def compute_per_timepoint_rsa(fmri_corr_vec, model_corr_vec, T):
    """
    Compute second-order RSA per timepoint.
    
    fmri_corr_vec and model_corr_vec are flattened upper-triangle vectors
    of shape (T*(T-1)/2,)
    """
    timepoint_corrs = np.zeros(T)
    # Precompute the indices of upper-triangle for each timepoint
    pair_idx = np.triu_indices(T, k=1)
    for t in range(T):
        # Select pairs involving timepoint t
        mask = (pair_idx[0] == t) | (pair_idx[1] == t)
        timepoint_corrs[t] = spearmanr(fmri_corr_vec[mask], model_corr_vec[mask]).correlation
    return timepoint_corrs