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


def RSA_fmri(groups: dict, roi_masks: dict, models: dict, storing_results: str, recompute_model_correlations: bool, run_start_indices: list = [0, 267, 492, 812, 1137, 1373]):
    """
    Perform RSA on fMRI data per participant, ROI, model.
    Returns:
        results_dict: RSA values (runs x timepoints)
    """
    # Precompute model correlations
    model_corrs = model_correlations(models=models, storing_results=storing_results, recompute=recompute_model_correlations)

    results_dict_timepoint = defaultdict(lambda: defaultdict(dict))
    results_dict_run = defaultdict(lambda: defaultdict(dict))

    for participantgroup, ids in groups.items():
        print(f"Processing participant group {participantgroup}")

        base_path_template = f"data/fMRI data/{participantgroup}/sub-{{subj_id:03d}}.nii.gz"

        for participant_id in ids:
            print(f"\tProcessing participant nr.{participant_id}")
            fmri_path = base_path_template.format(subj_id=participant_id)
            fmri_img = nib.load(fmri_path)

            for roi, mask in roi_masks.items():
                print(f"\t\tProcessing ROI {roi}")
                masked_data = mask_fmri_data(fmri_img, mask, roi)
                fmri_list = [(runs := split_runs(masked_data, run_start_indices, 0))[0][4:-3]] + \
                            [run[3:-3] for run in runs[1:]]

                for model_name, model_list in models.items():
                    print(f"\t\t\tProcessing model {model_name}")
                    L1 = []

                    for i in range(len(fmri_list)):

                        # Compute first-order similarity
                        fmri_corr_vec = first_order_similarity(fmri_list[i])
                        model_corr_vec = model_corrs[model_name][i]

                        # Compute second-order RSA per timepoint
                        T = fmri_list[i].shape[0]
                        timepoint_corrs = compute_per_timepoint_rsa(fmri_corr_vec, model_corr_vec, T)
                        rsa_score = safe_corr(fmri_corr_vec, model_corr_vec)

                        L1.append(timepoint_corrs)

                    results_dict_timepoint[participant_id][roi][model_name] = np.concatenate(L1)  # shape: (runs, T)
                    results_dict_run[participant_id][roi][model_name] = rsa_score


        # Save results_dict and saving for each participant group in the meantime
        with open(storing_results+'/results_pertimepoint.pkl', 'wb') as f:
            pickle.dump(dict(results_dict_timepoint), f)

        with open(storing_results+'/results_perrun.pkl', 'wb') as f:
            pickle.dump(dict(results_dict_run), f)

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
    elif roi=="visual":
        mask_img = math_img(f"img == 1", img=original_mask)
        mask_resampled = resample_to_img(mask_img, fmri_img, interpolation="nearest", force_resample=True, copy_header=True)

        # Apply mask
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


def load_and_split_trimmed(path_or_array, run_start_indices, trim_start=3, trim_end=3, is_path=True):
    """
    Load a model from file or use the provided array,
    split it into runs, and trim edges (extra 1 at start for first run).
    """
    data = np.loadtxt(path_or_array, delimiter=",") if is_path else path_or_array
    runs = split_runs(data, run_start_indices, 0)
    trimmed_runs = [runs[0][trim_start + 1 : -trim_end]] + [r[trim_start:-trim_end] for r in runs[1:]]
    return trimmed_runs

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

    else:
        print("Computing model correlations")
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


def first_order_similarity(arr):
    """
    Compute upper-triangle Spearman correlations between all pairs of timepoints.

    Parameters
    ----------
    arr : ndarray, shape (T, features)
        Time-by-feature matrix.

    Returns
    -------
    sim_vector : ndarray, shape (T*(T-1)/2,)
        Vector of pairwise Spearman correlations.
    """

    # Step 1: Rank transform each row (Spearman = Pearson on ranks)
    ranked = np.apply_along_axis(rankdata, 1, arr)

    # Step 2: Compute correlation matrix (vectorized)
    sim_matrix = np.corrcoef(ranked)

    # Step 3: Handle constant rows (std ≈ 0 → invalid correlations)
    stds = np.std(ranked, axis=1)
    invalid = np.isclose(stds, 0)

    if np.any(invalid):
        sim_matrix[invalid, :] = np.nan
        sim_matrix[:, invalid] = np.nan

    # Step 4: Extract upper triangle (excluding diagonal)
    iu = np.triu_indices_from(sim_matrix, k=1)
    sim_vector = sim_matrix[iu]

    return sim_vector

def safe_corr(a, b):
    """
    Compute Spearman correlation while ignoring NaNs.
    
    Returns nan if less than 2 valid points remain.
    """
    mask = ~np.isnan(a) & ~np.isnan(b)
    if mask.sum() < 2:
        return np.nan
    return spearmanr(a[mask], b[mask]).correlation


def compute_per_timepoint_rsa(fmri_corr_vec, model_corr_vec, T):
    """
    Compute second-order RSA per timepoint.
    
    fmri_corr_vec and model_corr_vec are flattened upper-triangle vectors
    of shape (T*(T-1)/2,)
    
    Returns
    -------
    timepoint_corrs : array, shape (T,)
        Spearman correlation for each timepoint. NaN if not enough valid pairs.
    """
    timepoint_corrs = np.zeros(T)
    
    # Precompute the indices of upper-triangle for each timepoint
    pair_idx = np.triu_indices(T, k=1)
    
    for t in range(T):
        # Select pairs involving timepoint t
        mask = (pair_idx[0] == t) | (pair_idx[1] == t)
        
        # Compute correlation safely
        timepoint_corrs[t] = safe_corr(fmri_corr_vec[mask], model_corr_vec[mask])
    
    return timepoint_corrs

def read_model(filepath):

    with open(filepath) as i:
        vecs = list()
        for l in i:
            line = vecs.append(l.strip().split('\t'))
        vecs = np.array(vecs, dtype=np.float64)
        assert line.shape == (1614, 2048)
    return vecs



