import numpy as np
import pickle
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

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

def first_order_similarity(fmri):
    L2a = []  # List to store fMRI correlations

    num_timepoints = fmri.shape[0]

    for t1 in range(num_timepoints):
        for t2 in range(t1 + 1, num_timepoints):  # Upper-triangular matrix only (t2 > t1)
            
            # fMRI correlation: Compute correlation between voxel activity at t1 and t2
            fmri_corr = spearmanr(fmri[t1, :], fmri[t2, :]).correlation
            L2a.append(fmri_corr)

    return L2a

def first_order_similarity(X):
    # X shape: (timepoints, features)
    X = np.asarray(X)
    std = X.std(axis=1, keepdims=True)

    # Avoid divide-by-zero: mark zero-variance rows
    zero_var = (std == 0).flatten()

    # Normalize safely
    X_norm = (X - X.mean(axis=1, keepdims=True)) / np.where(std == 0, 1, std)

    # Compute similarity
    sim = np.dot(X_norm, X_norm.T) / X.shape[1]

    # Set similarity rows/cols to NaN where variance is zero
    sim[zero_var, :] = np.nan
    sim[:, zero_var] = np.nan

    return sim


def compute_model_to_model_similarities_timewise_RSA(models_A, models_B, surprisal, res=False, save_path=None):
    """
    Time-resolved RSA between two model sets with optional residualization.
    Concatenates all runs so that correlation is computed per timepoint across full time.
    """
    model_similarities = {}

    for model_a in models_A:
        for model_b in models_B:
            print(f"Comparing {model_a} (A) to {model_b} (B)")
            runs = len(models_A[model_a])
            assert runs == len(models_B[model_b]) == len(surprisal), "Mismatch in run count"

            A_all, B_all = [], []

            # --- residualization and concatenation ---
            for run_idx in range(runs):
                A = models_A[model_a][run_idx]
                B = models_B[model_b][run_idx]
                sur = surprisal[run_idx]

                # Align to shortest shape for safety
                n_timepoints = min(A.shape[0], B.shape[0], len(sur))
                A, B, sur = A[:n_timepoints, :], B[:n_timepoints, :], sur[:n_timepoints]

                if res:
                    # leave-one-run-out residualization
                    other_runs = [i for i in range(runs) if i != run_idx]
                    B_train = np.vstack([surprisal[i].reshape(-1, 1) for i in other_runs])
                    A_train = np.vstack([models_A[model_a][i] for i in other_runs])

                    A_resid = np.empty_like(A)
                    for feat_i in range(A.shape[1]):
                        y_train = A_train[:, feat_i]
                        model = LinearRegression().fit(B_train, y_train)
                        y_pred = model.predict(sur.reshape(-1, 1))
                        A_resid[:, feat_i] = A[:, feat_i] - y_pred
                    A = A_resid

                A_all.append(A)
                B_all.append(B)

            # --- concatenate all runs along time dimension ---
            A_concat = np.vstack(A_all)
            B_concat = np.vstack(B_all)

            # --- Compute similarity matrices ---
            A_sim = first_order_similarity(A_concat)
            B_sim = first_order_similarity(B_concat)

            # --- Correlate each timepoint (row-wise) ---
            n_timepoints = A_sim.shape[0]
            corr_time = np.empty(n_timepoints)
            for t in range(n_timepoints):
                corr_time[t] = spearmanr(A_sim[t, :], B_sim[t, :], nan_policy='omit').correlation

            model_similarities[(model_a, model_b)] = corr_time

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model_similarities, f)
        print(f"Saved time-resolved RSA results to {save_path}")

    return model_similarities
