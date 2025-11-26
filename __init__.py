from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import os

run_start_indices = [0, 267, 492, 812, 1137, 1373]

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


# --- Your predefined directories ---
base_dirs = [
    "/Volumes/SSD-1TB/thesis paper/data/LLM embeddings/used LLM embeddings"
]

# Dictionary to store everything
models = {}

# --- Process the renamed LLM model files ---
for base_dir in base_dirs:
    for file in os.listdir(base_dir):
        if file.endswith(".txt") and ("embedding+" in file or "layers" in file):
            file_path = os.path.join(base_dir, file)
            print(f"🔍 Processing: {file_path}")

            try:
                data = load_and_split_trimmed(file_path, run_start_indices)
                model_label = os.path.splitext(file)[0]
                models[model_label] = data
                print(f"✅ Loaded: {model_label}")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")

# --- Add additional hand-specified models ---
try:
    models["binder_abstractness"] = load_and_split_trimmed(
        "/Volumes/SSD-1TB/UU - thesis/fMRI data/conv_models_1614/binder_abstractness_conv.1D",
        run_start_indices
    )

    models["binder_concreteness"] = load_and_split_trimmed(
        "/Volumes/SSD-1TB/UU - thesis/fMRI data/conv_models_1614/binder_concreteness_conv.1D",
        run_start_indices
    )

    print("✅ Additional models loaded.")

except Exception as e:
    print(f"❌ Error loading additional models: {e}")


stimuli_keys = [
    'binder_concreteness',
    'binder_abstractness'#,
    # 'binder_emotion',
    # 'binder_vision',
    # 'editing',
    # 'acoustic',
    # 'visual'
]

stimuli_models = {key: models[key] for key in stimuli_keys}

foundationmodel_keys = [
    "Llama_layers_layers7-11", "CLIPmultilingualmulti_layers11-15", "CLIPmultilingualtext_layers11-15", "XLM-roberta_layers11-15"
    ]

foundationmodel_models = {key: models[key] for key in foundationmodel_keys}


surprisal = load_and_split_trimmed("/Volumes/SSD-1TB/UU - thesis/fMRI data/convolved_surprisal", run_start_indices)