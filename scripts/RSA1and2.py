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

from ..utils import tools



ROIs = list(range(2, 6)) + ["body", "object", "scene", "face"]

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

ctrlAV = [12, 13, 14, 15, 16, 17, 18, 19, 22, 32]
ctrlA = [3, 4, 5, 6, 7, 8, 9, 10, 11, 27]
blind = [33, 35, 36, 38, 39, 41, 42, 43, 53]

groups = {'blind': blind
          ,'ctrlA': ctrlA
          ,'ctrlAV': ctrlAV}


newcorrelations = True
storing_results = "results"


run_start_indices = [0, 267, 492, 812, 1137, 1373]

base_dirs = [
    "/Volumes/SSD-1TB/thesis paper/data/LLM embeddings/used LLM embeddings/CLIPmultilingualmulti",
    "/Volumes/SSD-1TB/thesis paper/data/LLM embeddings/used LLM embeddings/CLIPmultilingualtext",
    "/Volumes/SSD-1TB/thesis paper/data/LLM embeddings/used LLM embeddings/XLM-roberta",
    "/Volumes/SSD-1TB/thesis paper/data/LLM embeddings/used LLM embeddings/Llama"
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



acoustic_full = np.hstack([
    np.loadtxt("/Volumes/SSD-1TB/UU - thesis/fMRI data/Full_Models/Original/lowlevel_soundenvelope_8pcs_conv.1D", delimiter=","),
    np.loadtxt("/Volumes/SSD-1TB/UU - thesis/fMRI data/Full_Models/Original/lowlevel_soundpowerspectrum_5pcs_conv.1D", delimiter=",")
])

editing = tools.load_and_split_trimmed("/Volumes/SSD-1TB/UU - thesis/fMRI data/Full_Models/Original/editing_4pcs_conv.1D", run_start_indices)
surprisal = tools.load_and_split_trimmed("/Volumes/SSD-1TB/UU - thesis/fMRI data/convolved_surprisal", run_start_indices)
acoustic = tools.load_and_split_trimmed(acoustic_full, run_start_indices, is_path=False)

model_corr_path = storing_results + "/model_correlations.pkl"

if os.path.exists(model_corr_path) and newcorrelations is False:
    print("Loading existing model correlations")
    with open(model_corr_path, 'rb') as f:
        model_correlations = pickle.load(f)
else:
    print("Computing model correlations")
    model_correlations = {}

    for model_name, model_list in models.items():
        per_run = []

        for i in range(len(model_list)):
            test_model = model_list[i]
            L2b = []

            num_timepoints = test_model.shape[0]

            for t1 in range(num_timepoints):
                for t2 in range(t1 + 1, num_timepoints):
                    if test_model.ndim == 1:
                        # 1D: Use negative absolute distance
                        dist = -abs(test_model[t1] - test_model[t2])
                        L2b.append(dist)
                    else:
                        # 2D: Use Spearman correlation
                        corr = spearmanr(test_model[t1, :], test_model[t2, :]).correlation
                        L2b.append(corr)

            per_run.append(L2b)

        model_correlations[model_name] = per_run

    with open(model_corr_path, 'wb') as f:
        pickle.dump(model_correlations, f)

for participantgroup, ids in groups.items():
    print(f"Processing participant group {participantgroup}")
    results_dict = defaultdict(lambda: defaultdict(dict))
    best_voxels_dict = defaultdict(lambda: defaultdict(dict))

    # Fill in only `participantgroup`, keep {subj_id:03d} as a template
    base_path_template = f"/Volumes/SSD-1TB/UU - thesis/fMRI data/{participantgroup}/sub-{{subj_id:03d}}.nii.gz"

    for participant_id in ids:
        print(f"\tProcessing participant nr.{participant_id}")
        fmri_path = base_path_template.format(subj_id=participant_id)
        fmri_img = nib.load(fmri_path)

        # Loop over ROIs
        for roi, mask in roi_masks.items():
            print(f"\t\tProcessing roi {roi}")

            masked_data = tools.mask_fmri_data(fmri_img, mask, roi)
            fmri_list = [(runs := tools.split_runs(masked_data, run_start_indices, 0))[0][4:-3]] + [run[3:-3] for run in runs[1:]]

            for model_name, model_list in models.items():
                if model_list[0].ndim == 1:
                    spearman = False
                else:
                    spearman = True

                print(f"\t\t\tProcessing model {model_name}")

                L1 = [] # Store per-participant, per roi, per model correlations
                BV = [] # Store per participant, per roi, per model best voxels

                confound_list = [np.hstack([editing[i], acoustic[i], surprisal[i].reshape(-1, 1)]) for i in range(len(fmri_list))]
                residualized_fmri_list = tools.cross_validated_residualize(fmri_list, confound_list)

                #Each run becomes a test set once
                for i in range(len(fmri_list)):
                    print(f"\t\t\t\tRun nr.{i} is now the testset")
                    
                    # Split residualized data
                    train_fmri = [residualized_fmri_list[j] for j in range(len(fmri_list)) if j != i]
                    train_model = [model_list[j] for j in range(len(model_list)) if j != i]

                    # Feature selection on residualized training data
                    best_voxels = tools.return_ranks_voxels(train_fmri, train_model)
                    BV.append(best_voxels)

                    # Apply to residualized test run
                    test_fmri = residualized_fmri_list[i][:, best_voxels]

                    # First-order similarity of residualized fMRI data
                    fmri_corr = tools.first_order_similarity(test_fmri)


                    #first order similarity model --> this only needs to be done once for each model (not per participant, roi and group)
                    model_corr = model_correlations[model_name][i] #L2b

                    second_order_corr = spearmanr(fmri_corr, model_corr).correlation
                    print(f"\t\t\t\t\tSecond order correlation: {second_order_corr}")

                    L1.append(second_order_corr)

                L0 = np.mean(L1)
                results_dict[participant_id][roi][model_name] = L0
                best_voxels_dict[participant_id][roi][model_name] = BV


    results_path = storing_results+f'/results_{participantgroup}.pkl'
    best_voxels_path = storing_results+f'/best_voxels_{participantgroup}.pkl'

    # Save results_dict
    with open(results_path, 'wb') as f:
        pickle.dump(dict(results_dict), f)

    # Save best_voxels_dict
    with open(best_voxels_path, 'wb') as f:
        pickle.dump(dict(best_voxels_dict), f)

