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

from utils import tools

def RSA_fmri(groups: dict, roi_masks: dict, models: dict, storing_results: str, run_start_indices: list = [0, 267, 492, 812, 1137, 1373], recompute_model_correlations: bool = False):
    """
    Perform cross-validated RSA on fMRI data per participant, ROI, model.
    Returns:
        results_dict: RSA values (runs x timepoints)
        best_voxels_dict: selected voxel indices per run
    """
    # Precompute model correlations
    model_corrs = tools.model_correlations(models=models, recompute=recompute_model_correlations)

    # Load low-level models
    acoustic_full = np.hstack([
        np.loadtxt("data/lowlevel models/Original setti models/lowlevel_soundenvelope_8pcs_conv.1D", delimiter=","),
        np.loadtxt("data/lowlevel models/Original setti models/lowlevel_soundpowerspectrum_5pcs_conv.1D", delimiter=",")
    ])
    editing = tools.load_and_split_trimmed("data/lowlevel models/Original setti models/editing_4pcs_conv.1D", run_start_indices)
    surprisal = tools.load_and_split_trimmed("data/lowlevel models/convolved_surprisal", run_start_indices)
    acoustic = tools.load_and_split_trimmed(acoustic_full, run_start_indices, is_path=False)

    results_dict = defaultdict(lambda: defaultdict(dict))
    best_voxels_dict = defaultdict(lambda: defaultdict(dict))

    for participantgroup, ids in groups.items():
        print(f"Processing participant group {participantgroup}")

        base_path_template = f"data/fMRI data/{participantgroup}/sub-{{subj_id:03d}}.nii.gz"

        for participant_id in ids:
            print(f"\tProcessing participant nr.{participant_id}")
            fmri_path = base_path_template.format(subj_id=participant_id)
            fmri_img = nib.load(fmri_path)

            for roi, mask in roi_masks.items():
                print(f"\t\tProcessing ROI {roi}")
                masked_data = tools.mask_fmri_data(fmri_img, mask, roi)
                fmri_list = [(runs := tools.split_runs(masked_data, run_start_indices, 0))[0][4:-3]] + \
                            [run[3:-3] for run in runs[1:]]

                for model_name, model_list in models.items():
                    print(f"\t\t\tProcessing model {model_name}")
                    L1 = []
                    BV = []

                    # Prepare confound regressors and residualize
                    confound_list = [np.hstack([editing[i], acoustic[i], surprisal[i][:, None]]) 
                                     for i in range(len(fmri_list))]
                    residualized_fmri_list = tools.cross_validated_residualize(fmri_list, confound_list)

                    for i in range(len(fmri_list)):
                        # Training set: all other runs
                        train_fmri = [residualized_fmri_list[j] for j in range(len(fmri_list)) if j != i]
                        train_model = [model_list[j] for j in range(len(model_list)) if j != i]

                        # Select best voxels based on training data
                        best_voxels = tools.return_ranks_voxels(train_fmri, train_model)
                        BV.append(best_voxels)

                        # Test run fMRI data
                        test_fmri = residualized_fmri_list[i][:, best_voxels]

                        # Compute first-order similarity
                        fmri_corr_vec = tools.first_order_similarity(test_fmri)
                        model_corr_vec = model_corrs[model_name][i]

                        # Compute second-order RSA per timepoint
                        T = test_fmri.shape[0]
                        timepoint_corrs = tools.compute_per_timepoint_rsa(fmri_corr_vec, model_corr_vec, T)
                        L1.append(timepoint_corrs)

                    results_dict[participant_id][roi][model_name] = np.concatenate(L1)  # shape: (runs, T)
                    best_voxels_dict[participant_id][roi][model_name] = BV

        results_path = storing_results+f'/results_{participantgroup}.pkl'
        best_voxels_path = storing_results+f'/best_voxels_{participantgroup}.pkl'

        # Save results_dict
        with open(results_path, 'wb') as f:
            pickle.dump(dict(results_dict), f)

        # Save best_voxels_dict
        with open(best_voxels_path, 'wb') as f:
            pickle.dump(dict(best_voxels_dict), f)

