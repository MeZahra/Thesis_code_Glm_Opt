# %%
import nibabel as nib
import numpy as np
from os.path import join
import math
import cvxpy as cp
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

# %%
ses = 1
sub = '04'
run = 1

base_path = '/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'
anat_img = nib.load(f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')

data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'
BOLD_path_org = join(base_path, f'sub-pd0{sub}', f'ses-{ses}', 'func', data_name)
bold_img = nib.load(BOLD_path_org)
bold_data = bold_img.get_fdata()
bold_data = bold_data.astype(np.float16)

mask_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)
back_mask = back_mask.get_fdata()
back_mask = back_mask.astype(np.float16)

mask_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
csf_mask = csf_mask.get_fdata()
csf_mask = csf_mask.astype(np.float16)

mask_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
white_mask = nib.load(mask_path)
white_mask = white_mask.get_fdata()
white_mask = white_mask.astype(np.float16)

# %%
back_mask_data = back_mask > 0
csf_mask_data = csf_mask > 0
white_mask_data = white_mask > 0.5
mask = np.logical_and(back_mask_data, ~csf_mask_data)
mask &= ~white_mask_data
nonzero_mask = np.where(mask)
masked_bold = bold_data[nonzero_mask]

print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%")
print('bold_data masked shape:', masked_bold.shape)

# %%
glm_dict = np.load(f'TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]
R2_run1, R2_run2 = glm_dict['R2run'][:,:,:,0], glm_dict['R2run'][:,:,:,1]

mask = np.logical_and(back_mask_data, ~csf_mask_data)
nonzero_mask = np.where(mask)
white_mask_flat = white_mask_data[nonzero_mask]
beta = beta_run1[~white_mask_flat]
print("Beta Range:[", np.nanmin(beta), np.nanmax(beta), "], Mean: ", np.nanmean(beta))

lower_thr, upper_thr = np.nanpercentile(beta, [1, 99])
print(f'low_thr: {lower_thr:.2f}, high_thr: {upper_thr:.2f}') #low_thr: -4.64, high_thr: 4.60
beta_extreme_mask = np.logical_or(beta < lower_thr, beta > upper_thr)
voxels_with_extreme_beta = np.any(beta_extreme_mask, axis=1)

print(f"percentage of voxels with extreme beta values: {np.sum(voxels_with_extreme_beta)/beta.shape[0]*100:.2f}%")

mask = np.logical_and(back_mask_data, ~csf_mask_data)
mask &= ~white_mask_data
nonzero_mask = np.where(mask)

# after removing voxels with extreme beta values
clean_beta = beta[~voxels_with_extreme_beta]

# %%
beta_volume_filter = np.load("beta_volume_filter.npy")
beta_volume_filter.shape

# %%
beta_volume_filter = beta_volume_filter.astype(np.float16)
spatial_shape = beta_volume_filter.shape[:-1]
voxels_with_any_nan = np.zeros(spatial_shape, dtype=bool)
voxels_with_all_nan = np.ones(spatial_shape, dtype=bool)

# Sweep the time dimension once
for t in range(beta_volume_filter.shape[-1]):
    frame_nan = np.isnan(beta_volume_filter[..., t])
    voxels_with_any_nan |= frame_nan
    voxels_with_all_nan &= frame_nan

n_trial = beta_volume_filter.shape[-1]
beta_volume_filter_2d = beta_volume_filter.reshape(-1, n_trial)

mask_2d = voxels_with_all_nan.reshape(-1)
beta_valume_clean_2d = beta_volume_filter_2d[~mask_2d]

beta_valume_clean_2d.shape

# %%
def calculate_matrices(beta_valume_clean_2d, bold_data, mask_2d, trial_indices=None, trial_len=9, num_components=600):
    print("begin")
    print(type(mask_2d))
    num_trials = beta_valume_clean_2d.shape[-1]
    trial_idx = np.arange(num_trials) if trial_indices is None else np.unique(np.asarray(trial_indices, int).ravel())

    # ----- reshape BOLD into trials -----
    bold_data_reshape = bold_data.reshape(-1, bold_data.shape[-1])
    print(bold_data.reshape(-1, bold_data.shape[-1]).shape[0], mask_2d.dtype, mask_2d.size)
    bold_data_selected = bold_data_reshape[~mask_2d]         # keep voxels of interest
    bold_data_selected_reshape = np.zeros((bold_data_selected.shape[0], num_trials, trial_len), dtype=np.float32)
    start = 0
    
    for i in range(num_trials):
        end = start + trial_len
        if end > bold_data_selected.shape[1]:
            raise ValueError("BOLD data does not contain enough timepoints for all trials")
        bold_data_selected_reshape[:, i, :] = bold_data_selected[:, start:end]
        start += trial_len
        if start == 270 or start == 560:   # your skips
            start += 20
    X = bold_data_selected_reshape[:, trial_idx, :]          # [Nvox, Ntrials, T]

    # ----- apply PCA -----
    print("PCA...")
    pca = PCA()
    X_reshap = X.reshape(X.shape[0], -1).astype(np.float32)
    X_pca = pca.fit_transform(X_reshap.T) #(810, 800)

    components = pca.components_[:num_components]
    mean = pca.mean_
    beta_reduced = (beta_valume_clean_2d.T - mean) @ components.T
    beta_reduced = beta_reduced.T


    # ----- L_task (same idea as yours) -----
    print("L_task...")
    beta_selected = beta_reduced[:, trial_idx]
    counts = np.count_nonzero(np.isfinite(beta_selected), axis=-1)
    sums = np.nansum(beta_selected, axis=-1, dtype=np.float64)
    mean_beta = np.zeros(beta_selected.shape[0], dtype=np.float32)
    m = counts > 0
    mean_beta[m] = (sums[m] / counts[m]).astype(np.float32)
    L_task = np.zeros_like(mean_beta, dtype=np.float32)
    v = np.abs(mean_beta) > 0
    L_task[v] = (1.0 / np.abs(mean_beta[v])).astype(np.float32)


    # ----- L_var: variance of trial differences, as sparse diagonal -----
    print("L_var...")
    X_pca = X_pca[:, :600].T
    num_trials = len(trial_idx)
    X = X_pca.reshape(X_pca.shape[0], num_trials, trial_len)
    L_var = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(num_trials-1):
        x1 = X[:, i, :]
        x2 = X[:, i+1, :]
        L_var += (x1-x2) @ (x1-x2).T
    L_var /= (num_trials - 1)

    # selected_BOLD_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    return L_task.astype(np.float32), L_var

# %%
def objective_func(w, L_task, L_var, alpha_var):
    print("Calculating objective...")
    quad = (w.T @ np.diag(L_task) @ w + alpha_var * (w.T @ L_var @ w))
    return quad

# %%
def optimize_voxel_weights(L_task, L_var, alpha_var):
    print("Optimizing voxel weights...")
    L_total = np.diag(L_task) + alpha_var * L_var
    n = L_total.shape[0]
    L_total = np.nan_to_num(L_total)
    L_total = 0.5*(L_total + L_total.T) + 1e-6*np.eye(n)
    eigvals, eigvecs = np.linalg.eigh(L_total)
    eigvals[eigvals < 0] = 0.0  # clip the numerical negatives
    L_total_psd = (eigvecs @ np.diag(eigvals) @ eigvecs.T).astype(np.float64)

    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1]

    # objective = cp.Minimize(cp.quad_form(w, L_total) + alpha_sparse * cp.norm1(w))
    # objective = cp.Minimize(cp.quad_form(w, L_total))
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(L_total_psd)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=True)
    return w.value


# %%
def calculate_weight(param_grid, beta_valume_clean_2d, bold_data, mask_2d, trial_len):
    kf = KFold(n_splits=2, shuffle=True, random_state=0)
    best_score = np.inf
    best_params = None
    num_trials = beta_valume_clean_2d.shape[-1]

    for a_var in param_grid["alpha_var"]:
        fold_scores = []
        print(f"a_var: {a_var}")
        count = 1

        for train_idx, val_idx in kf.split(np.arange(num_trials)):
            print(f"k-fold num: {count}")
            print(type(mask_2d))
            L_task_train, L_var_train = calculate_matrices(beta_valume_clean_2d, bold_data, mask_2d, train_idx, trial_len)
            w = optimize_voxel_weights(L_task_train, L_var_train, alpha_var=a_var)

            L_task_val, L_var_val = calculate_matrices(beta_valume_clean_2d, bold_data, mask_2d, val_idx, trial_len)

            fold_scores.append(objective_func(w, L_task_val, L_var_val, a_var))
            print(f"fold_scores: {fold_scores}")
            count += 1

        mean_score = np.mean(fold_scores)
        print(mean_score)
        if mean_score < best_score:
            best_score = mean_score
            best_params = (a_var)

    print("Best parameters:", best_params, "with CV loss:", best_score)
    return best_params, best_score

# %%
param_grid = {"alpha_var":   [0.1, 0.5, 0.9, 1.0, 2]}

import scipy.sparse as sp
trial_len = 9
best_params, best_score = calculate_weight(param_grid, beta_valume_clean_2d, bold_data, mask_2d, trial_len)

# L_task, L_var, L_smooth, selected_BOLD_data = calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, None, trial_len)
# weights = optimize_voxel_weights(L_task, L_var, L_smooth, alpha_var=best_params[0], alpha_smooth=best_params[1])
# weight_img, masked_weights, y = select_opt_weight(selected_BOLD_data, weights, active_low_var_voxels.astype(bool), affine)
# print(y.shape)

# %%
L_task, L_var = calculate_matrices(beta_valume_clean_2d, bold_data, mask_2d, None, trial_len)
weights = optimize_voxel_weights(L_task, L_var, alpha_var=best_params[0], alpha_smooth=best_params[1])
y = selected_BOLD_data.T @ weights

np.save('weights.npy', weights)
np.save('y.npy', y)
print("Finished!", flush=True)

# %%



