# %%
import nibabel as nib
import numpy as np
from os.path import join
import math
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
from nilearn import plotting
from nilearn.image import resample_to_img
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph
import cvxpy as cp
from sklearn.model_selection import KFold
from itertools import product
import scipy.io as sio
import h5py
import scipy.sparse as sp


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

mask_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)

mask_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)

mask_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
white_mask = nib.load(mask_path)


back_mask_data = back_mask.get_fdata() > 0
csf_mask_data = csf_mask.get_fdata() > 0
white_mask_data = white_mask.get_fdata() > 0.5
mask = np.logical_and(back_mask_data, ~csf_mask_data)
mask &= ~white_mask_data
nonzero_mask = np.where(mask)
masked_bold = bold_data[nonzero_mask]

print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%")
print('bold_data masked shape:', masked_bold.shape)

# %%
glm_dict = np.load(f'/home/zkavian/thesis_code_git/Optim_fMRI/Optim_fMRI_new/TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]
R2_run1, R2_run2 = glm_dict['R2run'][:,:,:,0], glm_dict['R2run'][:,:,:,1]

beta = beta_run1 #beta_glm.shape
R2 = R2_run1

mask = np.logical_and(back_mask_data, ~csf_mask_data)
nonzero_mask = np.where(mask)
white_mask_flat = white_mask_data[nonzero_mask]
beta = beta_run1[~white_mask_flat]
R2 = R2_run1[~white_mask_flat]

lower_thr, upper_thr = np.nanpercentile(beta, [1, 99])
print(f'low_thr: {lower_thr:.2f}, high_thr: {upper_thr:.2f}') #low_thr: -4.64, high_thr: 4.60
beta_extreme_mask = np.logical_or(beta < lower_thr, beta > upper_thr)
voxels_with_extreme_beta = np.any(beta_extreme_mask, axis=1)

print(f"percentage of voxels with extreme beta values: {np.sum(voxels_with_extreme_beta)/beta.shape[0]*100:.2f}%")


# %%
mask = np.logical_and(back_mask_data, ~csf_mask_data)
mask &= ~white_mask_data
nonzero_mask = np.where(mask)

# %%
beta_volume_filter = np.load("beta_volume_filter.npy")
bold_data = bold_img.get_fdata()

spatial_shape = beta_volume_filter.shape[:-1]
voxels_with_any_nan = np.zeros(spatial_shape, dtype=bool)
voxels_with_all_nan = np.ones(spatial_shape, dtype=bool)

# Sweep the time dimension once
for t in range(beta_volume_filter.shape[-1]):
    frame_nan = np.isnan(beta_volume_filter[..., t])
    voxels_with_any_nan |= frame_nan
    voxels_with_all_nan &= frame_nan

print(np.sum(voxels_with_any_nan), np.sum(voxels_with_all_nan))

n_trial = beta_volume_filter.shape[-1]
beta_volume_filter_2d = beta_volume_filter.reshape(-1, n_trial)
print(beta_volume_filter_2d.shape)
mask_2d = voxels_with_all_nan.reshape(-1)
beta_valume_clean_2d = beta_volume_filter_2d[~mask_2d]
print(beta_valume_clean_2d.shape)

###
def calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, trial_indices=None, trial_len=9):
    num_trials = beta_valume_clean_2d.shape[-1]
    trial_idx = np.arange(num_trials) if trial_indices is None else np.unique(np.asarray(trial_indices, int).ravel())

    # ----- L_task (same idea as yours) -----
    beta_selected = beta_valume_clean_2d[:, trial_idx]
    counts = np.count_nonzero(np.isfinite(beta_selected), axis=-1)
    sums = np.nansum(beta_selected, axis=-1, dtype=np.float64)
    mean_beta = np.zeros(beta_selected.shape[0], dtype=np.float32)
    m = counts > 0
    mean_beta[m] = (sums[m] / counts[m]).astype(np.float32)
    L_task = np.zeros_like(mean_beta, dtype=np.float32)
    v = np.abs(mean_beta) > 0
    L_task[v] = (1.0 / np.abs(mean_beta[v])).astype(np.float32)

    # ----- reshape BOLD into trials -----
    bold_data_reshape = bold_data.reshape(-1, bold_data.shape[-1])
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

    # ----- L_var: variance of trial differences, as sparse diagonal -----
    diff_mat = np.diff(X, axis=1)                            # [Nvox, Ntrials-1, T]
    diff_mat = diff_mat.reshape(diff_mat.shape[0], -1)       # [Nvox, (Ntrials-1)*T]
    diff_mat = diff_mat[:, np.all(np.isfinite(diff_mat), axis=0)]
    if diff_mat.shape[1] < 2:
        raise ValueError("Not enough finite samples to compute L_var")
    var_vec = np.nanvar(diff_mat, axis=1, ddof=1).astype(np.float32)  # per-voxel variance
    L_var = sp.diags(var_vec, format='csc')                           # sparse diagonal (PSD)

    # ----- L_smooth: sparse 6-neighbor Laplacian on the voxel grid -----
    mask3d = (~mask_2d).reshape(bold_data.shape[:3])
    idx = -np.ones(mask3d.shape, dtype=np.int64)
    idx[mask3d] = np.arange(mask3d.sum())
    rows, cols = [], []
    for ax in range(3):
        s1 = [slice(None)]*3; s2 = [slice(None)]*3
        s1[ax] = slice(1, None); s2[ax] = slice(0, -1)
        a, b = idx[tuple(s1)], idx[tuple(s2)]
        m = (a != -1) & (b != -1)
        i, j = a[m].ravel(), b[m].ravel()
        rows.append(np.concatenate([i, j])); cols.append(np.concatenate([j, i]))
    if rows:
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        A = sp.coo_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(idx.max()+1, idx.max()+1)).tocsr()
        d = np.asarray(A.sum(axis=1)).ravel()
        L_smooth = (sp.diags(d, 0) - A).astype(np.float32).tocsc()
    else:
        L_smooth = sp.csc_matrix((idx.max()+1, idx.max()+1), dtype=np.float32)

    selected_BOLD_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    return L_task.astype(np.float32), L_var, L_smooth, selected_BOLD_flat

def objective_func(w, L_task, L_var, L_smooth, alpha_var, alpha_smooth):
    quad = (w.T @ np.diag(L_task) @ w + alpha_var * (w.T @ L_var @ w) + alpha_smooth * (w.T @ L_smooth @ w))
    return quad

def optimize_voxel_weights(L_task, L_var, L_smooth, alpha_var, alpha_smooth):
    L_total = np.diag(L_task) + alpha_var * L_var + alpha_smooth * L_smooth
    n = L_total.shape[0]
    L_total = np.nan_to_num(L_total)
    L_total = 0.5*(L_total + L_total.T) + 1e-8*np.eye(n)
    w = cp.Variable(n, nonneg=True)
    constraints = [cp.sum(w) == 1]

    # objective = cp.Minimize(cp.quad_form(w, L_total) + alpha_sparse * cp.norm1(w))
    objective = cp.Minimize(cp.quad_form(w, L_total))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=True)
    return w.value

def calculate_weight(param_grid, beta_valume_clean_2d, bold_data, anat_img, mask_2d, trial_len):
    kf = KFold(n_splits=2, shuffle=True, random_state=0)
    best_score = np.inf
    best_params = None
    num_trials = beta_valume_clean_2d.shape[-1]

    for a_var, a_smooth in product(*param_grid.values()):
        fold_scores = []
        print(f"a_var: {a_var}, a_smooth: {a_smooth}")
        count = 1

        for train_idx, val_idx in kf.split(np.arange(num_trials)):
            print(f"k-fold num: {count}")
            L_task_train, L_var_train, L_smooth_train, _ = calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, train_idx, trial_len)
            w = optimize_voxel_weights(L_task_train, L_var_train, L_smooth_train, alpha_var=a_var, alpha_smooth=a_smooth)

            L_task_val, L_var_val, L_smooth_val, _ = calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, val_idx, trial_len)

            fold_scores.append(objective_func(w, L_task_val, L_var_val, L_smooth_val, a_var, a_smooth))
            print(f"fold_scores: {fold_scores}")
            count += 1

        mean_score = np.mean(fold_scores)
        print(mean_score)
        if mean_score < best_score:
            best_score = mean_score
            best_params = (a_var, a_smooth)

    print("Best parameters:", best_params, "with CV loss:", best_score)
    return best_params, best_score

param_grid = {
    "alpha_var":   [1.0],
    "alpha_smooth":[1.0]}

trial_len = 9
best_params, best_score = calculate_weight(param_grid, beta_valume_clean_2d, bold_data, anat_img, mask_2d, trial_len)