# %%
import nibabel as nib
import numpy as np
from os.path import join
import math
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import cvxpy as cp
from sklearn.model_selection import KFold
from itertools import product
from sklearn.decomposition import PCA

# %%
import psutil, os
p = psutil.Process(os.getpid())
p.cpu_affinity({20,21,22,23,24,25,26,27,28,29})   # use only core 0

# %%
import numpy as np
import nibabel as nib
from os.path import join

ses = 1
sub = '04'
run = 1

base_path = '/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'

# Load images without forcing full data into memory when not needed
anat_path = f'{base_path}/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz'
anat_img = nib.load(anat_path) 
print(anat_img.shape)

data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'
BOLD_path_org = join(base_path, f'sub-pd0{sub}', f'ses-{ses}', 'func', data_name)
bold_img = nib.load(BOLD_path_org)
bold_data = bold_img.get_fdata()

mask_base = f'{base_path}/sub-pd0{sub}/ses-{ses}/anat'

brain_mask_path = f'{mask_base}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
brain_mask_img = nib.load(brain_mask_path)
back_mask = brain_mask_img.get_fdata()

csf_path = f'{mask_base}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(csf_path).get_fdata()

wm_path = f'{mask_base}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
white_mask = nib.load(wm_path).get_fdata()

# %% [markdown]
# Apply Masks on Bold Dataset

# %%
back_mask_data = back_mask > 0
csf_mask_data = csf_mask > 0
white_mask_data = white_mask > 0.5
mask = np.logical_and(back_mask_data, ~csf_mask_data)
nonzero_mask = np.where(mask)

white_mask_flat = white_mask_data[nonzero_mask]
keep_voxels = ~white_mask_flat

bold_flat = bold_data[nonzero_mask]
masked_bold = bold_flat[keep_voxels]

masked_coords = tuple(ax[keep_voxels] for ax in nonzero_mask)

print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%")
print('bold_data masked shape:', masked_bold.shape)

# %%
num_trials = 90
trial_len = 9
masked_bold = masked_bold.astype(np.float32)
num_voxels, num_timepoints = masked_bold.shape
bold_data_reshape = np.full((num_voxels, num_trials, trial_len), np.nan, dtype=np.float32)

start = 0
for i in range(num_trials):
    end = start + trial_len
    if end > num_timepoints:
        raise ValueError("Masked BOLD data does not contain enough timepoints for all trials")
    bold_data_reshape[:, i, :] = masked_bold[:, start:end]
    start += trial_len
    if start in (270, 560):
        start += 20  # skip discarded timepoints

# %%
glm_dict = np.load(f'TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]

if run == 1:
    beta = beta_run1[keep_voxels]
else:
    beta = beta_run2[keep_voxels]
print("Beta Range:[", np.nanmin(beta), np.nanmax(beta), "], Mean: ", np.nanmean(beta))


# %%
# detect outlier beta after normalization
med = np.nanmedian(beta, keepdims=True)
mad = np.nanmedian(np.abs(beta - med), keepdims=True)
scale = 1.4826 * np.maximum(mad, 1e-9)    
beta_norm = (beta - med) / scale      
thr = np.nanpercentile(np.abs(beta_norm), 99.9)
outlier_mask = np.abs(beta_norm) > thr      
print(f"{np.sum(np.any(outlier_mask, axis=1))/beta.shape[0]*100:.2f}% voxels with at least one outlier beta")


# %%
clean_beta = beta.copy()
voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
valid_voxels = voxel_outlier_fraction <= 0.5
clean_beta[~valid_voxels] = np.nan
clean_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
keeped_mask = ~np.all(np.isnan(clean_beta), axis=1)
clean_beta = clean_beta[keeped_mask]
keeped_indices = np.flatnonzero(keeped_mask)

bold_data_reshape[~valid_voxels, :, :] = np.nan
bold_data_reshape = bold_data_reshape[keeped_mask]

print(f"{(beta.shape[0]-clean_beta.shape[0])/beta.shape[0]*100}% of voxels have more than 50% outlier trials")
print('Clean BOLD reshape shape:', bold_data_reshape.shape)
print(f"Clean beta range: {np.nanmin(clean_beta):.2f}, {np.nanmax(clean_beta):.2f}")

# %%
# one sample t-test against 0
tvals, pvals = ttest_1samp(clean_beta, popmean=0, axis=1, nan_policy='omit')

# FDR correction
tested = np.isfinite(pvals)
alpha=0.05
rej, q, _, _ = multipletests(pvals[tested], alpha=alpha, method='fdr_bh')

n_voxel = clean_beta.shape[0]
qvals  = np.full(n_voxel, np.nan)
reject = np.zeros(n_voxel, dtype=bool)
reject[tested] = rej
qvals[tested]  = q

# reject non-active voxels
clean_active_beta = clean_beta[reject]
clean_active_idx = keeped_indices[reject]
clean_active_bold = bold_data_reshape[reject]
print('Active BOLD shape:', clean_active_bold.shape)
print(f"{clean_active_beta.shape[0]/clean_beta.shape[0]*100:.2f}% of voxels are active at FDR q<{alpha}")
clean_active_beta.shape

# %%
num_trials = beta.shape[-1]
clean_active_volume = np.full(bold_data.shape[:3]+(num_trials,), np.nan)
active_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
clean_active_volume[active_coords[0], active_coords[1], active_coords[2], :] = clean_active_beta
clean_active_volume.shape

# %% [markdown]
# Load the filtered beta values

# %%
beta_valume_clean_2d = np.load(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy')
print(beta_valume_clean_2d.shape)
mask_2d = np.load("mask_all_nan_sub04_ses1_run1.npy")
np.sum(mask_2d)

# %%
active_flat_idx = np.ravel_multi_index(active_coords, clean_active_volume.shape[:3])
active_keep_mask = ~mask_2d[active_flat_idx]
clean_active_bold = clean_active_bold[active_keep_mask]
clean_active_bold.shape

# %%
def calculate_matrices(beta_valume_clean_2d, bold_data, mask_2d, trial_indices=None, trial_len=9, num_components=600, pca_components=None, pca_mean=None):
    print("begin", flush=True)
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
    print("BOLD reshaped before PCA", X.shape, flush=True)

    # ----- apply PCA -----
    print("PCA...", flush=True)
    X_reshap = X.reshape(X.shape[0], -1).astype(np.float32)

    if pca_components is None or pca_mean is None:
        print(1)
        pca = PCA()
        X_pca_full = pca.fit_transform(X_reshap.T).astype(np.float32)
        components = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)
        n_components = min(num_components, components.shape[0])
        components = components[:n_components]
        X_pca = X_pca_full[:, :n_components]
    else:
        print(2)
        components = pca_components.astype(np.float32)
        mean = pca_mean.astype(np.float32)
        n_components = components.shape[0]
        X_centered = X_reshap.T - mean
        X_pca = (X_centered @ components.T).astype(np.float32)

    print(beta_valume_clean_2d.shape)
    print(components.shape)
    beta_reduced = (beta_valume_clean_2d.T - mean) @ components.T
    beta_reduced = beta_reduced.T


    # ----- L_task (same idea as yours) -----
    print("L_task...", flush=True)
    beta_selected = beta_reduced[:, trial_idx]
    counts = np.count_nonzero(np.isfinite(beta_selected), axis=-1)
    sums = np.nansum(np.abs(beta_selected), axis=-1, dtype=np.float64)
    mean_beta = np.zeros(beta_selected.shape[0], dtype=np.float32)
    m = counts > 0
    mean_beta[m] = (sums[m] / counts[m]).astype(np.float32)
    L_task = np.zeros_like(mean_beta, dtype=np.float32)
    v = np.abs(mean_beta) > 0
    L_task[v] = (1.0 / mean_beta[v]).astype(np.float32)

    # ----- L_var_bold: variance of trial differences, as sparse diagonal -----
    print("L_var...", flush=True)
    X_pca = X_pca[:, :n_components].T
    num_trials = len(trial_idx)
    X = X_pca.reshape(X_pca.shape[0], num_trials, trial_len)
    L_var_bold = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(num_trials-1):
        x1 = X[:, i, :]
        x2 = X[:, i+1, :]
        L_var_bold += (x1-x2) @ (x1-x2).T
    L_var_bold /= (num_trials - 1)

    # ----- L_var_beta: variance of trial differences, as sparse diagonal -----
    print("L_var...", flush=True)
    num_trials = len(trial_idx)
    X = beta_reduced
    L_var_beta = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    for i in range(num_trials-1):
        x1 = X[:, i]
        x2 = X[:, i+1]
        L_var_bold += (x1-x2) @ (x1-x2).T
    L_var_beta /= (num_trials - 1)

    selected_BOLD_flat = X.reshape(X.shape[0], -1).astype(np.float32)
    return L_task.astype(np.float32), L_var_beta, L_var_beta, selected_BOLD_flat, components, mean

# %%
def objective_func(w, L_task, L_var_bold, L_var_beta, alpha_var_bold, alpha_var_beta):
    print("Calculating objective...", flush=True)
    def _safe_scale(arr):
        scale = np.max(np.abs(arr))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        print(f"scale: {scale}")
        return arr / scale

    L_task_scaled = _safe_scale(L_task)
    L_var_bold_scaled = _safe_scale(L_var_bold)
    L_var_beta_scaled = _safe_scale(L_var_beta)

    quad = (w.T @ np.diag(L_task_scaled) @ w + alpha_var_bold * (w.T @ L_var_bold_scaled @ w) + 
            alpha_var_beta * (w.T @ L_var_beta_scaled @ w))
    return quad

# %%
def optimize_voxel_weights(L_task, L_var_bold, L_var_beta, alpha_var_bold, alpha_var_beta):
    print("Optimizing voxel weights...", flush=True)
    def _safe_scale(arr):
        scale = np.max(np.abs(arr))
        print(scale)
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        print(f"scale: {scale}")
        return arr / scale

    L_task_scaled = _safe_scale(L_task)
    L_var_bold_scaled = _safe_scale(L_var_bold)
    L_var_beta_scaled = _safe_scale(L_var_beta)

    L_total = (np.diag(L_task_scaled) + alpha_var_bold * L_var_bold_scaled + alpha_var_beta * L_var_beta_scaled)
    L_total = np.nan_to_num(L_total, copy=False).astype(np.float64, copy=False)
    L_total = 0.5 * (L_total + L_total.T)
    L_total += 1e-8 * np.eye(L_total.shape[0])

    w = cp.Variable(L_total.shape[0], nonneg=True)
    constraints = [cp.sum(w) == 1]

    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(L_total)))
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, verbose=True)
    return w.value


# %%
def calculate_weight(param_grid, beta_valume_clean_2d, bold_data, mask_2d, trial_len):
    kf = KFold(n_splits=2, shuffle=True, random_state=42)
    best_score = np.inf
    best_alpha_var_bold = None
    best_alpha_var_beta = None
    num_trials = beta_valume_clean_2d.shape[-1]

    for alpha_var_bold, alpha_var_beta in product(param_grid["alpha_var_bold"], param_grid["alpha_var_beta"]):
        fold_scores = []
        print(f"a_var: {alpha_var_bold}, {alpha_var_beta}", flush=True)
        count = 1

        for train_idx, val_idx in kf.split(np.arange(num_trials)):
            print(f"k-fold num: {count}", flush=True)
            # print(type(mask_2d), flush=True)
            # print(f"11: {beta_valume_clean_2d.shape}")
            L_task_train, L_var_bold_train, L_var_beta_train, _, pca_components, pca_mean = calculate_matrices(
                beta_valume_clean_2d, bold_data, mask_2d, train_idx, trial_len)
            w = optimize_voxel_weights(L_task_train, L_var_bold_train, L_var_beta_train, alpha_var_bold, alpha_var_beta)

            L_task_val, L_var_bold_val, L_var_beta_val, _, _, _ = calculate_matrices(beta_valume_clean_2d, bold_data, 
            mask_2d, val_idx, trial_len, pca_components=pca_components, pca_mean=pca_mean)

            fold_scores.append(objective_func(w, L_task_val, L_var_bold_val, L_var_beta_val, alpha_var_bold, alpha_var_beta))
            print(f"fold_scores: {fold_scores}", flush=True)
            count += 1

        mean_score = np.mean(fold_scores)
        print(mean_score)
        if mean_score < best_score:
            best_score = mean_score
            best_alpha_var_bold = alpha_var_bold
            best_alpha_var_beta = alpha_var_beta

    print("Best alpha_var_bold:", best_alpha_var_bold, "Best alpha_var_beta:", best_alpha_var_beta, "with CV loss:", best_score, flush=True)
    return alpha_var_bold, alpha_var_beta, best_score


# %%
param_grid = {"alpha_var_bold": [0.01],
              "alpha_var_beta": [0.01]}

trial_len = 9

# %%
# Second optimization method
L_task, L_var_bold, L_var_beta, selected_BOLD_flat, pca_components, pca_mean = calculate_matrices(beta_valume_clean_2d, bold_data, mask_2d, None, trial_len)
# param_grid = {"alpha_var_bold": [0.05, 0.1, 1, 1.5, 10],
#               "alpha_var_beta": [0.05, 0.1, 1, 1.5, 10]}
param_grid = {"alpha_var_bold": [0.05],
              "alpha_var_beta": [0.05]}

for alpha_var_bold, alpha_var_beta in product(param_grid["alpha_var_bold"], param_grid["alpha_var_beta"]):
    print(1)
    objective = L_task + alpha_var_bold * L_var_bold + alpha_var_beta * L_var_beta
    print(2)
    objective = (objective+objective.T)/2
    print(3)
    evals, evecs = np.linalg.eigh(objective) 
    print(4)     
    lowest_vec = evecs[:, np.argmin(evals)] 
    print(5)
    lowest_val = evals.min()

