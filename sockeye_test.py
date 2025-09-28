import nibabel as nib
import numpy as np
from os.path import join
import math
import cvxpy as cp
from sklearn.model_selection import KFold
from itertools import product
import scipy.io as sio
import h5py
import scipy.sparse as sp


ses = 1
sub = '04'
run = 1

base_path = '/scratch/st-mmckeown-1/zkavian/fmri_models/'
anat_img = nib.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')

# data_name = f'fmri_sub{sub}_ses{ses}_run{run}.mat'
# BOLD_path_org = join(base_path, data_name)
# bold_img = sio.loadmat(BOLD_path_org)
# bold_data = bold_img.get_fdata()
bold_data = np.load(join(base_path, 'bold_data.npy'))
print(bold_data.shape, flush=True)
print(1, flush=True)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)
mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
white_mask = nib.load(mask_path)
print(anat_img.shape, bold_data.shape, back_mask.shape, csf_mask.shape, flush=True)

back_mask_data = back_mask.get_fdata() > 0
csf_mask_data = csf_mask.get_fdata() > 0
white_mask_data = white_mask.get_fdata() > 0.9
mask = np.logical_and(back_mask_data, ~csf_mask_data)
mask &= ~white_mask_data
nonzero_mask = np.where(mask)
masked_bold = bold_data[nonzero_mask]
print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%", flush=True)
print('bold_data masked shape:', masked_bold.shape, flush=True)

glm_dict = np.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]
R2_run1, R2_run2 = glm_dict['R2run'][:,:,:,0], glm_dict['R2run'][:,:,:,1]

beta = beta_run1 #beta_glm.shape
R2 = R2_run1

lower_thr, upper_thr = np.nanpercentile(beta, [1, 99])
print(f'low_thr: {lower_thr:.2f}, high_thr: {upper_thr:.2f}', flush=True) #low_thr: -4.64, high_thr: 4.60
beta_extreme_mask = np.logical_or(beta < lower_thr, beta > upper_thr)
voxels_with_extreme_beta = np.any(beta_extreme_mask, axis=1)
print(f"percentage of voxels with extreme beta values: {np.sum(voxels_with_extreme_beta)/beta.shape[0]*100:.2f}%", flush=True)

clean_beta = beta[~voxels_with_extreme_beta]
clean_R2 = R2[~voxels_with_extreme_beta]
print('clean_beta shape:', clean_beta.shape)

######################################################
# # one sample t-test against 0
# tvals, pvals = ttest_1samp(clean_beta, popmean=0, axis=1, nan_policy='omit')

# # FDR correction
# tested = np.isfinite(pvals)
# alpha=0.05
# rej, q, _, _ = multipletests(pvals[tested], alpha=alpha, method='fdr_bh')

# n_voxel = clean_beta.shape[0]
# qvals  = np.full(n_voxel, np.nan)
# reject = np.zeros(n_voxel, dtype=bool)
# reject[tested] = rej
# qvals[tested]  = q

# # reject non-active voxels
# clean_active_beta = clean_beta[reject]
# clean_active_R2 = clean_R2[reject]
# print(f"{clean_active_beta.shape[0]/clean_beta.shape[0]*100:.2f}% of voxels are active at FDR q<{alpha}")

# ###########################################################
# # transfer back beta value on the volume
# clean_mask = ~np.asarray(voxels_with_extreme_beta, dtype=bool)
# clean_indices = np.nonzero(clean_mask)[0]
# active_indices = clean_indices[np.asarray(reject, dtype=bool)]

# spatial_shape = bold_img.shape[:3]
# n_trials = clean_active_beta.shape[1]
# beta_volume = np.full(spatial_shape + (n_trials,), np.nan, dtype=np.float32)

# coords = tuple(axis[active_indices] for axis in nonzero_mask)
# beta_volume[coords[0], coords[1], coords[2], :] = clean_active_beta.astype(np.float32)
# ############################################################
# def hampel_filter_image(image, window_size, threshold_factor):
#     if window_size % 2 == 0:
#         raise ValueError("window_size must be odd")

#     filtered = image.astype(float).copy()
#     footprint = np.ones((window_size,) * 3, dtype=bool)

#     for t in range(image.shape[3]):
#         vol = image[..., t]
#         med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan)
#         mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan)
#         counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0)

#         scaled_mad = 1.4826 * mad  # Gaussian-consistent scaling
#         valid = np.isfinite(vol)
#         insufficient = counts < 3

#         filtered[..., t][insufficient] = np.nan

#         enough_data = (counts >= 3) & valid
#         outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)
#         filtered[..., t][outliers] = med[outliers]
#     return filtered
# beta_volume_filter = hampel_filter_image(beta_volume, window_size=5, threshold_factor=3)
# np.save(f'beta_volume_filter.npy', beta_volume_filter)
#########################################################
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
    print(1, flush=True)
    m = np.zeros((X.shape[0], X.shape[0]), dtype=np.float32)
    print(2, flush=True)
    for i in range(num_trials-1):
        print("trial diff num: ", i, flush=True)
        x1 = X[:, i, :]
        x2 = X[:, i+1, :]
        m += (x1-x2) @ (x1-x2).T
    L_var /= (num_trials - 1)
    print(3, flush=True)

    # diff_mat = np.diff(X, axis=1)
    # diff_mat_flat = diff_mat.reshape(diff_mat.shape[0], -1)
    # L_var = np.cov(diff_mat_flat, bias=False, dtype=np.float32)
    # L_var = (L_var + L_var.T) / 2 + 1e-6 * np.eye(L_var.shape[0], dtype=np.float32)

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

##
# def calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, trial_indices=None, trial_len=9):
#     num_trials = beta_valume_clean_2d.shape[-1]
#     trial_idx = np.arange(num_trials) if trial_indices is None else np.unique(np.asarray(trial_indices, int).ravel())

#     # ----- L_task (same idea as yours) -----
#     beta_selected = beta_valume_clean_2d[:, trial_idx]
#     counts = np.count_nonzero(np.isfinite(beta_selected), axis=-1)
#     sums = np.nansum(beta_selected, axis=-1, dtype=np.float64)
#     mean_beta = np.zeros(beta_selected.shape[0], dtype=np.float32)
#     m = counts > 0
#     mean_beta[m] = (sums[m] / counts[m]).astype(np.float32)
#     L_task = np.zeros_like(mean_beta, dtype=np.float32)
#     v = np.abs(mean_beta) > 0
#     L_task[v] = (1.0 / np.abs(mean_beta[v])).astype(np.float32)

#     # ----- reshape BOLD into trials -----
#     bold_data_reshape = bold_data.reshape(-1, bold_data.shape[-1])
#     bold_data_selected = bold_data_reshape[~mask_2d]         # keep voxels of interest
#     bold_data_selected_reshape = np.zeros((bold_data_selected.shape[0], num_trials, trial_len), dtype=np.float32)
#     start = 0
#     for i in range(num_trials):
#         end = start + trial_len
#         if end > bold_data_selected.shape[1]:
#             raise ValueError("BOLD data does not contain enough timepoints for all trials")
#         bold_data_selected_reshape[:, i, :] = bold_data_selected[:, start:end]
#         start += trial_len
#         if start == 270 or start == 560:   # your skips
#             start += 20
#     X = bold_data_selected_reshape[:, trial_idx, :]          # [Nvox, Ntrials, T]

#     # ----- L_var: variance of trial differences, as sparse diagonal -----
#     diff_mat = np.diff(X, axis=1)                            # [Nvox, Ntrials-1, T]
#     diff_mat = diff_mat.reshape(diff_mat.shape[0], -1)       # [Nvox, (Ntrials-1)*T]
#     diff_mat = diff_mat[:, np.all(np.isfinite(diff_mat), axis=0)]
#     if diff_mat.shape[1] < 2:
#         raise ValueError("Not enough finite samples to compute L_var")
#     var_vec = np.nanvar(diff_mat, axis=1, ddof=1).astype(np.float32)  # per-voxel variance
#     L_var = sp.diags(var_vec, format='csc')                           # sparse diagonal (PSD)

#     # ----- L_smooth: sparse 6-neighbor Laplacian on the voxel grid -----
#     mask3d = (~mask_2d).reshape(bold_data.shape[:3])
#     idx = -np.ones(mask3d.shape, dtype=np.int64)
#     idx[mask3d] = np.arange(mask3d.sum())
#     rows, cols = [], []
#     for ax in range(3):
#         s1 = [slice(None)]*3; s2 = [slice(None)]*3
#         s1[ax] = slice(1, None); s2[ax] = slice(0, -1)
#         a, b = idx[tuple(s1)], idx[tuple(s2)]
#         m = (a != -1) & (b != -1)
#         i, j = a[m].ravel(), b[m].ravel()
#         rows.append(np.concatenate([i, j])); cols.append(np.concatenate([j, i]))
#     if rows:
#         rows, cols = np.concatenate(rows), np.concatenate(cols)
#         A = sp.coo_matrix((np.ones(rows.size, np.float32), (rows, cols)), shape=(idx.max()+1, idx.max()+1)).tocsr()
#         d = np.asarray(A.sum(axis=1)).ravel()
#         L_smooth = (sp.diags(d, 0) - A).astype(np.float32).tocsc()
#     else:
#         L_smooth = sp.csc_matrix((idx.max()+1, idx.max()+1), dtype=np.float32)

#     selected_BOLD_flat = X.reshape(X.shape[0], -1).astype(np.float32)
#     return L_task.astype(np.float32), L_var, L_smooth, selected_BOLD_flat

##
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
        print(f"a_var: {a_var}, a_smooth: {a_smooth}", flush=True)
        count = 1

        for train_idx, val_idx in kf.split(np.arange(num_trials)):
            print(f"k-fold num: {count}", flush=True)
            L_task_train, L_var_train, L_smooth_train, _ = calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, train_idx, trial_len)
            w = optimize_voxel_weights(L_task_train, L_var_train, L_smooth_train, alpha_var=a_var, alpha_smooth=a_smooth)

            L_task_val, L_var_val, L_smooth_val, _ = calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, val_idx, trial_len)

            fold_scores.append(objective_func(w, L_task_val, L_var_val, L_smooth_val, a_var, a_smooth))
            print(f"fold_scores: {fold_scores}", flush=True)
            count += 1

        mean_score = np.mean(fold_scores)
        print(mean_score, flush=True)
        if mean_score < best_score:
            best_score = mean_score
            best_params = (a_var, a_smooth)

    print("Best parameters:", best_params, "with CV loss:", best_score, flush=True)
    return best_params, best_score

####################
beta_volume_filter = np.load("beta_volume_filter.npy")
# bold_data = bold_img.get_fdata()
print(f"beta_volume_filter shape : {beta_volume_filter.shape}", flush=True)
print(2, flush=True)
spatial_shape = beta_volume_filter.shape[:-1]
voxels_with_any_nan = np.zeros(spatial_shape, dtype=bool)
voxels_with_all_nan = np.ones(spatial_shape, dtype=bool)

# Sweep the time dimension once
for t in range(beta_volume_filter.shape[-1]):
    frame_nan = np.isnan(beta_volume_filter[..., t])
    voxels_with_any_nan |= frame_nan
    voxels_with_all_nan &= frame_nan

print(np.sum(voxels_with_any_nan), np.sum(voxels_with_all_nan), flush=True)

n_trial = beta_volume_filter.shape[-1]
beta_volume_filter_2d = beta_volume_filter.reshape(-1, n_trial)
print(beta_volume_filter_2d.shape, flush=True)
mask_2d = voxels_with_all_nan.reshape(-1)
beta_valume_clean_2d = beta_volume_filter_2d[~mask_2d]
print(beta_valume_clean_2d.shape, flush=True)


param_grid = {
    "alpha_var":   [0.01, 0.5, 1.0],
    "alpha_smooth":[0.01, 0.5, 1.0]}

trial_len = 9
best_params, best_score = calculate_weight(param_grid, beta_valume_clean_2d, bold_data, anat_img, mask_2d, trial_len)
print(3, flush=True)
L_task, L_var, L_smooth, selected_BOLD_data = calculate_matrices(beta_valume_clean_2d, bold_data, anat_img, mask_2d, None, trial_len)
print(4, flush=True)
weights = optimize_voxel_weights(L_task, L_var, L_smooth, alpha_var=best_params[0], alpha_smooth=best_params[1])
print(5, flush=True)
y = selected_BOLD_data.T @ weights

np.save('weights.npy', weights)
np.save('y.npy', y)
print("Finished!", flush=True)