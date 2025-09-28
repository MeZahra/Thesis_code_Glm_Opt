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

print(anat_img.shape)
print(bold_data.shape)
print(back_mask.shape)
print(csf_mask.shape)

# %%
back_mask_data = back_mask.get_fdata() > 0
csf_mask_data = csf_mask.get_fdata() > 0
mask = np.logical_and(back_mask_data, ~csf_mask_data)
nonzero_mask = np.where(mask)
masked_bold = bold_data[nonzero_mask]

print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%")
print('bold_data masked shape:', masked_bold.shape)

# %%
glm_dict = np.load(f'/home/zkavian/thesis_code_git/GLMOutputs-sub{sub}-ses{ses}/TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]
R2_run1, R2_run2 = glm_dict['R2run'][:,:,:,0], glm_dict['R2run'][:,:,:,1]

beta = beta_run1 #beta_glm.shape
R2 = R2_run1

lower_thr, upper_thr = np.nanpercentile(beta, [1, 99])
print(f'low_thr: {lower_thr:.2f}, high_thr: {upper_thr:.2f}') #low_thr: -4.64, high_thr: 4.60
beta_extreme_mask = np.logical_or(beta < lower_thr, beta > upper_thr)
voxels_with_extreme_beta = np.any(beta_extreme_mask, axis=1)

print(f"percentage of voxels with extreme beta values: {np.sum(voxels_with_extreme_beta)/beta.shape[0]*100:.2f}%")


# %%
# from plot_func import cfs_brain_mask_plot, mean_beta_outlier_voxels, csf_mask_with_outlier, check_beta_range_and_outliers, check_avg_beta_range

# overlay_view = cfs_brain_mask_plot(nonzero_mask, mask, bold_img, anat_img)
# overlay_view
# mean_beta_outlier_voxels(beta, bold_data, bold_img, anat_img, nonzero_mask, sub, ses, run)
# csf_mask_with_outlier(csf_mask_data, voxels_with_extreme_beta, bold_data, bold_img, anat_img, nonzero_mask, sub, ses, run)
# check_beta_range_and_outliers(beta, bold_data, bold_img, anat_img, nonzero_mask)
# check_avg_beta_range(beta, bold_data, bold_img, anat_img, nonzero_mask)

# %%
# after removing voxels with extreme beta values
clean_beta = beta[~voxels_with_extreme_beta]
clean_R2 = R2[~voxels_with_extreme_beta]
print('clean_beta shape:', clean_beta.shape)

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

# rehject non-active voxels
clean_active_beta = clean_beta[reject]
clean_active_R2 = clean_R2[reject]
print(f"{clean_active_beta.shape[0]/clean_beta.shape[0]*100:.2f}% of voxels are active at FDR q<{alpha}")

# # plot hist and beta values of ative voxels
# from plot_func import active_voxel_plot
# active_voxel_plot(clean_active_beta, voxels_with_extreme_beta, reject, bold_img, anat_img, nonzero_mask, sub, ses, run)

# %%
# transfer back beta value on the volume
clean_mask = ~np.asarray(voxels_with_extreme_beta, dtype=bool)
clean_indices = np.nonzero(clean_mask)[0]
active_indices = clean_indices[np.asarray(reject, dtype=bool)]

spatial_shape = bold_img.shape[:3]
n_trials = clean_active_beta.shape[1]
beta_volume = np.full(spatial_shape + (n_trials,), np.nan, dtype=np.float32)

coords = tuple(axis[active_indices] for axis in nonzero_mask)
beta_volume[coords[0], coords[1], coords[2], :] = clean_active_beta.astype(np.float32)

# %%
def hampel_filter_image(image, window_size, threshold_factor):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    filtered = image.astype(float).copy()
    footprint = np.ones((window_size,) * 3, dtype=bool)

    for t in range(image.shape[3]):
        vol = image[..., t]
        med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan)
        mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan)
        counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0)

        scaled_mad = 1.4826 * mad  # Gaussian-consistent scaling
        valid = np.isfinite(vol)
        insufficient = counts < 3

        filtered[..., t][insufficient] = np.nan

        enough_data = (counts >= 3) & valid
        outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)
        filtered[..., t][outliers] = med[outliers]
    return filtered


# %%
beta_volume_filter = hampel_filter_image(beta_volume, window_size=7, threshold_factor=3)
np.save(f'beta_volume_filter_sub{sub}_ses{ses}_run{run}_after_hampel.npy', beta_volume_filter)

# %%
# print(beta_volume_filter.shape)

# stat_img = nib.Nifti1Image(np.nanmean(beta_volume_filter, axis=-1), bold_img.affine, bold_img.header)
# stat_img = resample_to_img(stat_img, anat_img, interpolation='linear')

# view = plotting.view_img(stat_img, bg_img=anat_img, cmap='jet', colorbar=True, title='Mean beta for active voxels')
# view.save_as_html(f'mean_beta_sub{sub}_ses{ses}_run{run}_after_hampel.html')

# %%
# plt.figure()
# plt.hist(beta_volume_filter.ravel(), bins=30)
# plt.show()

# %%
# beta_volume_filter_reshape = beta_volume_filter.reshape(-1, beta_volume_filter.shape[-1])
# mean_beta_filtered = np.nanmean(beta_volume_filter_reshape, axis=-1)
# print(f"Beta range: {np.nanmin(mean_beta_filtered):.2f} to {np.nanmax(mean_beta_filtered):.2f}")

# L_task = np.divide(1., np.abs(mean_beta_filtered), where=mean_beta_filtered!=0)
# print(f"L_task range: {np.nanmin(L_task):.2f} to {np.nanmax(L_task):.2f}")

# # plt.figure()
# # plt.hist(L_task)
# # plt.show()

# # np.where((mean_beta_filtered <0.0001) & (mean_beta_filtered> 0))
# # np.where(abs(mean_beta_filtered) >0.1)

# %%
# finite_counts = np.sum(np.isfinite(beta_volume), axis=-1)
# tmp = finite_counts[finite_counts>80]

# plt.figure()
# plt.hist(finite_counts.ravel(), bins=30)
# plt.show()

# plt.figure()
# plt.hist(tmp)
# plt.show()

# %%
# finite_counts = np.sum(np.isfinite(beta_volume), axis=-1)
# min_finite_betas = 0
# valid_mask = finite_counts > min_finite_betas
# coords = np.argwhere(valid_mask)
# bold_data_selected = bold_data[valid_mask]

# bold_data_selected = np.reshape(bold_data_selected, (-1, bold_data.shape[-1]))
# num_trials= 90
# trial_len = 9
# bold_data_selected_reshape = np.zeros((bold_data_selected.shape[0], num_trials, trial_len))

# start = 0
# for i in range(num_trials):
#     bold_data_selected_reshape[:, i, :] = bold_data_selected[:, start:start+trial_len]
#     start += trial_len
#     if start == 270 or start == 560:
#         start += 20

# diff_mat = np.diff(bold_data_selected_reshape, axis=1)
# diff_mat_flat = diff_mat.reshape(diff_mat.shape[0], -1)
# L_var = np.cov(diff_mat_flat, bias=False, dtype=np.float16)
# L_var = (L_var + L_var.T) / 2 + 1e-6 * np.eye(L_var.shape[0])
# print(L_var.shape) # I need to reduce number of voxels at the end to 100,000 - 200,000 to make it computationally feasible

# %%
# voxel_indices = np.column_stack(coords)
# selected_world_coords = nib.affines.apply_affine(anat_img.affine, voxel_indices)
# D = cdist(selected_world_coords, selected_world_coords)
# nonzero = D[D > 0]
# sigma = np.median(nonzero) if nonzero.size else 1.0
# W = np.exp(-D**2 / (2 * sigma**2))
# np.fill_diagonal(W, 0.0)
# L_smooth = csgraph.laplacian(W, normed=False)

# %%
# def calculate_matrices():


#     return L_task, L_var, L_smooth, bold_data_selected_reshape

# %%
# def objective_func(w, L_task, L_var, L_smooth, alpha_var, alpha_smooth):
#     quad = (w.T @ np.diag(L_task) @ w + alpha_var * (w.T @ L_var @ w) + alpha_smooth * (w.T @ L_smooth @ w))
#     return quad

# %%
# def optimize_voxel_weights(L_task, L_var, L_smooth, alpha_var, alpha_smooth):
#     L_total = np.diag(L_task) + alpha_var * L_var + alpha_smooth * L_smooth
#     n = L_total.shape[0]
#     L_total = 0.5*(L_total + L_total.T) + 1e-8*np.eye(n)
#     w = cp.Variable(n, nonneg=True)
#     constraints = [cp.sum(w) == 1]
    
#     # objective = cp.Minimize(cp.quad_form(w, L_total) + alpha_sparse * cp.norm1(w))
#     objective = cp.Minimize(cp.quad_form(w, L_total))
#     problem = cp.Problem(objective, constraints)
#     problem.solve(solver=cp.OSQP, verbose=True)
#     return w.value

# %%
# def calculate_weight(param_grid, betasmd, active_low_var_voxels, anat_img, affine, BOLD_path_org, trial_len):
#     kf = KFold(n_splits=2, shuffle=True, random_state=0)
#     best_score = np.inf
#     best_params = None
#     num_trials = betasmd.shape[-1]

#     for a_var, a_smooth in product(*param_grid.values()):
#         fold_scores = []
#         print(f"a_var: {a_var}, a_smooth: {a_smooth}")
#         count = 1

#         for train_idx, val_idx in kf.split(np.arange(num_trials)):
#             print(f"k-fold num: {count}")
#             L_task_train, L_var_train, L_smooth_train, _ = calculate_matrices(betasmd, active_low_var_voxels, anat_img, affine, BOLD_path_org, train_idx, trial_len)
#             w = optimize_voxel_weights(L_task_train, L_var_train, L_smooth_train, alpha_var=a_var, alpha_smooth=a_smooth)

#             L_task_val, L_var_val, L_smooth_val, _ = calculate_matrices(betasmd, active_low_var_voxels, anat_img, affine, BOLD_path_org, val_idx, trial_len)

#             fold_scores.append(objective_func(w, L_task_val, L_var_val, L_smooth_val, a_var, a_smooth))
#             print(f"fold_scores: {fold_scores}")
#             count += 1

#         mean_score = np.mean(fold_scores)
#         print(mean_score)
#         if mean_score < best_score:
#             best_score = mean_score
#             best_params = (a_var, a_smooth)

#     print("Best parameters:", best_params, "with CV loss:", best_score)
#     return best_params, best_score

# %%
# def select_opt_weight(selected_BOLD_data, weights, selected_voxels, affine):
#     y = selected_BOLD_data.T @ weights
#     p95 = np.percentile(weights, 95)

#     weight_volume = np.zeros(selected_voxels.shape, dtype=np.float32)
#     weight_volume[selected_voxels] = weights

#     mask = np.zeros(selected_voxels.shape, dtype=bool)
#     selected_weights = (weights >= p95)
#     mask[selected_voxels] = selected_weights
#     weight_volume[~mask] = 0

#     masked_weights = np.where(weight_volume == 0, np.nan, weight_volume)
#     weight_img = nib.Nifti1Image(masked_weights, affine=affine)
    
#     return weight_img, masked_weights, y

# %%



