# %%
import nibabel as nib
import numpy as np
from os.path import join
import math
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.model_selection import KFold
from itertools import product
import scipy.io as sio


# %%
ses = 1
sub = '04'
run = 1

base_path = '/scratch/st-mmckeown-1/zkavian/fmri_models/'
anat_img = nib.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')

bold_data = np.load(join(base_path, 'bold_data.npy'))
print(bold_data.shape, flush=True)
print(1, flush=True)

mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)
back_mask = back_mask.get_fdata().astype(np.float16)
mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
csf_mask = csf_mask.get_fdata().astype(np.float16)
mask_path = f'/scratch/st-mmckeown-1/zkavian/fmri_models/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
white_mask = nib.load(mask_path)
white_mask = white_mask.get_fdata().astype(np.float16)
print(anat_img.shape, bold_data.shape, back_mask.shape, csf_mask.shape, flush=True)

back_mask_data = back_mask > 0
csf_mask_data = csf_mask > 0
white_mask_data = white_mask > 0.9
mask = np.logical_and(back_mask_data, ~csf_mask_data)
mask &= ~white_mask_data
nonzero_mask = np.where(mask)
masked_bold = bold_data[nonzero_mask]
print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%", flush=True)
print('bold_data masked shape:', masked_bold.shape, flush=True)

glm_dict = np.load(f'/scratch/st-mmckeown-1/zkavian/fmri_models/TYPED_FITHRF_GLMDENOISE_RR.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1 = beta_glm[:,0,0,:90]
beta = beta_run1 #beta_glm.shape
beta = beta.astype(np.float16)
del beta_run1

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
beta_volume_filter_2d = beta_volume_filter_2d.astype(np.float16)
print(beta_volume_filter_2d.shape, flush=True)

mask_2d = voxels_with_all_nan.reshape(-1)
beta_valume_clean_2d = beta_volume_filter_2d[~mask_2d]
beta_valume_clean_2d = beta_valume_clean_2d.astype(np.float16)
print(beta_valume_clean_2d.shape, flush=True)

bold_data_reshape = bold_data.reshape(-1, bold_data.shape[-1])
bold_data_reshape = bold_data_reshape.astype(np.float16)
bold_data_selected = bold_data_reshape[~mask_2d]         # keep voxels of interest
bold_data_selected = bold_data_selected.astype(np.float16)
trial_len = 9
num_trials = bold_data.shape[-1]
trial_idx = np.arange(num_trials)
bold_data_selected_reshape = np.zeros((bold_data_selected.shape[0], num_trials, trial_len), dtype=np.float16)

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
np.save("X.npy", X)