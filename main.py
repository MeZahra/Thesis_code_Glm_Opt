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
from sklearn.decomposition import PCA
import scipy.sparse as sp
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# %%
import psutil, os
p = psutil.Process(os.getpid())
p.cpu_affinity({0,1,2,3,4})   # use only core 0

# %% [markdown]
# Loading Datasets and Masks

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

print(anat_img.shape)
# print(bold_data.shape)
# print(back_mask.shape)
# print(csf_mask.shape)

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

print(f"number of selected voxels after masking: {masked_bold.shape[0]/math.prod(bold_data.shape[:3])*100:.2f}%")
print('bold_data masked shape:', masked_bold.shape)

# %% [markdown]
# Load Beta values, Mask it, Find threshold to determine and delete outlier voxels

# %%
glm_dict = np.load(f'TYPED_FITHRF_GLMDENOISE_RR_sub{sub}.npy', allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]

if run == 1:
    beta = beta_run1[keep_voxels]
else:
    beta = beta_run2[keep_voxels]
print("Beta Range:[", np.nanmin(beta), np.nanmax(beta), "], Mean: ", np.nanmean(beta))


# %% [markdown]
# check how many trials for each voxel have outlier beta (shows my previous method is incorrect)

# %%
# lower_thr, upper_thr = np.nanpercentile(beta, [1, 99])
# print(f'low_thr: {lower_thr:.2f}, high_thr: {upper_thr:.2f}') #low_thr: -4.64, high_thr: 4.60
# beta_extreme_mask = np.logical_or(beta < lower_thr, beta > upper_thr)
# voxels_with_extreme_beta = np.any(beta_extreme_mask, axis=1)

# print(f"percentage of voxels with extreme beta values: {np.sum(voxels_with_extreme_beta)/beta.shape[0]*100:.2f}%")

# mask = np.logical_and(back_mask_data, ~csf_mask_data)
# mask &= ~white_mask_data
# nonzero_mask = np.where(mask)

# clean_beta = beta[~voxels_with_extreme_beta]
# print('clean_beta shape:', clean_beta.shape)

# # This figure is related to my previous calculation (remove voxels even if they have one trial)
# tmp = np.sum(beta_extreme_mask[voxels_with_extreme_beta, :], axis=1)
# plt.hist(tmp)
# plt.xlabel('Num_trials')
# plt.ylabel("voxel count")
# plt.title('Trial-level Outlier Counts in Voxels with Extreme Beta')
# plt.show()

# %% [markdown]
# Normalize beta value, remove voxels with most of trials have bad beta value, interpolate rest of the beta

# %%
# detect outlier beta after normalization
# detect outlier beta after normalization
med = np.nanmedian(beta, keepdims=True)
mad = np.nanmedian(np.abs(beta - med), keepdims=True)
scale = 1.4826 * np.maximum(mad, 1e-9)    
beta_norm = (beta - med) / scale      
thr = np.nanpercentile(np.abs(beta_norm), 99.9)
outlier_mask = np.abs(beta_norm) > thr      
print(f"{np.sum(np.any(outlier_mask, axis=1))/beta.shape[0]*100:.2f}% voxels with at least one outlier beta")

# detect how many trials are bad for selected voxels
beta_clean = beta.copy()
trials = np.arange(beta.shape[1])
voxel_outlier_fraction = np.sum(outlier_mask, axis=1)/outlier_mask.shape[1]*100
valid_voxels = voxel_outlier_fraction <= 50
beta_clean[~valid_voxels] = np.nan 

# save the selected voxels indices for later
keeped_voxels_mask = ~np.all(np.isnan(beta_clean), axis=1)
keeped_voxels_indices = np.flatnonzero(keeped_voxels_mask)

# interpolate beta value for voxels which have less than 50% of trials as bad
orig_outlier = []
intrp_outlier = []
for v in np.flatnonzero(valid_voxels):
    mask = outlier_mask[v]
    if not mask.any():
        continue
    good = ~mask
    orig_outlier.append(beta[v, mask])
    beta_clean[v, mask] = np.interp(trials[mask], trials[good], beta[v, good])
    intrp_outlier.append(beta_clean[v, mask])

# remove voxels that have more than 50% of trials bad beta value
beta_clean1 = beta_clean[~np.all(np.isnan(beta_clean), axis=1)]
print(f"{(beta_clean.shape[0]-beta_clean1.shape[0])/beta_clean.shape[0]*100:.3f}% voxels have more than 50% trials with outlier")
clean_beta = beta_clean1
clean_beta.shape

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
print(f"{clean_active_beta.shape[0]/clean_beta.shape[0]*100:.2f}% of voxels are active at FDR q<{alpha}")
clean_active_beta.shape

# %% [markdown]
# Create 3-D beta dataset to use it for filtering
clean_active_beta_4d = np.full(bold_img.shape[:3]+(clean_beta.shape[1],), np.nan)
active_indices = keeped_voxels_indices[reject]
active_coords = tuple(axis[active_indices] for axis in nonzero_mask)
clean_active_beta_4d[active_coords + (slice(None), )] = clean_active_beta
clean_active_beta_4d.shape



mean_beta_volume = np.nanmean(clean_active_beta_4d, axis=-1)
mean_beta_img = nib.Nifti1Image(mean_beta_volume, bold_img.affine, bold_img.header)
mean_beta_img = resample_to_img(mean_beta_img, anat_img, interpolation='linear')

active_beta_view = plotting.view_img(
    mean_beta_img,
    bg_img=anat_img,
    cmap='seismic',
    symmetric_cmap=False,
    threshold=1e-6,
    colorbar=True,
    title='Mean beta for active voxels'
)
active_beta_view
active_beta_view.save_as_html(file_name=f'active_beta_map_sub{sub}_ses{ses}_run{run}.html')