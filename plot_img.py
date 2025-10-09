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
from scipy.stats import zscore

import numpy as np
import nibabel as nib
from os.path import join

ses = 1
sub = '04'
run = 1

base_path = '/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'

# Load images without forcing full data into memory when not needed
anat_path = f'{base_path}/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz'
anat_img = nib.load(anat_path)  # shape access doesn't load full data
print(anat_img.shape)

data_name = f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'
BOLD_path_org = join(base_path, f'sub-pd0{sub}', f'ses-{ses}', 'func', data_name)
bold_img = nib.load(BOLD_path_org)

# Load directly as float32 to avoid float64->float16 double allocation
# (float16 often causes instability downstream; float32 is a safer, still-light choice)
# bold_data = bold_img.get_fdata(dtype=np.float32, caching='unchanged')
bold_data = bold_img.get_fdata()

# Brain mask (binary): keep as uint8 (or bool) to reduce memory
mask_base = f'{base_path}/sub-pd0{sub}/ses-{ses}/anat'

brain_mask_path = f'{mask_base}/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
brain_mask_img = nib.load(brain_mask_path)
# Load as uint8; if mask is probabilistic 0/1, this will keep it tiny.
# back_mask = brain_mask_img.get_fdata(dtype=np.uint8, caching='unchanged')
back_mask = brain_mask_img.get_fdata()

# CSF and WM PVE maps are fractional -> keep as float32
csf_path = f'{mask_base}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
# csf_mask = nib.load(csf_path).get_fdata(dtype=np.float32, caching='unchanged')
csf_mask = nib.load(csf_path).get_fdata()

wm_path = f'{mask_base}/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
# white_mask = nib.load(wm_path).get_fdata(dtype=np.float32, caching='unchanged')
white_mask = nib.load(wm_path).get_fdata()

# print(bold_img.shape, brain_mask_img.shape)

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