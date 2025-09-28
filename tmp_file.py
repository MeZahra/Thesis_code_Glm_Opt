# %%
import nibabel as nib
import numpy as np
from os.path import join
import math
from scipy.spatial.distance import cdist
from scipy.sparse import csgraph
import cvxpy as cp
from sklearn.model_selection import KFold
from itertools import product
import h5py


# %%
ses = 1
sub = '04'
run = 1

base_path = '/scratch/st-mmckeown-1/zkavian/fmri_models/'
anat_img = nib.load(join(base_path, f"sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz"))
print(1, flush=True)
data_name = f'fmri_sub{sub}_ses{ses}_run{run}.mat'
BOLD_path_org = join(base_path, data_name)
BOLD_path_org = join(data_name)
print(2, flush=True)

with h5py.File(BOLD_path_org, 'r') as mat_file:
    bold_data = mat_file['data'][()]

print(3, flush=True)
# Matlab v7.3 stores arrays with time as the leading axis; align with beta volumes
bold_data = np.asarray(bold_data)
bold_data = np.transpose(bold_data, (3, 2, 1, 0))

print(4, flush=True)
np.save('bold_data.npy', bold_data)