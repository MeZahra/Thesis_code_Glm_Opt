import numpy as np
import nibabel as nib
from nilearn import plotting

beta_path = "/Data/zahra/results_beta_preprocessed/group_concat/cleaned_beta_volume_group.npy"
mask_path = "/Data/zahra/results_beta_preprocessed/group_concat/common_mask_group.npy"
anat_path = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"

beta = np.load(beta_path, mmap_mode="r")          # (91781, 5220)
mask = np.load(mask_path).astype(bool)            # (91, 109, 91)
anat_img = nib.load(anat_path)

# Mean over trials (5220) -> one value per voxel
mean_beta = np.nanmean(beta, axis=1).astype(np.float32)  # (91781,)
mean_beta = np.nan_to_num(mean_beta, nan=0.0, posinf=0.0, neginf=0.0)

vol = np.zeros(mask.shape, dtype=np.float32)
vol[mask] = mean_beta

mean_beta_img = nib.Nifti1Image(vol, anat_img.affine, anat_img.header)

# optional alias so you can call nib.plotting_view(...)
nib.plotting_view = plotting.view_img

thr = float(np.percentile(np.abs(mean_beta), 95))
view = nib.plotting_view(
    mean_beta_img,
    bg_img=anat_img,
    cmap="cold_hot",
    symmetric_cmap=True,
    threshold=thr,
    colorbar=True,
    title="Mean beta over 5220 trials",
)

view
