# %%
import nibabel as nib
import numpy as np
from os.path import join
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import scipy.sparse as sp
import scipy.ndimage as ndimage
from nilearn import plotting
from pathlib import Path
# from empca.empca import empca

# %%
ses = 1
sub = '09'
run = 1
num_trials = 90
trial_len = 9

base_path = f'/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/Rev_pipeline/derivatives'
glmsingle_root = Path('/mnt/TeamShare/Data_Masterfile/Zahra-Thesis-Data/Msc_Proj_Nov')


def _find_latest_glmsingle_typed_file(root: Path, subject: str, session: int) -> Path:
    ses_int = int(session)
    candidates = [
        root / f'GLMOutputs-sub{subject}-ses{ses_int}2',
        root / f'GLMOutputs-sub{subject}-ses{ses_int}',
        root / f'GLMOutputs-sub{subject}-ses{ses_int:02d}2',
        root / f'GLMOutputs-sub{subject}-ses{ses_int:02d}',
    ]
    patterns = [
        f'TYPED_FITHRF_GLMDENOISE_RR_sub{subject}_ses{ses_int}.npy',
        f'TYPED_FITHRF_GLMDENOISE_RR_sub{subject}_ses{ses_int:02d}.npy',
        f'TYPED_FITHRF_GLMDENOISE_RR_sub{subject}_ses{ses_int}*.npy',
        f'TYPED_FITHRF_GLMDENOISE_RR_sub{subject}_ses{ses_int:02d}*.npy',
        f'TYPED_FITHRF_GLMDENOISE_RR_sub{subject}*.npy',
        'TYPED_FITHRF_GLMDENOISE_RR*.npy',
    ]

    matches: list[Path] = []
    for directory in candidates:
        if not directory.exists():
            continue
        for pattern in patterns:
            matches.extend(directory.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f'Could not find GLMsingle TYPE-D output under {root} for sub={subject}, ses={session}.'
        )

    return max(matches, key=lambda p: p.stat().st_mtime)


def _infer_csf_exclusion_threshold(brain_mask: np.ndarray, csf_pve: np.ndarray, target_voxels: int) -> float:
    candidates = [0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    counts = {thr: int(np.count_nonzero(brain_mask & ~(csf_pve > thr))) for thr in candidates}

    for thr, count in counts.items():
        if count == target_voxels:
            return float(thr)

    best_thr = min(candidates, key=lambda thr: abs(counts[thr] - target_voxels))
    print(
        f'Warning: could not exactly match GLMsingle voxel count ({target_voxels}). '
        f'Using csf_threshold={best_thr} with mask voxels={counts[best_thr]}.',
        flush=True,
    )
    return float(best_thr)


def _precentral_cut_coords(derivatives_root: str, subject: str, session: int):
    roi_path = (
        Path(derivatives_root)
        / f'sub-pd0{subject}'
        / f'ses-{session}'
        / 'func'
        / 'atlas_files'
        / 'rois'
        / 'precentral.nii.gz'
    )
    if not roi_path.exists():
        return None

    roi_img = nib.load(str(roi_path))
    roi_data = roi_img.get_fdata() > 0
    if not np.any(roi_data):
        return None

    coords_vox = np.array(np.where(roi_data)).T
    centroid_vox = coords_vox.mean(axis=0)
    centroid_world = (roi_img.affine @ np.r_[centroid_vox, 1])[:3]
    return tuple(float(x) for x in centroid_world)


def _save_beta_overlay(mean_abs_beta: np.ndarray, anat_img: nib.Nifti1Image, out_html: str, cut_coords=None):
    finite = mean_abs_beta[np.isfinite(mean_abs_beta)]
    if finite.size == 0:
        raise ValueError('No finite beta values to visualize.')

    thr = float(np.percentile(finite, 90))
    vmax = float(np.percentile(finite, 99))
    img = nib.Nifti1Image(mean_abs_beta.astype(np.float32), anat_img.affine, anat_img.header)
    view = plotting.view_img(
        img,
        bg_img=anat_img,
        cmap='inferno',
        symmetric_cmap=False,
        threshold=thr,
        vmax=vmax,
        colorbar=True,
        title=f'Mean |beta| (thr p90={thr:.2f}, vmax p99={vmax:.2f})',
        cut_coords=cut_coords,
    )
    view.save_as_html(out_html)
    print(f'Saved overlay: {out_html}', flush=True)


output_dir = Path(f'sub{sub}')
output_dir.mkdir(parents=True, exist_ok=True)
cached_beta_path = output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy'

anat_img = nib.load(f'{base_path}/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz')
cut_coords = _precentral_cut_coords(base_path, sub, ses)

if cached_beta_path.exists():
    beta_volume_filter = np.load(cached_beta_path)
    with np.errstate(invalid='ignore'):
        mean_clean_active = np.nanmean(np.abs(beta_volume_filter), axis=-1).astype(np.float32)
    _save_beta_overlay(
        mean_clean_active,
        anat_img=anat_img,
        out_html=str(output_dir / f'clean_active_beta_overlay_sub{sub}_ses{ses}_run{run}.html'),
        cut_coords=cut_coords,
    )
    raise SystemExit(0)

bold_data = nib.load(
    join(
        base_path,
        f'sub-pd0{sub}',
        f'ses-{ses}',
        'func',
        f'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz',
    )
).get_fdata()

mask_path =  f'{base_path}/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
back_mask = nib.load(mask_path)
back_mask = back_mask.get_fdata()
back_mask = back_mask.astype(np.float16)

mask_path =  f'{base_path}/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
csf_mask = nib.load(mask_path)
csf_mask = csf_mask.get_fdata()
csf_mask = csf_mask.astype(np.float16)

mask_path =  f'{base_path}/sub-pd0{sub}/ses-{ses}/anat/sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
gray_mask = nib.load(mask_path)
gray_mask = gray_mask.get_fdata()
gray_mask = gray_mask.astype(np.float16)

print(1, flush=True)

# %%
back_mask_data = back_mask > 0
gray_mask_data = gray_mask > 0.5

typed_path = _find_latest_glmsingle_typed_file(glmsingle_root, sub, ses)
print(f'Loading GLMsingle output: {typed_path}', flush=True)
glm_dict = np.load(str(typed_path), allow_pickle=True).item()
beta_glm = glm_dict['betasmd']
target_voxels = int(beta_glm.shape[0])
csf_thr = _infer_csf_exclusion_threshold(back_mask_data, csf_mask, target_voxels)
print(f'Using csf_threshold={csf_thr} (target voxels={target_voxels})', flush=True)

mask = np.logical_and(back_mask_data, ~(csf_mask > csf_thr))
nonzero_mask = np.where(mask)

keep_voxels = gray_mask_data[nonzero_mask]

bold_flat = bold_data[nonzero_mask]
masked_bold = bold_flat[keep_voxels]
masked_coords = tuple(ax[keep_voxels] for ax in nonzero_mask)

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

print(2, flush=True)
# %%
beta_run1, beta_run2 = beta_glm[:,0,0,:90], beta_glm[:,0,0,90:]

if run == 1:
    beta = beta_run1[keep_voxels]
else:
    beta = beta_run2[keep_voxels]

nan_voxels = np.isnan(beta).all(axis=1)
if np.any(nan_voxels):
    beta = beta[~nan_voxels]
    bold_data_reshape = bold_data_reshape[~nan_voxels]
    masked_coords = tuple(coord[~nan_voxels] for coord in masked_coords)

print(3, flush=True)

# %% [markdown]
med = np.nanmedian(beta, keepdims=True)
mad = np.nanmedian(np.abs(beta - med), keepdims=True)
scale = 1.4826 * np.maximum(mad, 1e-9)    
beta_norm = (beta - med) / scale      
thr = np.nanpercentile(np.abs(beta_norm), 99.9)
outlier_mask = np.abs(beta_norm) > thr  


clean_beta = beta.copy()
voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
valid_voxels = voxel_outlier_fraction <= 0.5
clean_beta[~valid_voxels] = np.nan
clean_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
keeped_mask = ~np.all(np.isnan(clean_beta), axis=1)
clean_beta = clean_beta[keeped_mask]
keeped_indices = np.flatnonzero(keeped_mask)

bold_data_reshape[~valid_voxels, :, :] = np.nan
trial_outliers = np.logical_and(outlier_mask, valid_voxels[:, None])
bold_data_reshape = np.where(trial_outliers[:, :, None], np.nan, bold_data_reshape)
bold_data_reshape = bold_data_reshape[keeped_mask]

print(4, flush=True)
# %% [markdown]
# Apply t-test and FDR, detect & remove non-active voxels
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

# %%
# num_trials = beta.shape[-1]
clean_active_volume = np.full(bold_data.shape[:3]+(num_trials,), np.nan)
active_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
clean_active_volume[active_coords[0], active_coords[1], active_coords[2], :] = clean_active_beta
print(5, flush=True)

# %% [markdown]
# apply filter

# %%
def hampel_filter_image(image, window_size, threshold_factor, return_stats=False):
    footprint = np.ones((window_size,) * 3, dtype=bool)
    insufficient_any = np.zeros(image.shape[:3], dtype=bool)
    corrected_any = np.zeros(image.shape[:3], dtype=bool)

    for t in range(image.shape[3]):
        print(f"Trial Number: {t}", flush=True)
        vol = image[..., t]
        valid = np.isfinite(vol)

        med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0).astype(np.float32)
        neighbor_count = counts - valid.astype(np.float32)

        scaled_mad = 1.4826 * mad
        insufficient = valid & (neighbor_count < 3)
        insufficient_any |= insufficient
        image[..., t][insufficient] = np.nan

        enough_data = (neighbor_count >= 3) & valid
        outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)

        corrected_any |= outliers
        image[..., t][outliers] = med[outliers]

    if return_stats:
        stats = {
            'insufficient_total': int(np.count_nonzero(insufficient_any)),
            'corrected_total': int(np.count_nonzero(corrected_any)),
        }
        return image, stats

    return image

beta_volume_filter, hampel_stats = hampel_filter_image(clean_active_volume.astype(np.float32), window_size=5, threshold_factor=3, return_stats=True)
# print('Insufficient neighbours per frame:', hampel_stats['insufficient_counts'], flush=True)
print('Total voxels with <3 neighbours:', hampel_stats['insufficient_total'], flush=True)
print('Total corrected voxels:', hampel_stats['corrected_total'], flush=True)

# beta_volume_filter = beta_volume_filter[~np.all(np.isnan(beta_volume_filter), axis=-1)]
# np.save(f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)

nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1) 
mask_2d = nan_voxels.reshape(-1) 

np.save(output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)
np.save(output_dir / f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy', mask_2d)

######################################################################################################################
######################################################################################################################
######################################################################################################################

beta_volume_filter = np.load(output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy')
nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1) 
mask_2d = np.load(output_dir / f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy')
np.save(output_dir / f"nan_mask_flat_sub{sub}_ses{ses}_run{run}.npy", mask_2d)
beta_volume_clean_2d = beta_volume_filter[~nan_voxels]     
np.save(output_dir / f"beta_volume_filter_sub{sub}_ses{ses}_run{run}.npy", beta_volume_clean_2d) 

active_flat_idx = np.ravel_multi_index(active_coords, nan_voxels.shape)
np.save(output_dir / f"active_flat_indices__sub{sub}_ses{ses}_run{run}.npy", active_flat_idx)
keep_mask = ~mask_2d[active_flat_idx]
clean_active_bold = clean_active_bold[keep_mask, ...]
np.save(output_dir / f"active_bold_sub{sub}_ses{ses}_run{run}.npy", clean_active_bold)
clean_active_beta = clean_active_beta[keep_mask, ...]

# Drop voxels everywhere our Hampel mask removed them
clean_active_idx = clean_active_idx[keep_mask]
active_coords = tuple(coord[keep_mask] for coord in active_coords)
np.save(output_dir / f"active_coords_sub{sub}_ses{ses}_run{run}.npy", active_coords)

# active_flat_idx = np.ravel_multi_index(active_coords, clean_active_volume.shape[:3])
# active_keep_mask = ~mask_2d[active_flat_idx]
# clean_active_bold = clean_active_bold[active_keep_mask]

######################################################################################################################
######################################################################################################################
######################################################################################################################
# Visualize mean clean active beta volume on the anatomical image
with np.errstate(invalid='ignore'):
    mean_clean_active = np.nanmean(np.abs(beta_volume_filter), axis=-1).astype(np.float32)
_save_beta_overlay(
    mean_clean_active,
    anat_img=anat_img,
    out_html=str(output_dir / f'clean_active_beta_overlay_sub{sub}_ses{ses}_run{run}.html'),
    cut_coords=cut_coords,
)
