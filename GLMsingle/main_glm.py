# %%
import gc
import os
import subprocess
import sys
import time
from os.path import join
from pathlib import Path
from pprint import pformat
import nibabel as nib
import numpy as np
import psutil
from glmsingle.glmsingle import GLM_single


subject_id = '09'
session_id = '1'
runs = ['1', '2']

bold = 'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'
brain_mask_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
csf_mask_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
gray_mask_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
anat_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz'
go_times_template = 'PSPD0{sub}-ses-{ses}-go-times.txt'

mask_threshold_brain = 0.0
mask_threshold_csf = 0.0
mask_threshold_gray = 0.7  # Increased from 0.5 to reduce partial volume effects
mask_mode = 'brain_csf_gray'

num_timepoints = 850
num_trials = 90
stimdur = 9
tr = 1.0
trial_block_size = 30
trial_rest_trs = 20


extra_mode = 'csf'

def _env_override(name, cast, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError as exc:
        raise ValueError(f"Invalid value for {name}: {value!r}") from exc

trial_metric = _env_override('GLM_TRIAL_METRIC', str, 'std')
trial_z = _env_override('GLM_TRIAL_Z', float, 3)
trial_fallback = _env_override('GLM_TRIAL_FALLBACK', float, 95)
trial_max_drop = _env_override('GLM_TRIAL_MAX_DROP', float, 0.15)
trial_onsets_source = _env_override('GLM_TRIAL_ONSETS', str, 'blocks').lower()
results_cache_name = 'results_glmsingle.npy'
load_results_dir = _env_override('GLM_LOAD_DIR', str, '').strip()
load_results_dir = Path(load_results_dir).expanduser().resolve() if load_results_dir else None

glmsingle_wantlibrary = 1
glmsingle_wantglmdenoise = 1
glmsingle_wantfracridge = 1
glmsingle_wantfileoutputs = [0, 0, 0, 1]
glmsingle_wantmemoryoutputs = [0, 0, 0, 1]


def apply_mask(anat_data, bold_data, nonzero_mask):
    cleaned_bold_data = bold_data[nonzero_mask]
    cleaned_anat_data = anat_data[nonzero_mask]
    print('combined mask voxels:', nonzero_mask[0].size)
    print('bold_data masked shape:', cleaned_bold_data.shape)
    return cleaned_bold_data, cleaned_anat_data

def _build_mask(brain_mask, csf_mask, gray_mask, mask_mode):
    if mask_mode == 'brain_csf':
        return np.logical_and(brain_mask, ~csf_mask)
    if mask_mode == 'brain_csf_gray':
        return np.logical_and(brain_mask, np.logical_and(gray_mask, ~csf_mask))

def _trial_onsets_from_blocks(num_trials, stimdur, trials_per_block, rest_tr):
    onsets = []
    onset = 1
    for trial_idx in range(num_trials):
        onsets.append(onset)
        onset += stimdur
        if (trial_idx + 1) % trials_per_block == 0 and (trial_idx + 1) < num_trials:
            onset += rest_tr
    onsets = np.array(onsets, dtype=np.int32)
    return onsets

def _trial_metrics(masked_bold, onsets, stimdur, metric):
    metrics = []
    for onset in onsets:
        start = int(onset) - 1
        end = start + int(round(stimdur / tr))
        segment = masked_bold[:, start:end]
        if metric == 'mean_abs':
            value = float(np.nanmean(np.abs(segment)))
        elif metric == 'std':
            value = float(np.nanstd(segment))
        elif metric == 'dvars':
            diff = np.diff(segment, axis=1)
            value = float(np.sqrt(np.nanmean(diff * diff)))
        else:
            raise ValueError(f'Unknown trial metric: {metric}')
        metrics.append(value)
    return np.array(metrics, dtype=np.float32)

def _trial_keep_mask(metrics, z_thr, fallback_pct, max_drop_fraction, metric):
    if z_thr <= 0:
        return np.ones_like(metrics, dtype=bool)

    med = float(np.nanmedian(metrics))
    mad = float(np.nanmedian(np.abs(metrics - med)))
    scale = 1.4826 * max(mad, 1e-9)
    z = (metrics - med) / scale

    if metric == 'mean_abs':
        direction = 'low'
    else:
        direction = 'high'

    if direction == 'high':
        keep = z <= z_thr
    elif direction == 'low':
        keep = z >= -z_thr
    else:
        raise ValueError(f'Unknown trial metric direction: {direction}')
    keep &= np.isfinite(metrics)

    if np.all(keep) and fallback_pct > 0:
        if direction == 'high':
            cutoff = float(np.nanpercentile(metrics, fallback_pct))
            keep = metrics <= cutoff
        else:
            cutoff = float(np.nanpercentile(metrics, 100 - fallback_pct))
            keep = metrics >= cutoff

    drop_fraction = 1 - np.mean(keep)
    if drop_fraction > max_drop_fraction:
        n_keep = int(np.ceil(len(metrics) * (1 - max_drop_fraction)))
        order = np.argsort(metrics)
        keep = np.zeros_like(metrics, dtype=bool)
        if direction == 'high':
            keep[order[:n_keep]] = True
        else:
            keep[order[-n_keep:]] = True

    return keep

def _load_existing_results(load_dir, cache_name):
    if load_dir is None:
        return None
    cache_path = load_dir / cache_name
    if cache_path.is_file():
        print(f'Loading cached GLMsingle results from {cache_path}')
        return np.load(cache_path, allow_pickle=True).item()
    typed_candidates = sorted(load_dir.glob('TYPED_FITHRF_GLMDENOISE_RR*.npy'))
    if typed_candidates:
        typed_path = typed_candidates[0]
        print(f'Loading cached GLMsingle results from {typed_path}')
        return {'typed': np.load(typed_path, allow_pickle=True).item()}
    return None


# %%
sub = subject_id
ses = session_id
runs = list(runs)

print("loading files...")
files_cfg = {'bold_template': bold, 'brain_mask': brain_mask_template, 'csf_mask': csf_mask_template, 'gray_mask': gray_mask_template, 'anat': anat_template}
script_root = Path(__file__).resolve().parent
script_root = script_root.expanduser().resolve()
outputdir_glmsingle = script_root / f'GLMOutputs-sub{sub}-ses{ses}-{trial_metric}'
outputdir_glmsingle.mkdir(parents=True, exist_ok=True)
if load_results_dir is None:
    load_results_dir = outputdir_glmsingle
data_root = script_root.parent
data_dir = data_root / f'sub{sub}_ses{ses}'
go_times_root = data_dir
go_times_path = go_times_root / go_times_template.format(sub=sub, ses=ses)
brain_mask = nib.load(join(data_dir, files_cfg['brain_mask'].format(sub=sub, ses=ses)))
csf_mask = nib.load(join(data_dir, files_cfg['csf_mask'].format(sub=sub, ses=ses)))
gray_mask = nib.load(join(data_dir, files_cfg['gray_mask'].format(sub=sub, ses=ses)))
anat_file = nib.load(join(data_dir, files_cfg['anat'].format(sub=sub, ses=ses)))

print("apply masking...")
brain_mask_data = brain_mask.get_fdata() > mask_threshold_brain
csf_mask_data = csf_mask.get_fdata() > mask_threshold_csf
gray_mask_data = gray_mask.get_fdata() > mask_threshold_gray
mask = _build_mask(brain_mask_data, csf_mask_data, gray_mask_data, mask_mode)
mask_indices = np.where(mask)
print('Combined mask voxels:', mask_indices[0].size)
mask_indices_path = outputdir_glmsingle / 'mask_indices.npy'
np.save(mask_indices_path, np.array(mask_indices, dtype=object))
print(f'Saved mask indices: {mask_indices_path}')
anat_data = anat_file.get_fdata(dtype=np.float32)

# %%
data = []
extraregressors = []
trial_keep_by_run = []

print("Load Bold data....")
for run in runs:
    bold_name = files_cfg['bold_template'].format(sub=sub, ses=ses, run=run)
    bold_path = join(data_dir, bold_name)
    bold_run = nib.load(bold_path).get_fdata(dtype=np.float32)
    masked_bold, _ = apply_mask(anat_data, bold_run, mask_indices)
    data.append(masked_bold)
    
    csf_ts = np.mean(bold_run[csf_mask_data], axis=0)
    extraregressors.append(csf_ts[:, None])

del anat_data

# np.save(f'csf_reg_sub{sub}_ses{ses}.npy', extraregressors)
# %%
print("Select trials...")
go_flag = np.loadtxt(go_times_path, dtype=int)
run_onsets_metric = _trial_onsets_from_blocks(num_trials,stimdur, trial_block_size, trial_rest_trs)

trial_keep_by_run = []
trial_metrics_by_run = []
for idx, run in enumerate(runs):
    if trial_onsets_source == 'go_times':
        run_onsets = go_flag[idx][:num_trials]
    else:
        run_onsets = run_onsets_metric
    metrics = _trial_metrics(data[idx], run_onsets, stimdur, trial_metric)
    keep = _trial_keep_mask(metrics, trial_z, trial_fallback, trial_max_drop, trial_metric)
    trial_keep_by_run.append(keep)
    trial_metrics_by_run.append(metrics)
    dropped = int(np.count_nonzero(~keep))
    print(f'Run {run}: dropping {dropped}/{len(keep)} trials (metric={trial_metric}).')

# %%
print("Create Design matrix...")
design_matrix = []
for idx, run in enumerate(runs):
    design = np.zeros((num_timepoints, 1), dtype=int)
    # FIX: Use consistent onset source for trial selection and design matrix
    if trial_onsets_source == 'go_times':
        run_onsets_design = go_flag[idx][:num_trials]
    else:
        run_onsets_design = run_onsets_metric
    keep = trial_keep_by_run[idx]
    for onset, keep_trial in zip(run_onsets_design, keep):
        if not keep_trial:
            continue
        onset_idx = onset - 1
        if onset_idx < 0 or onset_idx >= num_timepoints:
            raise ValueError(f'Invalid onset {onset} for run {run} with T={num_timepoints}')
        design[onset_idx, 0] = 1
    design_matrix.append(design)

opt = {'wantlibrary': glmsingle_wantlibrary,'wantglmdenoise': int(glmsingle_wantglmdenoise),'wantfracridge': int(glmsingle_wantfracridge),
       'wantfileoutputs': glmsingle_wantfileoutputs, 'wantmemoryoutputs': glmsingle_wantmemoryoutputs}
if len(extraregressors) == len(runs):
    opt['extra_regressors'] = extraregressors
opt['chunklen'] = 100000
# %%
results_glmsingle = _load_existing_results(load_results_dir, results_cache_name)
if results_glmsingle is None and load_results_dir is not None:
    print(f'No cached results found in {load_results_dir}; running GLMsingle.')

if results_glmsingle is None:
    print('running GLMsingle...')
    glmsingle_obj = GLM_single(opt)
    results_glmsingle = glmsingle_obj.fit(design_matrix, data, stimdur, tr, outputdir=str(outputdir_glmsingle))
    np.save(outputdir_glmsingle / results_cache_name, results_glmsingle)

    # GLMsingle wipes outputdir at start, so save sidecar outputs after fit.
    np.save(outputdir_glmsingle / 'mask_indices.npy', np.array(mask_indices, dtype=object))
    for run, keep, metrics in zip(runs, trial_keep_by_run, trial_metrics_by_run):
        np.save(outputdir_glmsingle / f'trial_keep_run{run}.npy', keep)
        np.save(outputdir_glmsingle / f'trial_metric_run{run}.npy', metrics)
