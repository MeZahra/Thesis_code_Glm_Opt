# %%
import gc
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

from roi_analysis import _resolve_data_dirs, _run_roi_analysis


# User configuration
subject_id = '09'
session_id = '1'
runs = ['1', '2']

bold_template = 'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'
brain_mask_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz'
csf_mask_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz'
gray_mask_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz'
anat_template = 'sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz'
go_times_root = Path(
    '/mnt/TeamShare/Data_Masterfile/Zahra-Thesis-Data/Master_Thesis_Files/GLM_single_results/Go-times'
)
go_times_template = 'PSPD0{sub}-ses-{ses}-go-times.txt'

mask_threshold_brain = 0.0
mask_threshold_csf = 0.0
mask_threshold_gray = 0.5
mask_mode = 'brain_csf'

design_timepoints = 850
design_num_trials = 90
design_stimdur = 9
design_tr = 1.0
trial_block_size = 30
trial_rest_trs = 20

extra_mode = 'csf'
trial_metric = 'mean_abs'
trial_z = 0.0
trial_fallback = 0.0
trial_max_drop = 0.2

glmsingle_wantlibrary = 1
glmsingle_wantglmdenoise = 1
glmsingle_wantfracridge = 1
glmsingle_wantfileoutputs = [1, 1, 1, 1]
glmsingle_wantmemoryoutputs = [0, 0, 0, 1]

run_beta = True
beta_subject = None
beta_session = None
beta_run = None
beta_typed_path = None
beta_output_dir = None
beta_data_root = None
beta_glmsingle_root = None
beta_extra_args = []

run_roi = True
atlas_threshold = 25
summary_stat = 'mean_abs'
top_n = 10
motor_patterns = ['Precentral Gyrus']
motor_corr_threshold = 0.0
motor_corr_abs = False
atlas_register = False
atlas_register_reference = 'anat'
atlas_register_force = False
atlas_mni_template = None
atlas_data_dir = None


def apply_mask(anat_data, bold_data, nonzero_mask):
    cleaned_bold_data = bold_data[nonzero_mask]
    cleaned_anat_data = anat_data[nonzero_mask]
    print('combined mask voxels:', nonzero_mask[0].size)
    print('bold_data masked shape:', cleaned_bold_data.shape)
    return cleaned_bold_data, cleaned_anat_data

def _build_mask(brain_mask, csf_mask, gray_mask, mask_mode: str):
    if mask_mode == 'brain_csf':
        return np.logical_and(brain_mask, ~csf_mask)
    if mask_mode == 'brain_csf_gray':
        return np.logical_and(brain_mask, np.logical_and(gray_mask, ~csf_mask))

def _extract_regressors(bold_run, brain_mask, csf_mask, gray_mask, mode: str):
    if mode == 'none':
        return None
    parts = [p.strip() for p in mode.split('+') if p.strip()]
    regressors = []
    for part in parts:
        if part == 'csf':
            regressors.append(np.mean(bold_run[csf_mask], axis=0))
        elif part == 'gray':
            regressors.append(np.mean(bold_run[gray_mask], axis=0))
        elif part in {'brain', 'global'}:
            regressors.append(np.mean(bold_run[brain_mask], axis=0))
        else:
            raise ValueError(f'Unknown regressor mode: {part}')
    if not regressors:
        return None
    return np.vstack(regressors).T.astype(np.float32)


def _trial_onsets_from_blocks(num_trials, stimdur, trials_per_block, rest_tr, total_timepoints=None):
    onsets = []
    onset = 1
    for trial_idx in range(num_trials):
        onsets.append(onset)
        onset += stimdur
        if (trial_idx + 1) % trials_per_block == 0 and (trial_idx + 1) < num_trials:
            onset += rest_tr
    onsets = np.array(onsets, dtype=np.int32)
    if total_timepoints is not None:
        expected_end = int(onsets[-1]) - 1 + stimdur
        if expected_end > total_timepoints:
            raise ValueError(
                f'Expected {expected_end} TRs from trial structure, but only have {total_timepoints}.'
            )
        if expected_end < total_timepoints:
            print(
                f'Warning: expected {expected_end} TRs from trial structure, '
                f'but have {total_timepoints}. Trailing TRs will be ignored.'
            )
    return onsets


def _trial_metrics(masked_bold, onsets, stimdur, metric):
    metrics = []
    for onset in onsets:
        start = int(onset) - 1
        end = start + stimdur
        if end > masked_bold.shape[1]:
            raise ValueError(
                f'Trial window [{start}:{end}) exceeds BOLD length {masked_bold.shape[1]}.'
            )
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


def _trial_keep_mask(metrics, z_thr, fallback_pct, max_drop_fraction):
    if z_thr <= 0:
        return np.ones_like(metrics, dtype=bool)

    med = float(np.nanmedian(metrics))
    mad = float(np.nanmedian(np.abs(metrics - med)))
    scale = 1.4826 * max(mad, 1e-9)
    z = (metrics - med) / scale

    keep = z <= z_thr
    keep &= np.isfinite(metrics)

    if np.all(keep) and fallback_pct > 0:
        cutoff = float(np.nanpercentile(metrics, fallback_pct))
        keep = metrics <= cutoff

    drop_fraction = 1 - np.mean(keep)
    if drop_fraction > max_drop_fraction:
        n_keep = int(np.ceil(len(metrics) * (1 - max_drop_fraction)))
        order = np.argsort(metrics)
        keep = np.zeros_like(metrics, dtype=bool)
        keep[order[:n_keep]] = True

    return keep


# %%
sub = subject_id
ses = session_id
runs = list(runs)

files_cfg = {
    'bold_template': bold_template,
    'brain_mask': brain_mask_template,
    'csf_mask': csf_mask_template,
    'gray_mask': gray_mask_template,
    'anat': anat_template,
}

data_root = Path(__file__).resolve().parent
data_root = data_root.expanduser().resolve()
outputdir_glmsingle = data_root / f'GLMOutputs-sub{sub}-ses{ses}'
go_times_path = go_times_root / go_times_template.format(sub=sub, ses=ses)


typed_path = data_root / f'TYPED_FITHRF_GLMDENOISE_RR_sub{sub}_ses{int(ses):02d}.npy'
if not typed_path.exists():
    typed_path = None

run_glm = typed_path is None
if not run_glm:
    if typed_path is not None:
        print(f'Using existing GLMsingle output: {typed_path}', flush=True)
    print('Skipping GLMsingle fit.', flush=True)

if run_glm:
    outputdir_glmsingle.mkdir(parents=True, exist_ok=True)

    anat_dir, func_dir = _resolve_data_dirs(data_root, sub, ses)

    brain_mask = nib.load(join(anat_dir, files_cfg['brain_mask'].format(sub=sub, ses=ses)))
    csf_mask = nib.load(join(anat_dir, files_cfg['csf_mask'].format(sub=sub, ses=ses)))
    gray_mask = nib.load(join(anat_dir, files_cfg['gray_mask'].format(sub=sub, ses=ses)))
    anat_file = nib.load(join(anat_dir, files_cfg['anat'].format(sub=sub, ses=ses)))

    brain_mask_data = brain_mask.get_fdata() > mask_threshold_brain
    csf_mask_data = csf_mask.get_fdata() > mask_threshold_csf
    gray_mask_data = gray_mask.get_fdata() > mask_threshold_gray
    mask = _build_mask(brain_mask_data, csf_mask_data, gray_mask_data, mask_mode)
    mask_indices = np.where(mask)

    print('Combined mask voxels:', mask_indices[0].size)

    anat_data = anat_file.get_fdata(dtype=np.float32)

    # %%
    run_timepoints = None

    data = []
    extraregressors = []
    trial_keep_by_run = []

    for run in runs:
        bold_name = files_cfg['bold_template'].format(sub=sub, ses=ses, run=run)
        bold_path = join(func_dir, bold_name)
        bold_run = nib.load(bold_path).get_fdata(dtype=np.float32)
        masked_bold, _ = apply_mask(anat_data, bold_run, mask_indices)
        data.append(masked_bold)

        if run_timepoints is None:
            run_timepoints = bold_run.shape[-1]

        regressors = _extract_regressors(bold_run, brain_mask_data, csf_mask_data, gray_mask_data, extra_mode)
        if regressors is not None:
            extraregressors.append(regressors)
        print(f'Run {run}: masked data shape {masked_bold.shape}')

    del anat_data

    # %%
    run_timepoints = run_timepoints or data[0].shape[-1]
    configured_timepoints = design_timepoints
    if configured_timepoints is None:
        T = run_timepoints
    else:
        T = configured_timepoints
        if T != run_timepoints:
            print(f'Warning: configured timepoints ({T}) differ from BOLD length ({run_timepoints}).')

    num_trials = int(design_num_trials)
    stimdur = int(design_stimdur)
    tr = float(design_tr)

    go_flag = np.loadtxt(go_times_path, dtype=int)
    if go_flag.ndim == 1:
        go_flag = go_flag[np.newaxis, :]
    if go_flag.shape[0] < len(runs):
        raise ValueError('Go-times file does not contain enough rows for the configured runs.')
    if go_flag.shape[1] < num_trials:
        raise ValueError('Go-times file does not contain enough trials for the configured design.')

    run_onsets_metric = _trial_onsets_from_blocks(
        num_trials,
        stimdur,
        trial_block_size,
        trial_rest_trs,
        total_timepoints=T,
    )

    trial_keep_by_run = []
    for idx, run in enumerate(runs):
        metrics = _trial_metrics(data[idx], run_onsets_metric, stimdur, trial_metric)
        keep = _trial_keep_mask(metrics, trial_z, trial_fallback, trial_max_drop)
        trial_keep_by_run.append(keep)
        np.save(outputdir_glmsingle / f'trial_keep_run{run}.npy', keep)
        np.save(outputdir_glmsingle / f'trial_metric_run{run}.npy', metrics)
        dropped = int(np.count_nonzero(~keep))
        print(f'Run {run}: dropping {dropped}/{len(keep)} trials (metric={trial_metric}).')

    # %%
    design_matrix = []
    for idx, run in enumerate(runs):
        design = np.zeros((T, 1), dtype=int)
        run_onsets_design = go_flag[idx][:num_trials]
        keep = trial_keep_by_run[idx]
        for onset, keep_trial in zip(run_onsets_design, keep):
            if not keep_trial:
                continue
            onset_idx = onset - 1
            if onset_idx < 0 or onset_idx >= T:
                raise ValueError(f'Invalid onset {onset} for run {run} with T={T}')
            design[onset_idx, 0] = 1
        design_matrix.append(design)

    opt = {
        'wantlibrary': glmsingle_wantlibrary,
        'wantglmdenoise': int(glmsingle_wantglmdenoise),
        'wantfracridge': int(glmsingle_wantfracridge),
        'wantfileoutputs': glmsingle_wantfileoutputs,
        'wantmemoryoutputs': glmsingle_wantmemoryoutputs,
    }
    if extraregressors and len(extraregressors) == len(runs):
        opt['extra_regressors'] = extraregressors

    # %%
    glmsingle_obj = GLM_single(opt)
    start_time = time.time()

    print('running GLMsingle...')
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024 / 1024
    print(f'Memory usage before GLMsingle: {memory_before:.2f} GB')

    gc.collect()
    results_glmsingle = glmsingle_obj.fit(design_matrix, data, stimdur, tr, outputdir=str(outputdir_glmsingle))

    elapsed_time = time.time() - start_time

    print('\telapsed time: ', f'{time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
else:
    print('Skipping GLMsingle fit; using existing outputs if available.', flush=True)

# %%
# if run_beta:
#     beta_script = Path(__file__).resolve().parent.parent / 'Beta_preprocessing.py'
#     cmd = [sys.executable, str(beta_script)]
#     if beta_subject is not None:
#         cmd.extend(['--subject', str(beta_subject)])
#     if beta_session is not None:
#         cmd.extend(['--session', str(beta_session)])
#     if beta_run is not None:
#         cmd.extend(['--run', str(beta_run)])
#     if beta_typed_path is not None:
#         cmd.extend(['--typed-path', str(beta_typed_path)])
#     elif typed_path is not None:
#         cmd.extend(['--typed-path', str(typed_path)])
#     if beta_output_dir is not None:
#         cmd.extend(['--output-dir', str(beta_output_dir)])
#     if beta_data_root is not None:
#         cmd.extend(['--data-root', str(beta_data_root)])
#     if beta_glmsingle_root is not None:
#         cmd.extend(['--glmsingle-root', str(beta_glmsingle_root)])
#     if beta_extra_args:
#         cmd.extend(beta_extra_args)
#     subprocess.run(cmd, check=True)

# if run_roi:
    # roi_sub = str(beta_subject if beta_subject is not None else sub)
    # roi_ses = int(beta_session if beta_session is not None else ses)
    # fallback_run = runs[0] if runs else '1'
    # roi_run = int(beta_run if beta_run is not None else fallback_run)
    # roi_data_root = Path(beta_data_root if beta_data_root is not None else data_root).expanduser().resolve()
    # beta_output_dir_path = Path(beta_output_dir if beta_output_dir is not None else f'sub{roi_sub}').expanduser().resolve()

    # roi_output_dir = beta_output_dir_path
    # roi_motor_patterns = motor_patterns or ['Precentral Gyrus']

    # _run_roi_analysis(
    #     data_root=roi_data_root,
    #     files_cfg=files_cfg,
    #     sub=roi_sub,
    #     ses=roi_ses,
    #     run=roi_run,
    #     beta_output_dir=beta_output_dir_path,
    #     output_dir=roi_output_dir,
    #     atlas_threshold=atlas_threshold,
    #     summary_stat=summary_stat,
    #     motor_patterns=roi_motor_patterns,
    #     motor_corr_threshold=motor_corr_threshold,
    #     motor_corr_abs=motor_corr_abs,
    #     atlas_register=atlas_register,
    #     atlas_register_reference=atlas_register_reference,
    #     atlas_mni_template=atlas_mni_template,
    #     atlas_data_dir=atlas_data_dir,
    #     atlas_register_force=atlas_register_force,
    #     top_n=top_n,
    # )
