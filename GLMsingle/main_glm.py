# %%
import argparse
import gc
import shlex
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


CONFIG = {'subject_id': '09', 'session_id': '1', 'runs': ['1', '2'], 'paths': {'derivatives': None, 'output': None, 'go_times': None},
    'files': {'bold_template': 'sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz',
        'brain_mask': 'sub-pd0{sub}_ses-{ses}_T1w_brain_mask.nii.gz',
        'csf_mask': 'sub-pd0{sub}_ses-{ses}_T1w_brain_pve_0.nii.gz','gray_mask': 'sub-pd0{sub}_ses-{ses}_T1w_brain_pve_1.nii.gz',
        'anat': 'sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz'},
    'mask_thresholds': {'brain': 0.0, 'csf': 0.0, 'gray': 0.5},
    'design': {'timepoints': 850, 'num_trials': 90, 'stimdur': 9, 'tr': 1.0},
    'glmsingle': {'wantlibrary': 1, 'wantglmdenoise': 1, 'wantfracridge': 1, 'wantfileoutputs': [1, 1, 1, 1], 'wantmemoryoutputs': [0, 0, 0, 1]}}


def _parse_list(value):
    if isinstance(value, list):
        return value
    return [item.strip() for item in str(value).split(',') if item.strip()]


def apply_mask(anat_data, bold_data, nonzero_mask):
    cleaned_bold_data = bold_data[nonzero_mask]
    cleaned_anat_data = anat_data[nonzero_mask]
    print('combined mask voxels:', nonzero_mask[0].size)
    print('bold_data masked shape:', cleaned_bold_data.shape)
    return cleaned_bold_data, cleaned_anat_data


def _build_mask(brain_mask, csf_mask, gray_mask, mask_mode: str):
    if mask_mode == 'brain_minus_csf':
        return np.logical_and(brain_mask, ~csf_mask)
    if mask_mode == 'brain_only':
        return brain_mask
    if mask_mode == 'gray_only':
        return gray_mask
    if mask_mode == 'gray_minus_csf':
        return np.logical_and(gray_mask, ~csf_mask)
    raise ValueError(f'Unknown mask mode: {mask_mode}')


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


def _trial_metrics(masked_bold, onsets, stimdur, metric):
    metrics = []
    for onset in onsets:
        start = int(onset) - 1
        end = start + stimdur
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


def _parse_beta_args(beta_args: str) -> dict:
    if not beta_args:
        return {}
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--subject')
    parser.add_argument('--session', type=int)
    parser.add_argument('--run', type=int)
    parser.add_argument('--typed-path', type=Path, dest='typed_path')
    parser.add_argument('--output-dir', type=Path, dest='output_dir')
    parser.add_argument('--data-root', type=Path, dest='data_root')
    parser.add_argument('--glmsingle-root', type=Path, dest='glmsingle_root')
    args, _ = parser.parse_known_args(shlex.split(beta_args))
    return {key: value for key, value in vars(args).items() if value is not None}


# %%
cfg = CONFIG
sub = cfg['subject_id']
ses = cfg['session_id']
runs = _parse_list(cfg['runs'])

mask_thresholds = {key: float(value) for key, value in cfg['mask_thresholds'].items()}
mask_mode = 'brain_minus_csf'

extra_mode = 'csf'
trial_metric = 'mean_abs'
trial_z = 0.0
trial_fallback = 0.0
trial_max_drop = 0.2
wantglmdenoise = int(cfg['glmsingle']['wantglmdenoise'])
wantfracridge = int(cfg['glmsingle']['wantfracridge'])

paths = cfg['paths']
files_cfg = cfg['files']

data_root = Path(__file__).resolve().parent
data_root = data_root.expanduser().resolve()
paths['derivatives'] = str(data_root)
paths['output'] = str(data_root / f'GLMOutputs-sub{sub}-ses{ses}')
paths['go_times'] = str(data_root / f'PSPD0{sub}-ses-{ses}-go-times.txt')


typed_path = data_root / f'TYPED_FITHRF_GLMDENOISE_RR_sub{sub}_ses{int(ses):02d}.npy'
if not typed_path.exists():
    typed_path = None

run_glm = typed_path is None
if not run_glm:
    if typed_path is not None:
        print(f'Using existing GLMsingle output: {typed_path}', flush=True)
    print('Skipping GLMsingle fit.', flush=True)

if run_glm:
    outputdir_glmsingle = Path(paths['output'])
    outputdir_glmsingle.mkdir(parents=True, exist_ok=True)

    anat_dir, func_dir = _resolve_data_dirs(data_root, sub, ses)

    brain_mask = nib.load(join(anat_dir, files_cfg['brain_mask'].format(sub=sub, ses=ses)))
    csf_mask = nib.load(join(anat_dir, files_cfg['csf_mask'].format(sub=sub, ses=ses)))
    gray_mask = nib.load(join(anat_dir, files_cfg['gray_mask'].format(sub=sub, ses=ses)))
    anat_file = nib.load(join(anat_dir, files_cfg['anat'].format(sub=sub, ses=ses)))

    brain_mask_data = brain_mask.get_fdata() > mask_thresholds['brain']
    csf_mask_data = csf_mask.get_fdata() > mask_thresholds['csf']
    gray_mask_data = gray_mask.get_fdata() > mask_thresholds['gray']
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
    configured_timepoints = cfg['design'].get('timepoints')
    if configured_timepoints is None:
        T = run_timepoints
    else:
        T = configured_timepoints
        if T != run_timepoints:
            print(f'Warning: configured timepoints ({T}) differ from BOLD length ({run_timepoints}).')

    num_trials = int(cfg['design']['num_trials'])
    stimdur = int(cfg['design']['stimdur'])
    tr = float(cfg['design']['tr'])

    go_times_path = Path(paths['go_times'])
    go_flag = np.loadtxt(go_times_path, dtype=int)
    if go_flag.ndim == 1:
        go_flag = go_flag[np.newaxis, :]
    if go_flag.shape[0] < len(runs):
        raise ValueError('Go-times file does not contain enough rows for the configured runs.')

    trial_keep_by_run = []
    for idx, run in enumerate(runs):
        run_onsets = go_flag[idx][:num_trials]
        metrics = _trial_metrics(data[idx], run_onsets, stimdur, trial_metric)
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
        run_onsets = go_flag[idx][:num_trials]
        keep = trial_keep_by_run[idx]
        for onset, keep_trial in zip(run_onsets, keep):
            if not keep_trial:
                continue
            onset_idx = onset - 1
            if onset_idx < 0 or onset_idx >= T:
                raise ValueError(f'Invalid onset {onset} for run {run} with T={T}')
            design[onset_idx, 0] = 1
        design_matrix.append(design)

    opt = dict(cfg['glmsingle'])
    opt['wantglmdenoise'] = wantglmdenoise
    opt['wantfracridge'] = wantfracridge
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
beta_args = ''
beta_overrides = _parse_beta_args(beta_args)
run_beta = True
if run_beta:
    beta_script = Path(__file__).resolve().parent.parent / 'Beta_preprocessing.py'
    cmd = [sys.executable, str(beta_script)]
    if beta_args:
        cmd.extend(shlex.split(beta_args))
    if typed_path is not None and beta_overrides.get('typed_path') is None:
        cmd.extend(['--typed-path', str(typed_path)])
    subprocess.run(cmd, check=True)

run_roi = True
if run_roi:
    roi_sub = str(beta_overrides.get('subject', sub))
    roi_ses = int(beta_overrides.get('session', int(ses)))
    fallback_run = runs[0] if runs else '1'
    roi_run = int(beta_overrides.get('run', int(fallback_run)))
    roi_data_root = Path(beta_overrides.get('data_root', data_root)).expanduser().resolve()
    beta_output_dir = Path(beta_overrides.get('output_dir', f'sub{roi_sub}')).expanduser().resolve()

    roi_output_dir = beta_output_dir

    atlas_threshold = 25
    summary_stat = 'mean_abs'
    top_n = 10
    motor_patterns = _parse_list('Precentral Gyrus')
    motor_patterns = motor_patterns or ['Precentral Gyrus']
    motor_corr_threshold = 0.0
    motor_corr_abs = False
    atlas_register = False
    atlas_register_reference = 'anat'
    atlas_register_force = False
    atlas_mni_template = None
    atlas_data_dir = None

    _run_roi_analysis(
        data_root=roi_data_root,
        files_cfg=files_cfg,
        sub=roi_sub,
        ses=roi_ses,
        run=roi_run,
        beta_output_dir=beta_output_dir,
        output_dir=roi_output_dir,
        atlas_threshold=atlas_threshold,
        summary_stat=summary_stat,
        motor_patterns=motor_patterns,
        motor_corr_threshold=motor_corr_threshold,
        motor_corr_abs=motor_corr_abs,
        atlas_register=atlas_register,
        atlas_register_reference=atlas_register_reference,
        atlas_mni_template=atlas_mni_template,
        atlas_data_dir=atlas_data_dir,
        atlas_register_force=atlas_register_force,
        top_n=top_n,
    )
