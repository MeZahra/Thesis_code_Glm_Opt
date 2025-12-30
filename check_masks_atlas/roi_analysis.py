import csv
import json
import os
import subprocess
from os.path import join
from pathlib import Path

import nibabel as nib
import numpy as np


def _resolve_data_dirs(data_root: Path, sub: str, ses: str) -> tuple[Path, Path]:
    base = data_root / f'sub-pd0{sub}' / f'ses-{ses}'
    if base.exists():
        return base / 'anat', base / 'func'
    return data_root, data_root


def _find_first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = '\n  '.join(str(path) for path in paths)
    raise FileNotFoundError(f'Could not find expected files:\n  {joined}')


def _beta_file_candidates(output_dir: Path, sub: str, ses: int, run: int, prefix: str) -> list[Path]:
    ses_values = {str(ses), f'{int(ses):02d}'}
    run_values = {str(run), f'{int(run):02d}'}
    names = []
    for ses_val in ses_values:
        for run_val in run_values:
            base = f'{prefix}_sub{sub}_ses{ses_val}_run{run_val}.npy'
            names.append(base)
            names.append(f'{base}.npy')
    seen = set()
    candidates = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidates.append(output_dir / name)
    return candidates


def _load_active_beta_for_roi(output_dir: Path, sub: str, ses: int, run: int):
    beta_candidates = _beta_file_candidates(output_dir, sub, ses, run, 'beta_volume_filter')
    coords_candidates = _beta_file_candidates(output_dir, sub, ses, run, 'active_coords')
    beta_path = _find_first_existing(beta_candidates)
    coords_path = _find_first_existing(coords_candidates)
    beta_2d = np.load(beta_path, mmap_mode='r')
    coords = np.load(coords_path, allow_pickle=True)
    coords = np.asarray(coords)
    if coords.shape[0] != 3:
        raise ValueError(f'active_coords should have shape (3, N); got {coords.shape}')
    return beta_2d, coords, beta_path, coords_path


def _compute_voxel_summary(beta_2d: np.ndarray, stat: str) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        if stat == 'mean_abs':
            summary = np.nanmean(np.abs(beta_2d), axis=1)
        elif stat == 'mean':
            summary = np.nanmean(beta_2d, axis=1)
        elif stat == 'percentile_95':
            summary = np.nanpercentile(np.abs(beta_2d), 95, axis=1)
        elif stat == 'percentile_90':
            summary = np.nanpercentile(np.abs(beta_2d), 90, axis=1)
        elif stat == 'peak':
            summary = np.nanmax(np.abs(beta_2d), axis=1)
        elif stat == 'mean_z':
            mean_beta = np.nanmean(beta_2d, axis=1)
            finite = mean_beta[np.isfinite(mean_beta)]
            scale = float(np.nanstd(finite)) if finite.size else 0.0
            if scale <= 0:
                scale = 1.0
            center = float(np.nanmean(finite)) if finite.size else 0.0
            summary = (mean_beta - center) / scale
        else:
            raise ValueError(f'Unknown ROI summary stat: {stat}')
    return summary


def _fetch_ho_atlas(atlas_threshold: int, data_dir: Path | None = None):
    from nilearn import datasets

    atlas_name = f'cort-maxprob-thr{atlas_threshold}-2mm'
    atlas = datasets.fetch_atlas_harvard_oxford(atlas_name, data_dir=str(data_dir) if data_dir else None)
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    labels = list(atlas.labels)
    atlas_path = atlas.maps if isinstance(atlas.maps, str) else None
    return atlas_img, labels, atlas_name, atlas_path


def _load_resampled_atlas(bold_path: Path, atlas_threshold: int, data_dir: Path | None = None):
    from nilearn import image

    atlas_img, labels, atlas_name, atlas_path = _fetch_ho_atlas(atlas_threshold, data_dir=data_dir)
    bold_img = nib.load(str(bold_path))
    atlas_resampled = image.resample_to_img(
        atlas_img,
        bold_img,
        interpolation='nearest',
        force_resample=True,
        copy_header=True,
    )
    atlas_data = atlas_resampled.get_fdata().astype(np.int32)
    return atlas_data, labels, atlas_name, atlas_path


def _resolve_mni_template(path: Path | None = None) -> Path | None:
    if path is not None and path.exists():
        return path
    fsl_dir = os.environ.get('FSLDIR')
    if not fsl_dir:
        return None
    fsl_dir = Path(fsl_dir)
    candidates = [
        fsl_dir / 'data' / 'standard' / 'MNI152_T1_2mm_brain.nii.gz',
        fsl_dir / 'data' / 'standard' / 'MNI152_T1_1mm_brain.nii.gz',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_flirt(cmd: list[str]):
    env = os.environ.copy()
    env.setdefault('FSLOUTPUTTYPE', 'NIFTI_GZ')
    subprocess.run(cmd, check=True, env=env)


def _load_registered_atlas(
    *,
    data_root: Path,
    files_cfg: dict,
    sub: str,
    ses: int,
    bold_path: Path,
    atlas_threshold: int,
    output_dir: Path,
    reference: str,
    mni_template: Path | None,
    data_dir: Path | None,
    force: bool,
):
    import shutil

    atlas_img, labels, atlas_name, atlas_path = _fetch_ho_atlas(atlas_threshold, data_dir=data_dir)
    if atlas_path is None:
        cache_dir = output_dir / 'atlas_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        atlas_path = cache_dir / f'{atlas_name}.nii.gz'
        if not atlas_path.exists():
            nib.save(atlas_img, str(atlas_path))
    atlas_path = Path(atlas_path)

    if reference == 'bold':
        ref_path = bold_path
    else:
        anat_dir, _ = _resolve_data_dirs(data_root, sub, ses)
        ref_path = Path(join(anat_dir, files_cfg['anat'].format(sub=sub, ses=ses)))

    flirt = shutil.which('flirt')
    if not flirt:
        raise RuntimeError('FSL flirt not found in PATH.')

    mni_template = _resolve_mni_template(mni_template)
    if mni_template is None:
        raise RuntimeError('Could not resolve MNI template for atlas registration.')

    reg_dir = output_dir / f'atlas_{atlas_name}_to_{reference}'
    reg_dir.mkdir(parents=True, exist_ok=True)
    mat_path = reg_dir / 'mni_to_ref.mat'
    warped_path = reg_dir / 'atlas_in_ref.nii.gz'

    if force or not warped_path.exists():
        _run_flirt([
            flirt,
            '-in',
            str(mni_template),
            '-ref',
            str(ref_path),
            '-omat',
            str(mat_path),
            '-out',
            str(reg_dir / 'mni_in_ref.nii.gz'),
            '-dof',
            '12',
        ])
        _run_flirt([
            flirt,
            '-in',
            str(atlas_path),
            '-ref',
            str(ref_path),
            '-applyxfm',
            '-init',
            str(mat_path),
            '-interp',
            'nearestneighbour',
            '-out',
            str(warped_path),
        ])

    atlas_data = nib.load(str(warped_path)).get_fdata().astype(np.int32)
    return atlas_data, labels, atlas_name, str(warped_path), {
        'registered': True,
        'method': 'flirt',
        'reference': reference,
        'mni_template': str(mni_template),
        'warp_dir': str(reg_dir),
    }


def _compute_roi_stats(
    labels_at_active: np.ndarray,
    voxel_summary: np.ndarray,
    labels: list[str],
    voxel_mask: np.ndarray | None = None,
):
    valid = (labels_at_active > 0) & np.isfinite(voxel_summary)
    if voxel_mask is not None:
        valid &= voxel_mask
    if not np.any(valid):
        empty = np.full(len(labels), np.nan)
        return [], np.zeros(len(labels), dtype=int), empty, empty

    label_ids = labels_at_active[valid].astype(int)
    values = voxel_summary[valid].astype(np.float64)
    in_range = (label_ids >= 0) & (label_ids < len(labels))
    label_ids = label_ids[in_range]
    values = values[in_range]

    counts = np.bincount(label_ids, minlength=len(labels))
    sums = np.bincount(label_ids, weights=values, minlength=len(labels))
    sum_sq = np.bincount(label_ids, weights=values * values, minlength=len(labels))
    means = np.full(len(labels), np.nan, dtype=np.float64)
    stds = np.full(len(labels), np.nan, dtype=np.float64)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero]
    stds[nonzero] = np.sqrt(np.maximum(sum_sq[nonzero] / counts[nonzero] - means[nonzero] ** 2, 0.0))

    stats = []
    for idx in range(1, len(labels)):
        if counts[idx] == 0:
            continue
        stats.append({
            'index': int(idx),
            'label': labels[idx],
            'mean': float(means[idx]),
            'std': float(stds[idx]),
            'n_voxels': int(counts[idx]),
        })
    return stats, counts, means, stds


def _select_label_indices(labels: list[str], patterns: list[str]) -> list[int]:
    patterns = [pattern.lower() for pattern in patterns]
    indices = []
    for idx, name in enumerate(labels):
        if idx == 0:
            continue
        lname = name.lower()
        if any(pattern in lname for pattern in patterns):
            indices.append(idx)
    return indices


def _motor_corr_filter(
    beta_2d: np.ndarray,
    labels_at_active: np.ndarray,
    motor_indices: list[int],
    threshold: float,
    abs_corr: bool,
) -> tuple[np.ndarray | None, dict]:
    info = {
        'enabled': False,
        'threshold': float(threshold),
        'abs': bool(abs_corr),
        'n_voxels_total': int(beta_2d.shape[0]),
        'n_voxels_kept': None,
        'motor_voxels': 0,
        'reason': None,
    }

    if threshold <= 0 or not motor_indices:
        info['reason'] = 'disabled_or_missing_motor_indices'
        return None, info

    motor_mask = np.isin(labels_at_active, motor_indices)
    motor_count = int(np.count_nonzero(motor_mask))
    info['motor_voxels'] = motor_count
    if motor_count == 0:
        info['reason'] = 'no_motor_voxels'
        return None, info

    with np.errstate(invalid='ignore'):
        motor_series = np.nanmean(beta_2d[motor_mask], axis=0)
    if not np.isfinite(motor_series).any():
        info['reason'] = 'motor_series_all_nan'
        return None, info

    motor_series = motor_series - float(np.nanmean(motor_series))
    motor_std = float(np.nanstd(motor_series))
    if motor_std <= 0:
        info['reason'] = 'motor_series_zero_variance'
        return None, info
    motor_series = motor_series / (motor_std + 1e-6)

    beta_filled = np.array(beta_2d, dtype=np.float32)
    voxel_means = np.nanmean(beta_filled, axis=1)
    nan_idx = np.where(np.isnan(beta_filled))
    if nan_idx[0].size:
        beta_filled[nan_idx] = np.take(voxel_means, nan_idx[0])

    beta_filled -= beta_filled.mean(axis=1, keepdims=True)
    beta_filled /= (beta_filled.std(axis=1, keepdims=True) + 1e-6)

    corr = beta_filled @ motor_series / motor_series.size
    if abs_corr:
        corr = np.abs(corr)

    keep_mask = corr >= threshold
    info['enabled'] = True
    info['n_voxels_kept'] = int(np.count_nonzero(keep_mask))
    return keep_mask, info


def _run_roi_analysis(
    *,
    data_root: Path,
    files_cfg: dict,
    sub: str,
    ses: int,
    run: int,
    beta_output_dir: Path,
    output_dir: Path,
    atlas_threshold: int,
    summary_stat: str,
    motor_patterns: list[str],
    motor_corr_threshold: float,
    motor_corr_abs: bool,
    atlas_register: bool,
    atlas_register_reference: str,
    atlas_mni_template: Path | None,
    atlas_data_dir: Path | None,
    atlas_register_force: bool,
    top_n: int,
):
    _, func_dir = _resolve_data_dirs(data_root, sub, ses)
    bold_path = Path(join(func_dir, files_cfg['bold_template'].format(sub=sub, ses=ses, run=run)))
    if not bold_path.exists():
        raise FileNotFoundError(f'BOLD file not found for ROI analysis: {bold_path}')

    beta_2d, coords, beta_path, coords_path = _load_active_beta_for_roi(beta_output_dir, sub, ses, run)
    atlas_info = {'registered': False, 'method': 'resample'}
    if atlas_register:
        try:
            atlas_data, labels, atlas_name, atlas_path, reg_info = _load_registered_atlas(
                data_root=data_root,
                files_cfg=files_cfg,
                sub=sub,
                ses=ses,
                bold_path=bold_path,
                atlas_threshold=atlas_threshold,
                output_dir=output_dir,
                reference=atlas_register_reference,
                mni_template=atlas_mni_template,
                data_dir=atlas_data_dir,
                force=atlas_register_force,
            )
            atlas_info.update(reg_info)
        except Exception as exc:
            print(f'Warning: atlas registration failed ({exc}); falling back to resample.', flush=True)
            atlas_data, labels, atlas_name, atlas_path = _load_resampled_atlas(
                bold_path, atlas_threshold, data_dir=atlas_data_dir
            )
            atlas_info = {'registered': False, 'method': 'resample_fallback'}
    else:
        atlas_data, labels, atlas_name, atlas_path = _load_resampled_atlas(
            bold_path, atlas_threshold, data_dir=atlas_data_dir
        )

    labels_at_active = atlas_data[coords[0], coords[1], coords[2]]
    motor_indices = _select_label_indices(labels, motor_patterns)
    voxel_keep_mask, corr_info = _motor_corr_filter(
        beta_2d,
        labels_at_active,
        motor_indices,
        motor_corr_threshold,
        motor_corr_abs,
    )

    voxel_summary = _compute_voxel_summary(beta_2d, summary_stat)
    stats, counts, means, stds = _compute_roi_stats(labels_at_active, voxel_summary, labels, voxel_keep_mask)

    sorted_stats = sorted(stats, key=lambda item: item['mean'], reverse=True)
    rank_by_index = {item['index']: rank + 1 for rank, item in enumerate(sorted_stats)}

    motor_label_stats = []
    for idx in motor_indices:
        if idx >= len(counts) or counts[idx] == 0:
            continue
        motor_label_stats.append({
            'index': int(idx),
            'label': labels[idx],
            'mean': float(means[idx]),
            'std': float(stds[idx]),
            'n_voxels': int(counts[idx]),
            'rank': rank_by_index.get(idx),
        })

    motor_mask = np.isin(labels_at_active, motor_indices)
    if voxel_keep_mask is not None:
        motor_mask &= voxel_keep_mask
    motor_values = voxel_summary[motor_mask]
    motor_count = int(np.count_nonzero(np.isfinite(motor_values)))
    motor_mean = float(np.nanmean(motor_values)) if motor_count else None
    motor_rank = None
    if motor_mean is not None:
        motor_rank = 1 + sum(item['mean'] > motor_mean for item in sorted_stats)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f'roi_activation_summary_sub{sub}_ses{ses}_run{run}.json'
    csv_path = output_dir / f'roi_activation_ranked_sub{sub}_ses{ses}_run{run}.csv'

    summary = {
        'subject': str(sub),
        'session': int(ses),
        'run': int(run),
        'summary_stat': summary_stat,
        'atlas': {
            'name': atlas_name,
            'threshold': int(atlas_threshold),
            'path': str(atlas_path) if atlas_path else None,
            'num_labels': len(labels),
            'registration': atlas_info,
        },
        'inputs': {
            'bold_path': str(bold_path),
            'beta_path': str(beta_path),
            'coords_path': str(coords_path),
        },
        'motor': {
            'label_patterns': motor_patterns,
            'indices': motor_indices,
            'labels': [labels[idx] for idx in motor_indices],
            'combined': {
                'mean': motor_mean,
                'n_voxels': motor_count,
                'rank': motor_rank,
            },
            'per_label': motor_label_stats,
        },
        'voxel_filter': corr_info,
        'roi_stats_sorted': sorted_stats,
    }

    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['rank', 'index', 'label', 'n_voxels', 'mean', 'std'])
        for rank, item in enumerate(sorted_stats, start=1):
            writer.writerow([rank, item['index'], item['label'], item['n_voxels'], item['mean'], item['std']])

    top_n = max(1, top_n)
    print(f'ROI activation summary saved to: {summary_path}', flush=True)
    print(f'ROI activation ranking saved to: {csv_path}', flush=True)
    print(f'Top {min(top_n, len(sorted_stats))} ROIs by {summary_stat}:', flush=True)
    for item in sorted_stats[:top_n]:
        rank = rank_by_index[item['index']]
        print(f'  {rank:02d}. {item["label"]} (idx={item["index"]}) mean={item["mean"]:.4f}', flush=True)
    if motor_mean is not None:
        print(
            f'Motor ROI combined mean={motor_mean:.4f}, rank={motor_rank} of {len(sorted_stats)}.',
            flush=True,
        )
