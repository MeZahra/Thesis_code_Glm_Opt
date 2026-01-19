#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Union

import matplotlib
import nibabel as nib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
GLMSINGLE_ROOT = REPO_ROOT / "GLMsingle"
if str(GLMSINGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(GLMSINGLE_ROOT))

from glmsingle.design.convolve_design import convolve_design


def _parse_runs(runs_csv) -> list[int]:
    runs = []
    for item in runs_csv.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            runs.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid run value: {item!r}") from exc
    if not runs:
        raise ValueError("No runs specified.")
    return runs


def _load_design(design_path: Path):
    designinfo = np.load(design_path, allow_pickle=True).item()
    if "designSINGLE" not in designinfo:
        raise KeyError("DESIGNINFO missing designSINGLE.")
    if "params" not in designinfo:
        raise KeyError("DESIGNINFO missing params.")
    params = designinfo["params"]
    if "hrflibrary" not in params:
        raise KeyError("DESIGNINFO params missing hrflibrary.")
    return designinfo["designSINGLE"], params["hrflibrary"]


def _load_model(model_path: Path):
    model = np.load(model_path, allow_pickle=True).item()
    if "betasmd" not in model:
        raise KeyError("Model missing betasmd.")
    if "HRFindexrun" not in model:
        raise KeyError("Model missing HRFindexrun.")
    return model["betasmd"], model["HRFindexrun"]


def _flatten_betas(betasmd: np.ndarray) -> tuple[np.ndarray, int]:
    if betasmd.ndim < 2:
        raise ValueError(f"Unexpected betasmd shape: {betasmd.shape}")
    numtrials = betasmd.shape[-1]
    voxels = int(np.prod(betasmd.shape[:-1]))
    betas = betasmd.reshape((voxels, numtrials)).astype(np.float32, copy=False)
    return betas, numtrials


def _flatten_hrfindex(hrfindexrun: np.ndarray, numruns: int) -> np.ndarray:
    if hrfindexrun.ndim < 2:
        raise ValueError(f"Unexpected HRFindexrun shape: {hrfindexrun.shape}")
    if hrfindexrun.shape[-1] != numruns:
        raise ValueError(
            f"HRFindexrun last dim ({hrfindexrun.shape[-1]}) "
            f"does not match numruns ({numruns})."
        )
    voxels = int(np.prod(hrfindexrun.shape[:-1]))
    return hrfindexrun.reshape((voxels, numruns)).astype(np.int64, copy=False)


def _convolve_by_hrf(design_single: np.ndarray, hrflibrary: np.ndarray) -> list[np.ndarray]:
    num_hrf = hrflibrary.shape[1]
    conv = []
    for h in range(num_hrf):
        conv_h = convolve_design(design_single, hrflibrary[:, h]).astype(np.float32)
        conv.append(conv_h)
    return conv


def _write_task_prediction(betas: np.ndarray, hrf_idx_run: np.ndarray, conv_by_hrf: list[np.ndarray], out_path: Path, chunk_size: int):
    voxels, numtrials = betas.shape
    ntime = conv_by_hrf[0].shape[0]
    pred = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(voxels, ntime))

    for h, conv in enumerate(conv_by_hrf):
        voxel_idx = np.flatnonzero(hrf_idx_run == h)
        if voxel_idx.size == 0:
            continue
        conv_t = conv.T
        for start in range(0, voxel_idx.size, chunk_size):
            chunk = voxel_idx[start : start + chunk_size]
            pred[chunk, :] = betas[chunk, :] @ conv_t

    pred.flush()


def _parse_int_list(values_csv: str, allow_empty: bool, label: str) -> list[int]:
    values = []
    for item in values_csv.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values and not allow_empty:
        raise ValueError(f"No {label} specified.")
    return values


def _is_nifti(path: Path) -> bool:
    if path.suffix == ".nii":
        return True
    suffixes = path.suffixes
    return len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]


def _load_numpy_or_nifti(path: Path) -> np.ndarray:
    if _is_nifti(path):
        img = nib.load(str(path))
        return img.get_fdata(dtype=np.float32)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    raise ValueError(f"Unsupported file type: {path}")


def _load_mask(mask_path, expected_size, expected_shape):
    mask_data = _load_numpy_or_nifti(mask_path)
    mask = np.asarray(mask_data)
    if mask.ndim == 3:
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match expected {expected_shape}."
            )
        mask_flat = mask.reshape(-1).astype(bool)
    elif mask.ndim == 1:
        mask_flat = mask.astype(bool)
    else:
        raise ValueError(f"Mask must be 1D or 3D, got shape {mask.shape}.")
    if expected_size is not None and mask_flat.size != expected_size:
        raise ValueError(
            f"Mask has {mask_flat.size} voxels, expected {expected_size}."
        )
    if not np.any(mask_flat):
        raise ValueError("Mask has no nonzero voxels.")
    return mask_flat


def _load_bold_data(
    bold_path: Path,
    mask_path: Optional[Path],
    expected_timepoints: Optional[int],
    return_meta: bool = False):
    bold_data = _load_numpy_or_nifti(bold_path)
    bold = np.asarray(bold_data)
    volume_shape = None
    mask_flat = None
    if bold.ndim == 2:
        if expected_timepoints is not None:
            if bold.shape[1] != expected_timepoints and bold.shape[0] == expected_timepoints:
                bold = bold.T
                print("Transposed BOLD array to (voxels, timepoints) layout.", flush=True)
        if mask_path is not None:
            mask_data = _load_numpy_or_nifti(mask_path)
            mask = np.asarray(mask_data)
            if mask.ndim == 3:
                volume_shape = mask.shape
            mask_flat = _load_mask(mask_path, expected_size=bold.shape[0], expected_shape=volume_shape)
            bold = bold[mask_flat]
        bold = bold.astype(np.float32, copy=False)
        if return_meta:
            return bold, volume_shape, mask_flat
        return bold
    if bold.ndim == 4:
        volume_shape = bold.shape[:3]
        timepoints = bold.shape[3]
        bold_2d = bold.reshape(-1, timepoints)
        if mask_path is not None:
            mask_flat = _load_mask(
                mask_path, expected_size=bold_2d.shape[0], expected_shape=volume_shape
            )
            bold_2d = bold_2d[mask_flat]
        bold_2d = bold_2d.astype(np.float32, copy=False)
        if return_meta:
            return bold_2d, volume_shape, mask_flat
        return bold_2d
    raise ValueError(f"BOLD data must be 2D or 4D, got shape {bold.shape}.")


def _extract_trial_segments(
    bold_data: np.ndarray,
    trial_len: int,
    num_trials: int,
    rest_after: list[int],
    rest_len: int):
    num_voxels, num_timepoints = bold_data.shape
    segments = np.full((num_voxels, num_trials, trial_len), np.nan, dtype=np.float32)
    start = 0
    rest_after_set = set(rest_after)
    for trial_idx in range(num_trials):
        end = start + trial_len
        if end > num_timepoints:
            raise ValueError(
                f"Not enough timepoints for trial {trial_idx + 1}: need {end}, have {num_timepoints}."
            )
        segments[:, trial_idx, :] = bold_data[:, start:end]
        start = end
        if (trial_idx + 1) in rest_after_set:
            start += rest_len
    if start > num_timepoints:
        raise ValueError(
            f"Rest blocks exceed data length: ended at {start}, but data has {num_timepoints}."
        )
    return segments


def _offset_step(trials: np.ndarray) -> float:
    finite = trials[np.isfinite(trials)]
    if finite.size == 0:
        return 1.0
    p05, p95 = np.percentile(finite, [5, 95])
    span = float(p95 - p05)
    if span <= 0:
        span = float(np.max(finite) - np.min(finite))
    if span <= 0:
        span = 1.0
    return span * 1.2


def _plot_single_voxel_trials(
    trials: np.ndarray,
    voxel_idx: int,
    out_path: Path,
    tr: float):
    num_trials, trial_len = trials.shape
    timepoints = np.arange(1, trial_len + 1, dtype=np.int64)
    step = _offset_step(trials)
    fig_height = max(6.0, num_trials * 0.12)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    for idx in range(num_trials):
        ax.plot(
            timepoints,
            trials[idx] + step * idx,
            color="black",
            linewidth=0.6,
            alpha=0.7,
        )
    ax.set_title(f"Voxel {voxel_idx} trials (n={num_trials})")
    ax.set_xlabel(f"Timepoints (TR={tr:g}s)")
    ax.set_ylabel("Trial (offset)")
    ax.set_xticks(timepoints)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_variance_heatmap(
    variance: np.ndarray,
    voxel_indices: np.ndarray,
    out_path: Path):
    data = variance[voxel_indices, :]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Voxel")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(np.arange(1, data.shape[1] + 1))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(voxel_indices)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Variance across trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _path_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _parse_str_list(values_csv: str) -> list[str]:
    values = []
    for item in values_csv.split(","):
        item = item.strip()
        if item:
            values.append(item)
    return values


def _load_selected_mask(selected_path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    img = nib.load(str(selected_path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Selected map must be 3D, got shape {data.shape}.")
    mask = np.isfinite(data) & (data > 0)
    if not np.any(mask):
        raise ValueError("Selected map has no positive voxels.")
    return mask.reshape(-1), data.shape


def _load_motor_coords(npz_path: Path, include_keys: Optional[list[str]]):
    loaded = np.load(npz_path)
    available_keys = sorted(loaded.files)
    keys = available_keys
    if include_keys:
        missing = [key for key in include_keys if key not in available_keys]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Motor keys not found in {npz_path}: {missing_str}")
        keys = include_keys
    coords_list = []
    key_counts = {}
    for key in keys:
        coords = np.asarray(loaded[key])
        if coords.size == 0:
            key_counts[key] = 0
            continue
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Motor coords for {key} must be (N, 3), got {coords.shape}.")
        coords = coords.astype(np.int64, copy=False)
        coords_list.append(coords)
        key_counts[key] = int(coords.shape[0])
    if not coords_list:
        raise ValueError("Motor region npz has no voxel coordinates.")
    return np.vstack(coords_list), key_counts


def _coords_to_mask(coords: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    mask_flat = np.zeros(int(np.prod(volume_shape)), dtype=bool)
    coords = np.asarray(coords)
    if coords.size == 0:
        return mask_flat
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coords must be (N, 3), got {coords.shape}.")
    coords = coords.astype(np.int64, copy=False)
    if np.any(coords < 0):
        raise ValueError("Coords contain negative indices.")
    if np.any(coords[:, 0] >= volume_shape[0]) or np.any(coords[:, 1] >= volume_shape[1]) or np.any(
        coords[:, 2] >= volume_shape[2]
    ):
        raise ValueError("Coords fall outside the selected map volume.")
    flat_idx = np.ravel_multi_index(coords.T, volume_shape)
    mask_flat[flat_idx] = True
    return mask_flat


def _align_mask_to_bold(mask_flat, bold_mask_flat, bold_voxels):
    if bold_mask_flat is None:
        if mask_flat.size != bold_voxels:
            raise ValueError(f"Mask size ({mask_flat.size}) does not match bold voxels ({bold_voxels}).")
        return mask_flat
    if mask_flat.size != bold_mask_flat.size:
        raise ValueError(f"Mask size ({mask_flat.size}) does not match bold mask ({bold_mask_flat.size}).")
    return mask_flat[bold_mask_flat]


def _cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    n_a = int(sample_a.size)
    n_b = int(sample_b.size)
    if n_a < 2 or n_b < 2:
        return float("nan")
    mean_a = float(np.mean(sample_a))
    mean_b = float(np.mean(sample_b))
    var_a = float(np.var(sample_a, ddof=1))
    var_b = float(np.var(sample_b, ddof=1))
    pooled = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    if pooled <= 0:
        return float("nan")
    return (mean_a - mean_b) / np.sqrt(pooled)


def _bootstrap_mean_diff(sample_a, sample_b, num_boot, rng: np.random.Generator):
    diffs = np.empty(num_boot, dtype=np.float64)
    n_a = sample_a.size
    n_b = sample_b.size
    for idx in range(num_boot):
        boot_a = rng.choice(sample_a, size=n_a, replace=True)
        boot_b = rng.choice(sample_b, size=n_b, replace=True)
        diffs[idx] = float(np.mean(boot_a) - np.mean(boot_b))
    return diffs


def _resample_control_means(control_values, target_size, num_resamples, rng):
    if num_resamples <= 0:
        return np.array([], dtype=np.float64)
    replace = control_values.size < target_size
    means = np.empty(num_resamples, dtype=np.float64)
    for idx in range(num_resamples):
        sample = rng.choice(control_values, size=target_size, replace=replace)
        means[idx] = float(np.mean(sample))
    return means


def _mean_sem(data: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(data, axis=axis)
    count = np.sum(np.isfinite(data), axis=axis)
    std = np.nanstd(data, axis=axis, ddof=1)
    denom = np.sqrt(np.maximum(count, 1))
    sem = std / denom
    if np.isscalar(sem):
        sem = np.array(sem)
    sem = np.where(count > 1, sem, np.nan)
    return mean, sem


def _plot_metric_distributions(selected_metric, control_metric, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.violinplot([selected_metric, control_metric], showmeans=True, showmedians=False, showextrema=False)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Selected", "Control"])
    ax.set_ylabel("Mean trial variance")
    ax.set_title("Per-voxel mean variance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_resample_distribution(resample_means, selected_mean, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(resample_means, bins=30, color="steelblue", alpha=0.7, label="Control resamples")
    ax.axvline(selected_mean, color="crimson", linewidth=2, label="Selected mean")
    ax.set_xlabel("Mean trial variance")
    ax.set_ylabel("Resample count")
    ax.set_title("Control resample means")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_variance_timecourse(selected_variance, control_variance, out_path, tr):
    mean_sel, sem_sel = _mean_sem(selected_variance, axis=0)
    mean_ctrl, sem_ctrl = _mean_sem(control_variance, axis=0)
    timepoints = np.arange(1, mean_sel.shape[0] + 1, dtype=np.int64)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(timepoints, mean_sel, color="crimson", label="Selected")
    ax.fill_between(timepoints, mean_sel - sem_sel, mean_sel + sem_sel, color="crimson", alpha=0.2)
    ax.plot(timepoints, mean_ctrl, color="navy", label="Control")
    ax.fill_between(timepoints, mean_ctrl - sem_ctrl, mean_ctrl + sem_ctrl, color="navy", alpha=0.2)
    ax.set_xlabel(f"Timepoints (TR={tr:g}s)")
    ax.set_ylabel("Variance across trials")
    ax.set_title("Mean trial variance by timepoint")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_selection_analysis(args, bold_data, trial_variance, trials, bold_mask_flat, volume_shape, out_dir):
    if args.selected_map is None:
        return

    selected_path = Path(args.selected_map)
    if not selected_path.is_absolute():
        selected_path = (REPO_ROOT / selected_path).resolve()
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected map not found: {selected_path}")

    if args.motor_voxels_npz is None:
        raise ValueError("Selection analysis requires --motor-voxels-npz.")
    motor_path = Path(args.motor_voxels_npz)
    if not motor_path.is_absolute():
        motor_path = (REPO_ROOT / motor_path).resolve()
    if not motor_path.exists():
        raise FileNotFoundError(f"Motor voxels npz not found: {motor_path}")

    analysis_dir = Path(args.selection_out_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = out_dir / analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)

    analysis_prefix = _path_stem(selected_path)

    selected_mask_flat, selected_shape = _load_selected_mask(selected_path)
    if volume_shape is not None and volume_shape != selected_shape:
        raise ValueError(f"BOLD volume shape {volume_shape} does not match selected map {selected_shape}.")

    motor_keys = _parse_str_list(args.motor_keys) if args.motor_keys else []
    coords, motor_key_counts = _load_motor_coords(motor_path, motor_keys or None)
    motor_mask_flat = _coords_to_mask(coords, selected_shape)

    restrict_to_motor = args.restrict_to_motor
    if restrict_to_motor is None:
        restrict_to_motor = True

    selected_in_motor = selected_mask_flat & motor_mask_flat
    selected_mask_use = selected_in_motor if restrict_to_motor else selected_mask_flat
    control_mask_flat = motor_mask_flat & ~selected_in_motor

    selected_mask = _align_mask_to_bold(selected_mask_use, bold_mask_flat, bold_data.shape[0])
    control_mask = _align_mask_to_bold(control_mask_flat, bold_mask_flat, bold_data.shape[0])

    selected_indices = np.flatnonzero(selected_mask)
    control_indices = np.flatnonzero(control_mask)

    if selected_indices.size == 0:
        raise ValueError("No selected voxels remain after motor restriction/masking.")
    if control_indices.size == 0:
        raise ValueError("No control voxels remain after motor restriction/masking.")

    mean_variance = np.nanmean(trial_variance, axis=1)
    selected_metric = mean_variance[selected_indices]
    control_metric = mean_variance[control_indices]

    selected_finite = np.isfinite(selected_metric)
    control_finite = np.isfinite(control_metric)
    selected_metric = selected_metric[selected_finite]
    control_metric = control_metric[control_finite]
    selected_indices = selected_indices[selected_finite]
    control_indices = control_indices[control_finite]

    if selected_metric.size == 0:
        raise ValueError("Selected voxels have no finite variance metrics.")
    if control_metric.size == 0:
        raise ValueError("Control voxels have no finite variance metrics.")

    mean_selected = float(np.mean(selected_metric))
    mean_control = float(np.mean(control_metric))
    diff_mean = mean_selected - mean_control

    ttest = stats.ttest_ind(selected_metric, control_metric, equal_var=False, nan_policy="omit")
    t_stat = float(ttest.statistic) if np.isfinite(ttest.statistic) else float("nan")
    p_two = float(ttest.pvalue) if np.isfinite(ttest.pvalue) else float("nan")
    if np.isfinite(t_stat) and np.isfinite(p_two):
        p_one = p_two / 2.0 if t_stat < 0 else 1.0 - p_two / 2.0
    else:
        p_one = float("nan")

    effect_d = _cohens_d(selected_metric, control_metric)

    analysis_seed = args.analysis_seed if args.analysis_seed is not None else args.seed
    rng = np.random.default_rng(analysis_seed)

    resample_means = _resample_control_means(control_metric, selected_metric.size, args.control_resamples, rng)
    resample_mean = float(np.mean(resample_means)) if resample_means.size else float("nan")
    resample_std = float(np.std(resample_means, ddof=1)) if resample_means.size > 1 else float("nan")
    if resample_means.size:
        p_resample = (np.sum(resample_means <= mean_selected) + 1) / (resample_means.size + 1)
    else:
        p_resample = float("nan")

    boot_diffs = np.array([], dtype=np.float64)
    ci_low = float("nan")
    ci_high = float("nan")
    if args.bootstrap_samples > 0:
        boot_diffs = _bootstrap_mean_diff(selected_metric, control_metric, args.bootstrap_samples, rng)
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])

    selected_variance = trial_variance[selected_indices]
    control_variance = trial_variance[control_indices]

    metric_plot = analysis_dir / f"{analysis_prefix}_metric_distribution.png"
    _plot_metric_distributions(selected_metric, control_metric, metric_plot)
    if resample_means.size:
        resample_plot = analysis_dir / f"{analysis_prefix}_control_resample_means.png"
        _plot_resample_distribution(resample_means, mean_selected, resample_plot)

    timecourse_plot = analysis_dir / f"{analysis_prefix}_variance_timecourse.png"
    _plot_variance_timecourse(selected_variance, control_variance, timecourse_plot, args.tr)

    if args.selection_variance_voxel_count > 0:
        sel_count = min(args.selection_variance_voxel_count, selected_indices.size)
        ctrl_count = min(args.selection_variance_voxel_count, control_indices.size)
        sel_indices = rng.choice(selected_indices, size=sel_count, replace=False)
        ctrl_indices = rng.choice(control_indices, size=ctrl_count, replace=False)
        selected_heatmap = analysis_dir / f"{analysis_prefix}_variance_heatmap_selected.png"
        control_heatmap = analysis_dir / f"{analysis_prefix}_variance_heatmap_control.png"
        _plot_variance_heatmap(trial_variance, sel_indices, selected_heatmap)
        _plot_variance_heatmap(trial_variance, ctrl_indices, control_heatmap)

    if args.selection_example_voxel_count > 0:
        sel_count = min(args.selection_example_voxel_count, selected_indices.size)
        ctrl_count = min(args.selection_example_voxel_count, control_indices.size)
        sel_indices = rng.choice(selected_indices, size=sel_count, replace=False)
        ctrl_indices = rng.choice(control_indices, size=ctrl_count, replace=False)
        sel_dir = analysis_dir / f"{analysis_prefix}_examples_selected"
        ctrl_dir = analysis_dir / f"{analysis_prefix}_examples_control"
        sel_dir.mkdir(parents=True, exist_ok=True)
        ctrl_dir.mkdir(parents=True, exist_ok=True)
        for voxel_idx in sel_indices:
            out_path = sel_dir / f"voxel_{int(voxel_idx)}_trials.png"
            _plot_single_voxel_trials(trials[voxel_idx], int(voxel_idx), out_path, args.tr)
        for voxel_idx in ctrl_indices:
            out_path = ctrl_dir / f"voxel_{int(voxel_idx)}_trials.png"
            _plot_single_voxel_trials(trials[voxel_idx], int(voxel_idx), out_path, args.tr)

    metrics_path = analysis_dir / f"{analysis_prefix}_selection_metrics.npz"
    np.savez(metrics_path, selected_metric=selected_metric, control_metric=control_metric,
        control_resample_means=resample_means, bootstrap_diffs=boot_diffs, selected_indices=selected_indices, control_indices=control_indices)

    summary = {"selected_map": str(selected_path),
        "motor_voxels_npz": str(motor_path),
        "analysis_prefix": analysis_prefix,
        "restrict_to_motor": bool(restrict_to_motor),
        "motor_keys": motor_keys,
        "motor_key_counts": motor_key_counts,
        "selected_voxels_total": int(np.count_nonzero(selected_mask_flat)),
        "selected_voxels_motor": int(np.count_nonzero(selected_in_motor)),
        "selected_voxels_used": int(np.count_nonzero(selected_mask_use)),
        "motor_voxels_total": int(np.count_nonzero(motor_mask_flat)),
        "control_voxels_total": int(np.count_nonzero(control_mask_flat)),
        "selected_metric_count": int(selected_metric.size),
        "control_metric_count": int(control_metric.size),
        "selected_mean": mean_selected,
        "control_mean": mean_control,
        "mean_diff": diff_mean,
        "t_stat": t_stat,
        "p_two_sided": p_two,
        "p_one_sided": p_one,
        "cohens_d": effect_d,
        "control_resample_mean": resample_mean,
        "control_resample_std": resample_std,
        "control_resample_p_one_sided": p_resample,
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_ci_95": [float(ci_low), float(ci_high)],
        "control_resamples": int(args.control_resamples),
        "analysis_seed": analysis_seed,
    }
    summary_path = analysis_dir / f"{analysis_prefix}_selection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Selection analysis complete: selected={selected_metric.size}, control={control_metric.size}", flush=True)
    print(f"Mean variance: selected={mean_selected:.6f}, control={mean_control:.6f}, diff={diff_mean:.6f}", flush=True)
    print(f"t={t_stat:.3f}, p(one-sided)={p_one:.3g}, d={effect_d:.3f}, resample p={p_resample:.3g}", flush=True)


def _run_trial_viz(args):
    bold_path = Path(args.bold_path)
    if not bold_path.is_absolute():
        bold_path = (REPO_ROOT / bold_path).resolve()
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD data not found: {bold_path}")

    mask_path = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if not mask_path.is_absolute():
            mask_path = (REPO_ROOT / mask_path).resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.trial_len <= 0 or args.num_trials <= 0:
        raise ValueError("trial_len and num_trials must be positive.")
    if args.rest_len < 0:
        raise ValueError("rest_len must be >= 0.")

    rest_after = _parse_int_list(args.rest_after, allow_empty=True, label="rest-after")
    rest_after = sorted(set(rest_after))
    for trial_idx in rest_after:
        if trial_idx <= 0 or trial_idx >= args.num_trials:
            raise ValueError(
                f"rest-after entries must be between 1 and {args.num_trials - 1}, got {trial_idx}."
            )

    expected_timepoints = args.num_trials * args.trial_len + args.rest_len * len(rest_after)
    bold_data, volume_shape, bold_mask_flat = _load_bold_data(bold_path, mask_path, expected_timepoints, return_meta=True)
    num_voxels, num_timepoints = bold_data.shape
    if num_timepoints != expected_timepoints:
        raise ValueError(
            f"BOLD data has {num_timepoints} timepoints, expected {expected_timepoints} "
            f"(trials={args.num_trials}, trial_len={args.trial_len}, "
            f"rest_after={rest_after}, rest_len={args.rest_len})."
        )
    print(f"Loaded BOLD data: {bold_data.shape}", flush=True)

    trials = _extract_trial_segments(
        bold_data,
        trial_len=args.trial_len,
        num_trials=args.num_trials,
        rest_after=rest_after,
        rest_len=args.rest_len,
    )
    print(f"Reshaped trials: {trials.shape}", flush=True)

    trial_variance = np.nanvar(trials, axis=1).astype(np.float32, copy=False)
    variance_path = Path(args.variance_out)
    if not variance_path.is_absolute():
        variance_path = out_dir / variance_path
    np.save(variance_path, trial_variance)
    print(f"Saved trial variance: {variance_path}", flush=True)

    if num_voxels == 0:
        raise ValueError("No voxels available after masking.")

    rng = np.random.default_rng(args.seed)

    if args.single_voxel_count > 0:
        voxel_count = min(args.single_voxel_count, num_voxels)
        voxel_indices = rng.choice(num_voxels, size=voxel_count, replace=False)
        single_voxel_dir = Path(args.single_voxel_dir)
        if not single_voxel_dir.is_absolute():
            single_voxel_dir = out_dir / single_voxel_dir
        single_voxel_dir.mkdir(parents=True, exist_ok=True)
        for voxel_idx in voxel_indices:
            out_path = single_voxel_dir / f"voxel_{voxel_idx}_trials.png"
            _plot_single_voxel_trials(trials[voxel_idx], int(voxel_idx), out_path, args.tr)
        print(f"Saved {voxel_count} single-voxel trial plots in {single_voxel_dir}.", flush=True)

    if args.variance_voxel_count > 0:
        voxel_count = min(args.variance_voxel_count, num_voxels)
        voxel_indices = rng.choice(num_voxels, size=voxel_count, replace=False)
        variance_fig = Path(args.variance_fig)
        if not variance_fig.is_absolute():
            variance_fig = out_dir / variance_fig
        _plot_variance_heatmap(trial_variance, voxel_indices, variance_fig)
        print(f"Saved variance heatmap: {variance_fig}", flush=True)

    _run_selection_analysis(args, bold_data, trial_variance, trials, bold_mask_flat, volume_shape, out_dir)


def main():
    parser = argparse.ArgumentParser(description=( "Compute X_task * beta_task from GLMsingle outputs, "
            "or visualize trial-wise BOLD when --bold-path is set."))
    parser.add_argument("--runs", default="1,2", help="Comma-separated run numbers (1-based).")
    parser.add_argument("--model-path", default="data/sub09_ses01/TYPED_FITHRF_GLMDENOISE_RR.npy")
    parser.add_argument("--design-path", default="data/sub09_ses01/DESIGNINFO.npy")
    parser.add_argument("--out-dir", default="data/sub09_ses01")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Voxel chunk size for matrix multiplies.")
    viz_group = parser.add_argument_group("Trial visualization")
    viz_group.add_argument("--bold-path", type=Path, default= '/scratch/st-mmckeown-1/zkavian/fmri_opt/Thesis_code_Glm_Opt/data/sub09_ses01/sub-pd009_ses-1_run-1_task-mv_bold_corrected_smoothed_reg.nii.gz',
        help="Path to BOLD data (.npy or NIfTI). When set, run trial visualization pipeline.")
    viz_group.add_argument("--mask-path", type=Path,  help="Optional mask (.npy or NIfTI) to select voxels.")
    viz_group.add_argument("--num-trials", type=int, default=90, help="Number of trials (default: 90).")
    viz_group.add_argument("--trial-len", type=int, default=9, help="Timepoints per trial (default: 9).")
    viz_group.add_argument("--rest-after", default="30,60",
        help="Comma-separated trial indices after which rest blocks occur (default: 30,60).")
    viz_group.add_argument("--rest-len", type=int, default=20, help="Rest block length in timepoints.")
    viz_group.add_argument("--tr", type=float, default=1.0, help="TR in seconds (default: 1.0).")
    viz_group.add_argument("--single-voxel-count", type=int, default=10, help="Number of random voxels for per-trial plots (default: 10).")
    viz_group.add_argument("--variance-voxel-count", type=int, default=20,
        help="Number of random voxels for variance heatmap (default: 20).")
    viz_group.add_argument("--variance-out", default="trial_variance.npy",
        help="Output path for variance array (default: trial_variance.npy).")
    viz_group.add_argument("--variance-fig", default="trial_variance_heatmap.png",
        help="Output path for variance heatmap (default: trial_variance_heatmap.png).")
    viz_group.add_argument("--single-voxel-dir", default="single_voxel_trials",
        help="Directory for single-voxel trial plots (default: single_voxel_trials).")
    viz_group.add_argument("--seed", type=int, default=None, help="Random seed for voxel selection (default: random).")
    analysis_group = parser.add_argument_group("Selection analysis")
    analysis_group.add_argument("--selected-map", type=Path, help="Path to voxel_weights_mean_<avg_prefix>_bold_thr95.nii.gz (or similar).")
    analysis_group.add_argument("--motor-voxels-npz", type=Path, help="Path to voxel_weights_mean_<avg_prefix>_motor_region_voxels.npz.")
    analysis_group.add_argument("--motor-keys", default="", help="Comma-separated keys from motor npz to include (default: all).")
    analysis_group.add_argument("--restrict-to-motor", dest="restrict_to_motor", action="store_true", default=None, help="Restrict selected voxels to motor region (default: True when motor npz provided).")
    analysis_group.add_argument("--no-restrict-to-motor", dest="restrict_to_motor", action="store_false", help="Do not restrict selected voxels to motor region.")
    analysis_group.add_argument("--selection-out-dir", default="selection_analysis", help="Output directory for selection analysis plots/metrics.")
    analysis_group.add_argument("--control-resamples", type=int, default=1000, help="Number of control resamples for null distribution (default: 1000).")
    analysis_group.add_argument("--bootstrap-samples", type=int, default=2000, help="Number of bootstrap samples for CI (default: 2000).")
    analysis_group.add_argument("--selection-example-voxel-count", type=int, default=5, help="Example voxels per group for trial plots (default: 5).")
    analysis_group.add_argument("--selection-variance-voxel-count", type=int, default=20, help="Voxels per group for variance heatmaps (default: 20).")
    analysis_group.add_argument("--analysis-seed", type=int, default=None, help="Random seed for selection analysis resampling (default: --seed).")
    args = parser.parse_args()

    if args.bold_path is not None:
        _run_trial_viz(args)
        return

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()
    design_path = Path(args.design_path)
    if not design_path.is_absolute():
        design_path = (REPO_ROOT / design_path).resolve()

    design_single_list, hrflibrary = _load_design(design_path)
    betasmd, hrfindexrun = _load_model(model_path)

    betas, numtrials = _flatten_betas(betasmd)
    numruns = len(design_single_list)
    hrfindex_flat = _flatten_hrfindex(hrfindexrun, numruns)

    if hrflibrary.ndim != 2:
        raise ValueError(f"Unexpected hrflibrary shape: {hrflibrary.shape}")

    if design_single_list[0].shape[1] != numtrials:
        raise ValueError(
            f"Design has {design_single_list[0].shape[1]} trials, "
            f"but betasmd has {numtrials} trials."
        )

    runs = _parse_runs(args.runs)
    for run in runs:
        run_idx = run - 1
        if run_idx < 0 or run_idx >= numruns:
            raise ValueError(f"Run {run} is out of range for {numruns} runs.")

        design_single = design_single_list[run_idx]
        conv_by_hrf = _convolve_by_hrf(design_single, hrflibrary)

        out_path = out_dir / f"Xtask_beta_task_run{run}.npy"
        _write_task_prediction(
            betas,
            hrfindex_flat[:, run_idx],
            conv_by_hrf,
            out_path,
            args.chunk_size,
        )
        print(f"Saved task prediction: {out_path}")


if __name__ == "__main__":
    main()
