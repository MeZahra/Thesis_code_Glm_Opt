#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import nibabel as nib
from scipy import stats
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

REPO_ROOT = Path(__file__).resolve().parent
GLMSINGLE_ROOT = REPO_ROOT / "GLMsingle"
if str(GLMSINGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(GLMSINGLE_ROOT))

from glmsingle.design.convolve_design import convolve_design

SMALL_N_THRESHOLD = 20
BALANCED_CI_LEVEL = 0.95

_SUBJECT_ID_RE = re.compile(r"sub(\d)(?!\d)")


def _parse_runs(runs_csv):
    """Parse comma-separated run numbers into a list of ints."""
    runs = []
    for item in runs_csv.split(","):
        item = item.strip()
        if not item:
            continue
        runs.append(int(item))
    return runs


def _resolve_path(path):
    """Resolve a path relative to the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _pad_subject_id(path_str):
    """Zero-pad single-digit subject IDs in paths (sub9 -> sub09)."""
    return _SUBJECT_ID_RE.sub(r"sub0\1", path_str)


def _load_design(design_path):
    """Load GLMsingle design matrix and HRF library."""
    designinfo = np.load(design_path, allow_pickle=True).item()
    params = designinfo["params"]
    return designinfo["designSINGLE"], params["hrflibrary"]


def _load_model(model_path):
    """Load GLMsingle model betas and HRF indices."""
    model = np.load(model_path, allow_pickle=True).item()
    return model["betasmd"], model["HRFindexrun"]


def _flatten_betas(betasmd):
    """Reshape beta volume to (voxels, trials) and return volume shape."""
    if betasmd.ndim < 2:
        raise ValueError(f"betasmd must be at least 2D, got shape {betasmd.shape}.")
    numtrials = betasmd.shape[-1]
    if betasmd.ndim == 2:
        voxels = betasmd.shape[0]
        volume_shape = (voxels,)
        betas = betasmd.astype(np.float32, copy=False)
        return betas, numtrials, volume_shape
    volume_shape = betasmd.shape[:-1]
    voxels = int(np.prod(volume_shape))
    betas = betasmd.reshape((voxels, numtrials)).astype(np.float32, copy=False)
    return betas, numtrials, volume_shape


def _flatten_hrfindex(hrfindexrun, numruns, voxels_expected):
    """Flatten HRF index array to (voxels, runs)."""
    if hrfindexrun.ndim == 2:
        if hrfindexrun.shape != (voxels_expected, numruns):
            raise ValueError(
                f"HRF index shape {hrfindexrun.shape} does not match ({voxels_expected}, {numruns})."
            )
        return hrfindexrun.astype(np.int64, copy=False)
    voxels = int(np.prod(hrfindexrun.shape[:-1]))
    if voxels != voxels_expected:
        raise ValueError(
            f"HRF index voxel count {voxels} does not match betas {voxels_expected}."
        )
    return hrfindexrun.reshape((voxels, numruns)).astype(np.int64, copy=False)


def _convolve_by_hrf(design_single, hrflibrary):
    """Convolve the design matrix with each HRF in the library."""
    num_hrf = hrflibrary.shape[1]
    conv = []
    for h in range(num_hrf):
        conv_h = convolve_design(design_single, hrflibrary[:, h]).astype(np.float32)
        conv.append(conv_h)
    return conv


def _normalize_hrf_indices(hrf_idx_run, num_hrf):
    """Normalize HRF indices to 0-based when they look 1-based."""
    hrf_idx_run = np.asarray(hrf_idx_run)
    if hrf_idx_run.size == 0:
        return hrf_idx_run
    finite = hrf_idx_run[np.isfinite(hrf_idx_run)]
    if finite.size == 0:
        return hrf_idx_run
    min_idx = int(np.min(finite))
    max_idx = int(np.max(finite))
    if min_idx >= 1 and max_idx == num_hrf:
        return hrf_idx_run - 1
    return hrf_idx_run


def _predict_run_timeseries(betas_run, hrf_idx_run, design_single, hrflibrary, chunk_size):
    """Predict a run's BOLD time series for all voxels."""
    if betas_run.shape[1] != design_single.shape[1]:
        raise ValueError(f"Betas trials {betas_run.shape[1]} do not match design trials {design_single.shape[1]}.")
    conv_by_hrf = _convolve_by_hrf(design_single, hrflibrary)
    voxels = betas_run.shape[0]
    timepoints = conv_by_hrf[0].shape[0]
    pred = np.zeros((voxels, timepoints), dtype=np.float32)

    num_hrf = hrflibrary.shape[1]
    hrf_idx_run = _normalize_hrf_indices(hrf_idx_run, num_hrf)
    invalid = (hrf_idx_run < 0) | (hrf_idx_run >= num_hrf)
    if np.any(invalid):
        invalid_count = int(np.count_nonzero(invalid))
        print(f"Warning: {invalid_count} voxels have invalid HRF indices for this run.", flush=True)

    for h, conv in enumerate(conv_by_hrf):
        voxel_idx = np.flatnonzero(hrf_idx_run == h)
        if voxel_idx.size == 0:
            continue
        conv_t = conv.T
        for start in range(0, voxel_idx.size, chunk_size):
            chunk = voxel_idx[start : start + chunk_size]
            pred[chunk, :] = betas_run[chunk, :] @ conv_t
    return pred


def _trial_onsets_from_design(design_single, eps=1e-6):
    """Get per-trial onset indices from a design matrix."""
    design = np.asarray(design_single)
    if design.ndim != 2:
        raise ValueError(f"Design matrix must be 2D, got shape {design.shape}.")
    num_trials = design.shape[1]
    onsets = np.empty(num_trials, dtype=np.int64)
    missing = 0
    for idx in range(num_trials):
        nz = np.flatnonzero(np.abs(design[:, idx]) > eps)
        if nz.size == 0:
            onsets[idx] = -1
            missing += 1
        else:
            onsets[idx] = int(nz[0])
    if missing:
        print(f"Warning: {missing} trials have no nonzero onset in design.", flush=True)
    return onsets


def _accumulate_trial_stats(sum_vals, sumsq_vals, count_vals, segment):
    """Accumulate sums for variance estimation with NaN handling."""
    finite = np.isfinite(segment)
    if not finite.all():
        segment = np.where(finite, segment, 0.0)
    sum_vals += segment
    sumsq_vals += segment * segment
    count_vals += finite.astype(np.int64)


def _compute_trial_variance(sum_vals, sumsq_vals, count_vals):
    """Compute variance across trials from running sums."""
    variance = np.full(sum_vals.shape, np.nan, dtype=np.float32)
    valid = count_vals > 1
    if np.any(valid):
        numerator = sumsq_vals - (sum_vals ** 2) / np.maximum(count_vals, 1)
        variance[valid] = (numerator[valid] / (count_vals[valid] - 1)).astype(np.float32, copy=False)
    return variance


def _first_glob(parent, pattern):
    for path in sorted(parent.glob(pattern)):
        if path.is_file():
            return path
    return None


def _load_mask_indices_file(path):
    data = np.load(path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object:
        data = tuple(data.tolist())
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == 3:
        axes = tuple(data)
    elif isinstance(data, (tuple, list)) and len(data) == 3:
        axes = tuple(data)
    else:
        raise ValueError(f"Mask indices must be a 3-array tuple or (3, N) array, got {getattr(data, 'shape', None)}.")
    return tuple(np.asarray(axis, dtype=np.int64) for axis in axes)


def _mask_indices_from_masks(brain_path, csf_path, gray_path, target_voxels=None, gray_threshold=None):
    brain = nib.load(brain_path).get_fdata(dtype=np.float32) > 0
    csf = nib.load(csf_path).get_fdata(dtype=np.float32) > 0
    gray = nib.load(gray_path).get_fdata(dtype=np.float32)
    if brain.shape != csf.shape or brain.shape != gray.shape:
        raise ValueError(
            f"Mask volumes must match shapes; brain={brain.shape}, csf={csf.shape}, gray={gray.shape}."
        )
    base = brain & (~csf)
    gray_base = gray[base]
    if gray_threshold is None:
        candidates = (0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0)
        for cand in candidates:
            count = int(np.count_nonzero(gray_base > cand))
            if target_voxels is None or count == target_voxels:
                gray_threshold = float(cand)
                break
        if gray_threshold is None:
            raise ValueError(
                f"Could not infer gray threshold to match {target_voxels} voxels; "
                "provide --mask-gray-threshold or --mask-indices."
            )
    else:
        count = int(np.count_nonzero(gray_base > gray_threshold))
        if target_voxels is not None and count != target_voxels:
            raise ValueError(
                f"Gray threshold {gray_threshold} yields {count} voxels, expected {target_voxels}."
            )
    mask = base & (gray > gray_threshold)
    count = int(np.count_nonzero(mask))
    if target_voxels is not None and count != target_voxels:
        raise ValueError(f"Mask voxel count {count} does not match expected {target_voxels}.")
    return tuple(np.where(mask)), brain.shape, gray_threshold


def _coords_to_mask_indices(coords, mask_indices, mask_shape):
    if mask_shape is None:
        raise ValueError("mask_shape is required to map coords to mask indices.")
    coords = np.asarray(coords, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Coords must be (N, 3), got shape {coords.shape}.")
    mask_flat = np.ravel_multi_index(mask_indices, mask_shape)
    coords_flat = np.ravel_multi_index(coords.T, mask_shape)
    if mask_flat.size == 0:
        raise ValueError("Mask indices are empty.")
    if np.all(mask_flat[:-1] <= mask_flat[1:]):
        order = None
        mask_sorted = mask_flat
    else:
        order = np.argsort(mask_flat)
        mask_sorted = mask_flat[order]
    pos = np.searchsorted(mask_sorted, coords_flat)
    valid = pos < mask_sorted.size
    matched = np.zeros(coords_flat.shape, dtype=bool)
    if np.any(valid):
        matched[valid] = mask_sorted[pos[valid]] == coords_flat[valid]
    if not np.all(matched):
        missing = int(np.count_nonzero(~matched))
        raise ValueError(f"{missing} coordinates are not present in the mask index map.")
    if order is None:
        return pos
    return order[pos]


def _load_indices_npz(npz_path, volume_shape, mask_indices=None, mask_shape=None):
    """Load voxel indices from an npz file (coords or flat indices)."""
    loaded = np.load(npz_path)
    key = None
    for candidate in ("indices", "coords", "voxel_indices"):
        if candidate in loaded.files:
            key = candidate
            break
    if key is None:
        if len(loaded.files) == 1:
            key = loaded.files[0]
        else:
            raise ValueError(f"Could not find indices array in {npz_path}; keys={loaded.files}.")
    indices = np.asarray(loaded[key])
    if indices.ndim == 2 and indices.shape[1] == 3:
        if len(volume_shape) != 3:
            if mask_indices is None:
                raise ValueError(f"Cannot map 3D coords to flat indices with volume shape {volume_shape}.")
            flat = _coords_to_mask_indices(indices, mask_indices, mask_shape)
        else:
            try:
                flat = np.ravel_multi_index(indices.T, volume_shape)
            except ValueError:
                if mask_indices is None:
                    raise
                flat = _coords_to_mask_indices(indices, mask_indices, mask_shape)
    elif indices.ndim == 1:
        flat = indices.astype(np.int64, copy=False)
    else:
        raise ValueError(f"Indices must be (N, 3) coords or flat 1D, got shape {indices.shape}.")
    return np.unique(flat)


def _validate_indices(indices, voxels, label):
    """Validate indices against voxel count."""
    if indices.size == 0:
        raise ValueError(f"{label} indices are empty.")
    if np.min(indices) < 0 or np.max(indices) >= voxels:
        raise ValueError(f"{label} indices are out of bounds for {voxels} voxels.")
    return indices


def _cohens_d(sample_a, sample_b):
    """Compute Cohen's d effect size between two samples."""
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


def _hedges_g(sample_a, sample_b):
    """Compute Hedges' g (small-sample corrected) effect size."""
    d = _cohens_d(sample_a, sample_b)
    if not np.isfinite(d):
        return float("nan")
    n_a = int(sample_a.size)
    n_b = int(sample_b.size)
    denom = 4 * (n_a + n_b) - 9
    if denom <= 0:
        return float("nan")
    correction = 1.0 - 3.0 / denom
    return d * correction


def _auc_from_samples(sample_a, sample_b):
    """Compute AUC = P(A > B) + 0.5 * P(A == B) via rank sums."""
    n_a = int(sample_a.size)
    n_b = int(sample_b.size)
    if n_a == 0 or n_b == 0:
        return float("nan")
    combined = np.concatenate([sample_a, sample_b])
    ranks = stats.rankdata(combined, method="average")
    rank_a = float(np.sum(ranks[:n_a]))
    u_stat = rank_a - n_a * (n_a + 1) / 2.0
    return u_stat / (n_a * n_b)


def _summarize_distribution(dist, ci_level=BALANCED_CI_LEVEL):
    """Summarize a distribution with median and percentile CI."""
    dist = np.asarray(dist, dtype=np.float64)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0:
        return {
            "median": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n": 0,
        }
    alpha = 1.0 - ci_level
    lower = np.percentile(dist, 100 * alpha / 2.0)
    upper = np.percentile(dist, 100 * (1.0 - alpha / 2.0))
    return {
        "median": float(np.median(dist)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n": int(dist.size),
    }


def _balanced_resample_stats(
    selected_vals,
    nonselected_vals,
    rng,
    num_resamples,
    small_n_threshold=SMALL_N_THRESHOLD,
    ci_level=BALANCED_CI_LEVEL,
):
    """Balanced resampling with fixed n to reduce sample-size sensitivity."""
    n_selected = int(selected_vals.size)
    n_nonselected = int(nonselected_vals.size)
    n_balanced = int(min(n_selected, n_nonselected))
    effect_label = "hedges_g" if n_selected < small_n_threshold else "cohens_d"

    summary = {
        "balanced_n": n_balanced,
        "balanced_num_resamples": int(num_resamples),
        "balanced_effect_size_label": effect_label,
        "balanced_ci_level": float(ci_level),
        "balanced_small_n_threshold": int(small_n_threshold),
    }

    if n_balanced < 2 or num_resamples <= 0:
        nan_fields = [
            "balanced_mean_diff_median",
            "balanced_mean_diff_ci_lower",
            "balanced_mean_diff_ci_upper",
            "balanced_median_diff_median",
            "balanced_median_diff_ci_lower",
            "balanced_median_diff_ci_upper",
            "balanced_cohens_d_median",
            "balanced_cohens_d_ci_lower",
            "balanced_cohens_d_ci_upper",
            "balanced_hedges_g_median",
            "balanced_hedges_g_ci_lower",
            "balanced_hedges_g_ci_upper",
            "balanced_effect_size_median",
            "balanced_effect_size_ci_lower",
            "balanced_effect_size_ci_upper",
            "balanced_cliff_delta_median",
            "balanced_cliff_delta_ci_lower",
            "balanced_cliff_delta_ci_upper",
            "balanced_auc_selected_lower_median",
            "balanced_auc_selected_lower_ci_lower",
            "balanced_auc_selected_lower_ci_upper",
        ]
        for key in nan_fields:
            summary[key] = float("nan")
        empty = np.array([], dtype=np.float64)
        distributions = {
            "mean_diff": empty,
            "median_diff": empty,
            "cohens_d": empty,
            "hedges_g": empty,
            "cliff_delta": empty,
            "auc_selected_lower": empty,
        }
        return summary, distributions

    mean_diffs = np.empty(num_resamples, dtype=np.float64)
    median_diffs = np.empty(num_resamples, dtype=np.float64)
    cohens_ds = np.empty(num_resamples, dtype=np.float64)
    hedges_gs = np.empty(num_resamples, dtype=np.float64)
    cliff_deltas = np.empty(num_resamples, dtype=np.float64)
    auc_lowers = np.empty(num_resamples, dtype=np.float64)

    for idx in range(num_resamples):
        sel_sample = rng.choice(selected_vals, size=n_balanced, replace=True)
        nonsel_sample = rng.choice(nonselected_vals, size=n_balanced, replace=True)
        mean_diffs[idx] = np.mean(sel_sample) - np.mean(nonsel_sample)
        median_diffs[idx] = np.median(sel_sample) - np.median(nonsel_sample)
        cohens_ds[idx] = _cohens_d(sel_sample, nonsel_sample)
        hedges_gs[idx] = _hedges_g(sel_sample, nonsel_sample)
        auc_high = _auc_from_samples(sel_sample, nonsel_sample)
        auc_lowers[idx] = 1.0 - auc_high
        cliff_deltas[idx] = 2.0 * auc_high - 1.0

    mean_summary = _summarize_distribution(mean_diffs, ci_level=ci_level)
    median_summary = _summarize_distribution(median_diffs, ci_level=ci_level)
    cohens_summary = _summarize_distribution(cohens_ds, ci_level=ci_level)
    hedges_summary = _summarize_distribution(hedges_gs, ci_level=ci_level)
    cliff_summary = _summarize_distribution(cliff_deltas, ci_level=ci_level)
    auc_summary = _summarize_distribution(auc_lowers, ci_level=ci_level)

    if effect_label == "hedges_g":
        effect_summary = hedges_summary
    else:
        effect_summary = cohens_summary

    summary.update({
        "balanced_mean_diff_median": mean_summary["median"],
        "balanced_mean_diff_ci_lower": mean_summary["ci_lower"],
        "balanced_mean_diff_ci_upper": mean_summary["ci_upper"],
        "balanced_median_diff_median": median_summary["median"],
        "balanced_median_diff_ci_lower": median_summary["ci_lower"],
        "balanced_median_diff_ci_upper": median_summary["ci_upper"],
        "balanced_cohens_d_median": cohens_summary["median"],
        "balanced_cohens_d_ci_lower": cohens_summary["ci_lower"],
        "balanced_cohens_d_ci_upper": cohens_summary["ci_upper"],
        "balanced_hedges_g_median": hedges_summary["median"],
        "balanced_hedges_g_ci_lower": hedges_summary["ci_lower"],
        "balanced_hedges_g_ci_upper": hedges_summary["ci_upper"],
        "balanced_effect_size_median": effect_summary["median"],
        "balanced_effect_size_ci_lower": effect_summary["ci_lower"],
        "balanced_effect_size_ci_upper": effect_summary["ci_upper"],
        "balanced_cliff_delta_median": cliff_summary["median"],
        "balanced_cliff_delta_ci_lower": cliff_summary["ci_lower"],
        "balanced_cliff_delta_ci_upper": cliff_summary["ci_upper"],
        "balanced_auc_selected_lower_median": auc_summary["median"],
        "balanced_auc_selected_lower_ci_lower": auc_summary["ci_lower"],
        "balanced_auc_selected_lower_ci_upper": auc_summary["ci_upper"],
    })

    distributions = {
        "mean_diff": mean_diffs,
        "median_diff": median_diffs,
        "cohens_d": cohens_ds,
        "hedges_g": hedges_gs,
        "cliff_delta": cliff_deltas,
        "auc_selected_lower": auc_lowers,
    }

    return summary, distributions

def _resample_control_means(control_values, target_size, num_resamples, rng):
    """Resample control means for a null distribution."""
    if num_resamples <= 0:
        return np.array([], dtype=np.float64)
    # Always sample with replacement to keep the null distribution consistent.
    replace = True
    means = np.empty(num_resamples, dtype=np.float64)
    for idx in range(num_resamples):
        sample = rng.choice(control_values, size=target_size, replace=replace)
        means[idx] = float(np.mean(sample))
    return means


def _size_matched_resample_summary(
    selected_vals,
    nonselected_vals,
    rng,
    num_resamples,
    ci_level=BALANCED_CI_LEVEL,
):
    """Size-matched resampling summary using mean variance for non-selected voxels."""
    n_selected = int(selected_vals.size)
    n_nonselected = int(nonselected_vals.size)
    observed_selected_mean = float(np.mean(selected_vals)) if n_selected else float("nan")

    resampled_means = _resample_control_means(nonselected_vals, n_selected, num_resamples, rng)
    resample_summary = _summarize_distribution(resampled_means, ci_level=ci_level)

    if resampled_means.size:
        fraction_greater = float(np.mean(resampled_means > observed_selected_mean))
        percentile = float(stats.percentileofscore(resampled_means, observed_selected_mean, kind="mean"))
        resampled_mean = float(np.mean(resampled_means))
    else:
        fraction_greater = float("nan")
        percentile = float("nan")
        resampled_mean = float("nan")

    summary = {
        "selected_count": n_selected,
        "nonselected_count": n_nonselected,
        "resample_size": n_selected,
        "num_resamples": int(num_resamples),
        "resample_ci_level": float(ci_level),
        "observed_selected_mean": observed_selected_mean,
        "nonselected_resample_mean_mean": resampled_mean,
        "nonselected_resample_mean_median": resample_summary["median"],
        "nonselected_resample_mean_ci_lower": resample_summary["ci_lower"],
        "nonselected_resample_mean_ci_upper": resample_summary["ci_upper"],
        "fraction_nonselected_mean_greater": fraction_greater,
        "percentile_selected_in_nonselected": percentile,
        "resample_with_replacement": bool(n_nonselected < n_selected),
    }

    return summary, resampled_means


def _compare_groups(selected_vals, motor_vals, rng, num_resamples):
    """Compute summary statistics with balanced resampling for uncertainty."""
    summary = {
        "selected_count": int(selected_vals.size),
        "motor_count": int(motor_vals.size),
        "selected_mean": float(np.mean(selected_vals)),
        "motor_mean": float(np.mean(motor_vals)),
        "selected_median": float(np.median(selected_vals)),
        "motor_median": float(np.median(motor_vals)),
        "selected_std": float(np.std(selected_vals, ddof=1)) if selected_vals.size > 1 else float("nan"),
        "motor_std": float(np.std(motor_vals, ddof=1)) if motor_vals.size > 1 else float("nan"),
    }
    summary["mean_diff"] = summary["selected_mean"] - summary["motor_mean"]
    summary["mean_diff_pct"] = (
        100 * summary["mean_diff"] / summary["motor_mean"] if summary["motor_mean"] != 0 else float("nan")
    )

    cohens_d = _cohens_d(selected_vals, motor_vals)
    hedges_g = _hedges_g(selected_vals, motor_vals)
    effect_label = "hedges_g" if selected_vals.size < SMALL_N_THRESHOLD else "cohens_d"
    effect_value = hedges_g if effect_label == "hedges_g" else cohens_d

    auc_high = _auc_from_samples(selected_vals, motor_vals)
    auc_selected_lower = 1.0 - auc_high
    cliff_delta = 2.0 * auc_high - 1.0

    balanced_summary, balanced_dists = _balanced_resample_stats(
        selected_vals, motor_vals, rng, num_resamples, small_n_threshold=SMALL_N_THRESHOLD
    )

    summary.update({
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "effect_size_label": effect_label,
        "effect_size_value": float(effect_value),
        "auc_selected_lower": float(auc_selected_lower),
        "cliff_delta": float(cliff_delta),
    })
    summary.update(balanced_summary)

    motor_std = summary["motor_std"]
    if np.isfinite(motor_std) and motor_std > 0:
        summary["selected_mean_z"] = (summary["selected_mean"] - summary["motor_mean"]) / motor_std
    else:
        summary["selected_mean_z"] = float("nan")

    return summary, balanced_dists


def _compute_motor_cortex_comparison(voxel_var_flat, selected_indices, motor_indices):
    """
    Compute variability comparison within motor cortex only.

    This is the KEY comparison: selected voxels WITHIN motor cortex vs
    non-selected voxels WITHIN motor cortex.
    """
    selected_set = set(selected_indices.tolist())
    motor_set = set(motor_indices.tolist())

    # Voxels that are BOTH selected AND in motor cortex
    selected_motor = np.array(sorted(selected_set & motor_set), dtype=np.int64)
    # Voxels that are in motor cortex but NOT selected
    nonselected_motor = np.array(sorted(motor_set - selected_set), dtype=np.int64)

    # Get variability values
    selected_motor_vals = voxel_var_flat[selected_motor]
    nonselected_motor_vals = voxel_var_flat[nonselected_motor]

    # Filter to finite values
    selected_motor_vals = selected_motor_vals[np.isfinite(selected_motor_vals)]
    nonselected_motor_vals = nonselected_motor_vals[np.isfinite(nonselected_motor_vals)]

    return selected_motor_vals, nonselected_motor_vals, selected_motor, nonselected_motor


def _permutation_test(selected_vals, nonselected_vals, num_permutations, rng, statistic="mean"):
    """
    Permutation test: randomly sample from combined pool to build null distribution.

    Tests whether the selected voxels have lower variability than expected by chance.
    """
    n_selected = selected_vals.size
    n_nonselected = nonselected_vals.size
    combined = np.concatenate([selected_vals, nonselected_vals])

    if statistic == "mean":
        observed_diff = np.mean(selected_vals) - np.mean(nonselected_vals)
        observed_selected = np.mean(selected_vals)
    elif statistic == "median":
        observed_diff = np.median(selected_vals) - np.median(nonselected_vals)
        observed_selected = np.median(selected_vals)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    null_diffs = np.empty(num_permutations, dtype=np.float64)
    null_selected_means = np.empty(num_permutations, dtype=np.float64)

    for i in range(num_permutations):
        perm = rng.permutation(combined)
        perm_selected = perm[:n_selected]
        perm_nonselected = perm[n_selected:]
        if statistic == "mean":
            null_diffs[i] = np.mean(perm_selected) - np.mean(perm_nonselected)
            null_selected_means[i] = np.mean(perm_selected)
        else:
            null_diffs[i] = np.median(perm_selected) - np.median(perm_nonselected)
            null_selected_means[i] = np.median(perm_selected)

    # One-sided p-value: probability of observing a difference as extreme or more
    # (in the direction of selected being lower)
    p_value = (np.sum(null_diffs <= observed_diff) + 1) / (num_permutations + 1)

    # Percentile rank of observed selected mean in null distribution
    percentile = float(stats.percentileofscore(null_selected_means, observed_selected, kind="mean"))

    return {
        "observed_diff": float(observed_diff),
        "observed_selected_stat": float(observed_selected),
        "p_value_one_sided": float(p_value),
        "percentile_rank": percentile,
        "null_diffs": null_diffs,
        "null_selected_means": null_selected_means,
    }


def _bootstrap_ci(values, num_bootstrap, rng, ci_level=0.95, statistic="mean"):
    """Compute bootstrap confidence interval for mean or median."""
    n = values.size
    bootstrap_stats = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        if statistic == "mean":
            bootstrap_stats[i] = np.mean(sample)
        else:
            bootstrap_stats[i] = np.median(sample)

    alpha = 1 - ci_level
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return float(lower), float(upper), bootstrap_stats


def _compare_motor_groups(selected_vals, nonselected_vals, rng, num_resamples):
    """Comprehensive comparison with balanced resampling summaries."""
    summary = {
        "selected_motor_count": int(selected_vals.size),
        "nonselected_motor_count": int(nonselected_vals.size),
        "selected_mean": float(np.mean(selected_vals)),
        "nonselected_mean": float(np.mean(nonselected_vals)),
        "selected_median": float(np.median(selected_vals)),
        "nonselected_median": float(np.median(nonselected_vals)),
        "selected_std": float(np.std(selected_vals, ddof=1)) if selected_vals.size > 1 else float("nan"),
        "nonselected_std": float(np.std(nonselected_vals, ddof=1)) if nonselected_vals.size > 1 else float("nan"),
    }

    summary["mean_diff"] = summary["selected_mean"] - summary["nonselected_mean"]
    summary["mean_diff_pct"] = (
        100 * summary["mean_diff"] / summary["nonselected_mean"]
        if summary["nonselected_mean"] != 0
        else float("nan")
    )

    cohens_d = _cohens_d(selected_vals, nonselected_vals)
    hedges_g = _hedges_g(selected_vals, nonselected_vals)
    effect_label = "hedges_g" if selected_vals.size < SMALL_N_THRESHOLD else "cohens_d"
    effect_value = hedges_g if effect_label == "hedges_g" else cohens_d

    auc_high = _auc_from_samples(selected_vals, nonselected_vals)
    auc_selected_lower = 1.0 - auc_high
    cliff_delta = 2.0 * auc_high - 1.0

    balanced_summary, balanced_dists = _balanced_resample_stats(
        selected_vals, nonselected_vals, rng, num_resamples, small_n_threshold=SMALL_N_THRESHOLD
    )

    summary.update({
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "effect_size_label": effect_label,
        "effect_size_value": float(effect_value),
        "auc_selected_lower": float(auc_selected_lower),
        "cliff_delta": float(cliff_delta),
    })
    summary.update(balanced_summary)

    return summary, balanced_dists


def _plot_comprehensive_figure(
    selected_vals, nonselected_vals, summary, resampled_means, out_path
):
    """
    Create a publication-quality multi-panel figure demonstrating voxel selection validity.

    Panel A: KDE density comparison
    Panel B: Size-matched resampling distribution (non-selected)
    Panel C: Summary statistics box
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25, height_ratios=[1.2, 1])

    # Color scheme
    sel_color = "#E63946"  # Red for selected
    nonsel_color = "#457B9D"  # Blue for non-selected
    null_color = "#A8DADC"  # Light blue for null distribution

    selected_mean = summary.get("selected_mean_variance", summary.get("selected_mean", float("nan")))
    nonselected_mean = summary.get("nonselected_mean_variance", summary.get("nonselected_mean", float("nan")))

    # ===== Panel A: KDE Density Comparison =====
    ax_kde = fig.add_subplot(gs[0, 0])

    # Compute KDE with robust bandwidth
    combined = np.concatenate([selected_vals, nonselected_vals])
    # Use percentile-based range to avoid outlier influence
    xmin = np.percentile(combined, 1)
    xmax = np.percentile(combined, 99)
    pad = (xmax - xmin) * 0.1
    xmin = max(0, xmin - pad)  # Variance/std must be >= 0
    xmax = xmax + pad

    grid = np.linspace(xmin, xmax, 500)

    # Compute KDEs
    sel_kde = stats.gaussian_kde(selected_vals, bw_method="scott")
    nonsel_kde = stats.gaussian_kde(nonselected_vals, bw_method="scott")

    ax_kde.fill_between(grid, sel_kde(grid), alpha=0.4, color=sel_color, label=f"Selected (n={selected_vals.size})")
    ax_kde.fill_between(grid, nonsel_kde(grid), alpha=0.4, color=nonsel_color, label=f"Non-selected (n={nonselected_vals.size})")
    ax_kde.plot(grid, sel_kde(grid), color=sel_color, linewidth=2)
    ax_kde.plot(grid, nonsel_kde(grid), color=nonsel_color, linewidth=2)

    # Add mean lines
    ax_kde.axvline(selected_mean, color=sel_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax_kde.axvline(nonselected_mean, color=nonsel_color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax_kde.set_xlabel("Trial-to-Trial Variance", fontsize=11)
    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.set_title("A. Variability Distributions in Motor Cortex", fontsize=12, fontweight="bold")
    ax_kde.legend(loc="upper right", fontsize=9)
    ax_kde.set_xlim(xmin, xmax)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    # ===== Panel B: Size-matched Resampling Distribution =====
    ax_perm = fig.add_subplot(gs[0, 1])

    if resampled_means.size:
        ax_perm.hist(resampled_means, bins=50, color=null_color, edgecolor="white", alpha=0.8, density=True)
        median = summary["nonselected_resample_mean_median"]
        ci_low = summary["nonselected_resample_mean_ci_lower"]
        ci_high = summary["nonselected_resample_mean_ci_upper"]
        observed = summary["observed_selected_mean"]
        ax_perm.axvline(observed, color=sel_color, linewidth=2.5, linestyle="--", label=f"Selected mean = {observed:.4f}")
        ax_perm.axvline(median, color="gray", linewidth=1.5, linestyle="-", label="Resample median")
        ax_perm.axvline(ci_low, color="gray", linewidth=1.5, linestyle="--")
        ax_perm.axvline(ci_high, color="gray", linewidth=1.5, linestyle="--", label="95% CI")
        textstr = f"n={summary['resample_size']}\nB={summary['num_resamples']}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
        ax_perm.text(0.95, 0.95, textstr, transform=ax_perm.transAxes, fontsize=10,
                     verticalalignment="top", horizontalalignment="right", bbox=props)
    else:
        ax_perm.text(0.5, 0.5, "No resamples", transform=ax_perm.transAxes,
                     fontsize=11, horizontalalignment="center", verticalalignment="center")

    ax_perm.set_xlabel("Mean Variance (non-selected resamples)", fontsize=11)
    ax_perm.set_ylabel("Density", fontsize=11)
    ax_perm.set_title("B. Size-matched Resampling Distribution", fontsize=12, fontweight="bold")
    ax_perm.legend(loc="upper left", fontsize=9)
    ax_perm.spines["top"].set_visible(False)
    ax_perm.spines["right"].set_visible(False)

    # ===== Panel C: Summary Statistics (spans bottom row) =====
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis("off")

    # Create formatted statistics table
    selected_mean = summary.get("selected_mean_variance", summary.get("selected_mean", float("nan")))
    nonselected_mean = summary.get("nonselected_mean_variance", summary.get("nonselected_mean", float("nan")))
    selected_median = summary.get("selected_median_variance", summary.get("selected_median", float("nan")))
    nonselected_median = summary.get("nonselected_median_variance", summary.get("nonselected_median", float("nan")))
    sel_count = summary.get("selected_count", 0)
    nonsel_count = summary.get("nonselected_count", 0)
    stats_text = [
        ("Metric", "Selected", "Non-selected / Resample"),
        ("─" * 18, "─" * 18, "─" * 26),
        ("Counts", f"{sel_count:,}", f"{nonsel_count:,} (n={summary['resample_size']}, B={summary['num_resamples']})"),
        ("Mean variance", f"{selected_mean:.4f}", f"{nonselected_mean:.4f}"),
        ("Median variance", f"{selected_median:.4f}", f"{nonselected_median:.4f}"),
        ("Resample mean", "", f"{summary['nonselected_resample_mean_median']:.4f} "
         f"[{summary['nonselected_resample_mean_ci_lower']:.4f}, {summary['nonselected_resample_mean_ci_upper']:.4f}]"),
        ("P(resample > selected)", "", f"{summary['fraction_nonselected_mean_greater']:.3f}"),
        ("Selected percentile", "", f"{summary['percentile_selected_in_nonselected']:.1f}%"),
    ]

    table_text = "\n".join([f"{row[0]:<20} {row[1]:<20} {row[2]}" for row in stats_text])

    ax_stats.text(0.5, 0.95, table_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment="top", horizontalalignment="center",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax_stats.set_title("C. Statistical Summary", fontsize=12, fontweight="bold", loc="center")

    # Overall figure title
    fig.suptitle(
        "Selected vs Non-selected Trial-to-Trial Variance\nWithin Motor Cortex",
        fontsize=14, fontweight="bold", y=0.98
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "(negligible)"
    elif d < 0.5:
        return "(small)"
    elif d < 0.8:
        return "(medium)"
    else:
        return "(large)"


# =========================================================================
# ENHANCED ANALYSIS FUNCTIONS FOR BETTER SEPARATION
# =========================================================================

def _log_transform_variability(values, epsilon=1e-6):
    """
    Log-transform variability values to reduce skewness.

    The log transform compresses extreme values and expands small differences,
    making the distributions more Gaussian-like and better revealing separation.
    """
    return np.log10(values + epsilon)


def _compute_rank_percentiles(selected_vals, nonselected_vals):
    """
    Compute rank percentiles of selected voxels within the full motor cortex distribution.

    If selected voxels have systematically lower variability, their rank percentiles
    should cluster in the lower portion of the distribution.
    """
    combined = np.concatenate([selected_vals, nonselected_vals])
    n_total = combined.size

    # Compute rank of each selected voxel within the combined distribution
    selected_ranks = np.empty(selected_vals.size, dtype=np.float64)
    for i, val in enumerate(selected_vals):
        # Rank = fraction of values that are <= this value
        selected_ranks[i] = np.sum(combined <= val) / n_total * 100

    # Expected percentile under null hypothesis (uniform distribution)
    expected_median_percentile = 50.0

    return selected_ranks, expected_median_percentile


def _matched_bootstrap_effect(selected_vals, nonselected_vals, num_bootstrap, rng, match_size=None):
    """
    Bootstrap comparison with matched sample sizes.

    For each bootstrap iteration:
    1. Resample selected voxels with replacement
    2. Sample the same number of non-selected voxels WITHOUT replacement
    3. Compute the difference in means/medians

    This controls for sample size effects and provides robust effect estimates.
    """
    if match_size is None:
        match_size = selected_vals.size

    # Ensure we can sample without replacement from nonselected
    use_replace = nonselected_vals.size < match_size

    mean_diffs = np.empty(num_bootstrap, dtype=np.float64)
    median_diffs = np.empty(num_bootstrap, dtype=np.float64)
    cohens_ds = np.empty(num_bootstrap, dtype=np.float64)

    for i in range(num_bootstrap):
        # Resample selected WITH replacement
        sel_sample = rng.choice(selected_vals, size=match_size, replace=True)
        # Sample non-selected (with replacement only if necessary)
        nonsel_sample = rng.choice(nonselected_vals, size=match_size, replace=use_replace)

        mean_diffs[i] = np.mean(sel_sample) - np.mean(nonsel_sample)
        median_diffs[i] = np.median(sel_sample) - np.median(nonsel_sample)
        cohens_ds[i] = _cohens_d(sel_sample, nonsel_sample)

    return {
        "mean_diffs": mean_diffs,
        "median_diffs": median_diffs,
        "cohens_ds": cohens_ds,
        "mean_diff_mean": float(np.mean(mean_diffs)),
        "mean_diff_ci_lower": float(np.percentile(mean_diffs, 2.5)),
        "mean_diff_ci_upper": float(np.percentile(mean_diffs, 97.5)),
        "median_diff_mean": float(np.mean(median_diffs)),
        "median_diff_ci_lower": float(np.percentile(median_diffs, 2.5)),
        "median_diff_ci_upper": float(np.percentile(median_diffs, 97.5)),
        "cohens_d_mean": float(np.mean(cohens_ds)),
        "cohens_d_ci_lower": float(np.percentile(cohens_ds, 2.5)),
        "cohens_d_ci_upper": float(np.percentile(cohens_ds, 97.5)),
        "match_size": match_size,
    }


def _compute_low_variability_fractions(selected_vals, nonselected_vals, thresholds=None):
    """
    Compute the fraction of voxels below various variability thresholds.

    If selected voxels have systematically lower variability, they should have
    a higher fraction falling below any given threshold.
    """
    combined = np.concatenate([selected_vals, nonselected_vals])

    if thresholds is None:
        # Use percentiles of the combined distribution as thresholds
        thresholds = np.percentile(combined, [10, 20, 30, 40, 50])

    results = []
    for thresh in thresholds:
        sel_frac = np.mean(selected_vals <= thresh)
        nonsel_frac = np.mean(nonselected_vals <= thresh)
        # Odds ratio: how much more likely is a selected voxel to be below threshold?
        if nonsel_frac > 0 and nonsel_frac < 1:
            odds_ratio = (sel_frac / (1 - sel_frac + 1e-10)) / (nonsel_frac / (1 - nonsel_frac + 1e-10))
        else:
            odds_ratio = np.nan
        results.append({
            "threshold": float(thresh),
            "selected_fraction": float(sel_frac),
            "nonselected_fraction": float(nonsel_frac),
            "fraction_ratio": float(sel_frac / (nonsel_frac + 1e-10)),
            "odds_ratio": float(odds_ratio),
        })

    return results


def _compute_stochastic_dominance(selected_vals, nonselected_vals):
    """
    Compute probability that a randomly chosen selected voxel has lower
    variability than a randomly chosen non-selected voxel.

    This is the area under the ROC curve (AUC) and equals P(X < Y) + 0.5*P(X == Y).
    A value > 0.5 indicates selected voxels tend to have lower variability.
    """
    # Mann-Whitney U statistic gives us this directly
    n_sel = selected_vals.size
    n_nonsel = nonselected_vals.size

    # Count pairs where selected < nonselected
    count_less = 0
    count_equal = 0

    # For large samples, use a more efficient approach
    if n_sel * n_nonsel > 1e8:
        # Use sorted arrays and binary search
        sorted_nonsel = np.sort(nonselected_vals)
        for val in selected_vals:
            # Count how many non-selected values are > val
            count_less += np.searchsorted(sorted_nonsel, val, side='right')
            count_equal += (np.searchsorted(sorted_nonsel, val, side='right') -
                           np.searchsorted(sorted_nonsel, val, side='left'))
    else:
        # Direct comparison
        for val in selected_vals:
            count_less += np.sum(nonselected_vals > val)
            count_equal += np.sum(nonselected_vals == val)

    p_dominance = (count_less + 0.5 * count_equal) / (n_sel * n_nonsel)
    return p_dominance


def _compute_cliff_delta(selected_vals, nonselected_vals):
    """
    Compute Cliff's delta - a non-parametric effect size measure.

    Cliff's delta = (P(X > Y) - P(X < Y)) where X is selected, Y is non-selected.
    Ranges from -1 (all X < Y) to +1 (all X > Y).
    Negative values indicate selected have lower variability.
    """
    n_sel = selected_vals.size
    n_nonsel = nonselected_vals.size

    # Use efficient comparison
    # For each selected value, count how many non-selected values are:
    # - greater (selected < nonsel, i.e., selected has lower variability)
    # - less (selected > nonsel, i.e., selected has higher variability)
    count_sel_less = 0  # pairs where selected < nonselected
    count_sel_greater = 0  # pairs where selected > nonselected

    sorted_nonsel = np.sort(nonselected_vals)
    for val in selected_vals:
        idx_right = np.searchsorted(sorted_nonsel, val, side='right')
        idx_left = np.searchsorted(sorted_nonsel, val, side='left')
        # Values in sorted_nonsel that are > val (selected is smaller)
        count_sel_less += n_nonsel - idx_right
        # Values in sorted_nonsel that are < val (selected is larger)
        count_sel_greater += idx_left

    # Cliff's delta: (# selected > nonsel) - (# selected < nonsel)
    # Negative when selected values are generally LOWER
    cliff_delta = (count_sel_greater - count_sel_less) / (n_sel * n_nonsel)
    return cliff_delta


def _plot_enhanced_figure(
    selected_vals,
    nonselected_vals,
    summary,
    resampled_means,
    log_selected,
    log_nonselected,
    out_path,
):
    """
    Create an enhanced multi-panel publication figure with better separation visualization.

    Panels:
    A. Variance-scale KDE comparison
    B. Log-transformed KDE comparison (reduces skew)
    C. Low-variability prevalence ratio across percentile thresholds
    D. Size-matched resampling distribution
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # Color scheme
    sel_color = "#E63946"  # Red for selected
    nonsel_color = "#457B9D"  # Blue for non-selected
    null_color = "#A8DADC"  # Light blue for null distribution

    # ===== Panel A: Variance-scale KDE Comparison =====
    ax_var = fig.add_subplot(gs[0, 0])

    combined_raw = np.concatenate([selected_vals, nonselected_vals])
    xmin_raw = np.percentile(combined_raw, 1)
    xmax_raw = np.percentile(combined_raw, 99)
    pad_raw = (xmax_raw - xmin_raw) * 0.1
    xmin_raw = max(0, xmin_raw - pad_raw)
    xmax_raw = xmax_raw + pad_raw
    grid_raw = np.linspace(xmin_raw, xmax_raw, 500)

    sel_kde_raw = stats.gaussian_kde(selected_vals, bw_method="scott")
    nonsel_kde_raw = stats.gaussian_kde(nonselected_vals, bw_method="scott")

    ax_var.fill_between(grid_raw, sel_kde_raw(grid_raw), alpha=0.4, color=sel_color,
                        label=f"Selected (n={selected_vals.size})")
    ax_var.fill_between(grid_raw, nonsel_kde_raw(grid_raw), alpha=0.4, color=nonsel_color,
                        label=f"Non-selected (n={nonselected_vals.size})")
    ax_var.plot(grid_raw, sel_kde_raw(grid_raw), color=sel_color, linewidth=2)
    ax_var.plot(grid_raw, nonsel_kde_raw(grid_raw), color=nonsel_color, linewidth=2)

    selected_mean_raw = summary.get("selected_mean_variance", float(np.mean(selected_vals)))
    nonselected_mean_raw = summary.get("nonselected_mean_variance", float(np.mean(nonselected_vals)))
    ax_var.axvline(selected_mean_raw, color=sel_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax_var.axvline(nonselected_mean_raw, color=nonsel_color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax_var.set_xlabel("Trial-to-Trial Variance", fontsize=11)
    ax_var.set_ylabel("Density", fontsize=11)
    ax_var.legend(loc="upper right", fontsize=9)
    ax_var.set_xlim(xmin_raw, xmax_raw)
    ax_var.spines["top"].set_visible(False)
    ax_var.spines["right"].set_visible(False)

    # ===== Panel B: Log-transformed KDE Comparison =====
    ax_log = fig.add_subplot(gs[0, 1])

    combined_log = np.concatenate([log_selected, log_nonselected])
    xmin = np.percentile(combined_log, 1)
    xmax = np.percentile(combined_log, 99)
    pad = (xmax - xmin) * 0.1
    xmin = xmin - pad
    xmax = xmax + pad
    grid = np.linspace(xmin, xmax, 500)

    sel_kde = stats.gaussian_kde(log_selected, bw_method='scott')
    nonsel_kde = stats.gaussian_kde(log_nonselected, bw_method='scott')

    ax_log.fill_between(grid, sel_kde(grid), alpha=0.4, color=sel_color,
                        label=f"Selected (n={log_selected.size})")
    ax_log.fill_between(grid, nonsel_kde(grid), alpha=0.4, color=nonsel_color,
                        label=f"Non-selected (n={log_nonselected.size})")
    ax_log.plot(grid, sel_kde(grid), color=sel_color, linewidth=2)
    ax_log.plot(grid, nonsel_kde(grid), color=nonsel_color, linewidth=2)

    # Add mean lines
    ax_log.axvline(np.mean(log_selected), color=sel_color, linestyle='--', linewidth=1.5, alpha=0.8)
    ax_log.axvline(np.mean(log_nonselected), color=nonsel_color, linestyle='--', linewidth=1.5, alpha=0.8)

    ax_log.set_xlabel("Log₁₀(Trial-to-Trial Variance)", fontsize=11)
    ax_log.set_ylabel("Density", fontsize=11)
    ax_log.legend(loc='upper right', fontsize=9)
    ax_log.set_xlim(xmin, xmax)
    ax_log.spines['top'].set_visible(False)
    ax_log.spines['right'].set_visible(False)

    # ===== Panel C: Low-Variability Prevalence =====
    # Shows: at each percentile threshold, what fraction of selected vs non-selected
    # voxels fall below that threshold? Ratio > 1 means selected are enriched among low-variability voxels.
    ax_enrich = fig.add_subplot(gs[1, 0])

    combined = np.concatenate([selected_vals, nonselected_vals])
    percentile_thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    enrichment_ratios = []

    for pct in percentile_thresholds:
        thresh = np.percentile(combined, pct)
        sel_frac = np.mean(selected_vals <= thresh)
        nonsel_frac = np.mean(nonselected_vals <= thresh)
        if nonsel_frac > 0:
            enrichment = sel_frac / nonsel_frac
        else:
            enrichment = np.nan
        enrichment_ratios.append(enrichment)

    bars = ax_enrich.bar(percentile_thresholds, enrichment_ratios, width=8,
                         color=sel_color, alpha=0.7, edgecolor='white')
    ax_enrich.axhline(1.0, color='gray', linestyle='--', linewidth=2, label='Selected = non-selected')

    # Color bars above/below 1.0 differently
    for bar, ratio in zip(bars, enrichment_ratios):
        if ratio > 1.0:
            bar.set_color(sel_color)
        else:
            bar.set_color(nonsel_color)
            bar.set_alpha(0.5)

    ax_enrich.set_xlabel("Variability Percentile Threshold", fontsize=11, labelpad=8)
    ax_enrich.set_ylabel("Relative Prevalence\n(Selected / Non-selected)", fontsize=11)
    ax_enrich.set_xticks(percentile_thresholds)
    ax_enrich.set_xticklabels([f"{p}%" for p in percentile_thresholds], fontsize=9, rotation=0, ha="center")
    ax_enrich.tick_params(axis="x", pad=6)
    ax_enrich.legend(loc='upper right', fontsize=9)
    ax_enrich.spines['top'].set_visible(False)
    ax_enrich.spines['right'].set_visible(False)
    ax_enrich.set_ylim(0, max(enrichment_ratios) * 1.15)

    # ===== Panel D: Size-matched Resampling Distribution =====
    ax_perm = fig.add_subplot(gs[1, 1])

    if resampled_means.size:
        ax_perm.hist(resampled_means, bins=50, color=null_color, edgecolor='white', alpha=0.8, density=True)
        median = summary["nonselected_resample_mean_median"]
        ci_low = summary["nonselected_resample_mean_ci_lower"]
        ci_high = summary["nonselected_resample_mean_ci_upper"]
        observed = summary["observed_selected_mean"]
        ax_perm.axvline(observed, color=sel_color, linewidth=2.5, linestyle="--")
        ax_perm.axvline(median, color="gray", linewidth=1.5, linestyle="-")
        ax_perm.axvline(ci_low, color="gray", linewidth=1.5, linestyle="--")
        ax_perm.axvline(ci_high, color="gray", linewidth=1.5, linestyle="--")
        line_handles = [
            Line2D([0], [0], color=sel_color, linewidth=2.5, linestyle="--"),
            Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-"),
            Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--"),
        ]
        line_labels = [
            f"Selected mean = {observed:.4f}",
            f"Resample median = {median:.4f}",
            f"95% CI = [{ci_low:.4f}, {ci_high:.4f}]",
        ]
        ax_perm.legend(
            line_handles,
            line_labels,
            loc="upper right",
            fontsize=8,
            frameon=True,
            borderpad=0.4,
            handlelength=2.2,
            labelspacing=0.4,
        )
    else:
        ax_perm.text(0.5, 0.5, "No resamples", transform=ax_perm.transAxes,
                     fontsize=11, horizontalalignment="center", verticalalignment="center")

    ax_perm.set_xlabel("Mean Variance (non-selected voxels)", fontsize=11)
    ax_perm.set_ylabel("Density", fontsize=11)
    ax_perm.spines['top'].set_visible(False)
    ax_perm.spines['right'].set_visible(False)

    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _plot_statistical_summary_figure(summary, out_path):
    """Plot the statistical summary table as a standalone figure."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    selected_mean = summary.get("selected_mean_variance", summary.get("selected_mean", float("nan")))
    nonselected_mean = summary.get("nonselected_mean_variance", summary.get("nonselected_mean", float("nan")))
    selected_median = summary.get("selected_median_variance", summary.get("selected_median", float("nan")))
    nonselected_median = summary.get("nonselected_median_variance", summary.get("nonselected_median", float("nan")))
    sel_count = summary.get("selected_count", 0)
    nonsel_count = summary.get("nonselected_count", 0)

    stats_lines = [
        "=" * 90,
        f"{'STATISTICAL SUMMARY':^90}",
        "=" * 90,
        f"{'Sample Sizes:':<25} Selected: {sel_count:,}    Non-selected: {nonsel_count:,}",
        f"{'Resampling:':<25} n={summary['resample_size']}    B={summary['num_resamples']}",
        "-" * 90,
        f"{'VARIABILITY':<25} {'Selected':<15} {'Non-selected':<15}",
        f"Mean variance:{'':<12} {selected_mean:<15.4f} {nonselected_mean:<15.4f}",
        f"Median variance:{'':<10} {selected_median:<15.4f} {nonselected_median:<15.4f}",
        "-" * 90,
        f"{'RESAMPLING SUMMARY':<25} {'Median [95% CI]':<40}",
        f"Resample mean:{'':<13} {summary['nonselected_resample_mean_median']:.4f} "
        f"[{summary['nonselected_resample_mean_ci_lower']:.4f}, {summary['nonselected_resample_mean_ci_upper']:.4f}]",
        f"P(resample > selected):{'':<5} {summary['fraction_nonselected_mean_greater']:.3f}",
        f"Selected percentile:{'':<6} {summary['percentile_selected_in_nonselected']:.1f}%",
        "=" * 90,
    ]

    table_text = "\n".join(stats_lines)

    ax.text(0.5, 0.95, table_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6", pad=1))

    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_histogram_comparison(selected_vals, motor_vals, bins, out_path, selected_mean, motor_mean, title_note=""):
    """Plot two smooth density estimates with a shared range."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    combined = np.concatenate([selected_vals, motor_vals])
    if np.ndim(bins) == 0:
        xmin = float(np.min(combined))
        xmax = float(np.max(combined))
        grid_points = max(200, int(bins) * 10)
    else:
        edges = np.asarray(bins, dtype=np.float64)
        xmin = float(edges[0])
        xmax = float(edges[-1])
        grid_points = max(200, (edges.size - 1) * 5)
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        raise ValueError("Histogram range must be finite to plot KDE.")
    if xmin == xmax:
        pad = 1.0 if xmin == 0.0 else abs(xmin) * 0.05
        xmin -= pad
        xmax += pad
    else:
        pad = (xmax - xmin) * 0.05
        xmin -= pad
        xmax += pad

    grid = np.linspace(xmin, xmax, grid_points, dtype=np.float64)
    selected_kde = stats.gaussian_kde(selected_vals)
    motor_kde = stats.gaussian_kde(motor_vals)
    selected_density = selected_kde(grid)
    motor_density = motor_kde(grid)

    ax.plot(grid, selected_density, color="crimson", linewidth=2.0, label=f"Selected (n={selected_vals.size})")
    ax.fill_between(grid, 0.0, selected_density, color="crimson", alpha=0.25, linewidth=0)
    ax.plot(grid, motor_density, color="navy", linewidth=2.0, label=f"Motor-only (n={motor_vals.size})")
    ax.fill_between(grid, 0.0, motor_density, color="navy", alpha=0.25, linewidth=0)
    ax.axvline(selected_mean, color="crimson", linestyle="--", linewidth=1)
    ax.axvline(motor_mean, color="navy", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean variance across trials")
    ax.set_ylabel("Density (KDE)")
    title = "Voxel variability comparison"
    if title_note:
        title = f"{title}\n{title_note}"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_resampled_mean_distribution(
    resampled_means,
    observed_selected_mean,
    out_path,
    n_selected,
    num_resamples,
    fraction_greater,
    metric_label="Variance",
    title="Size-matched resampling of non-selected voxels",
):
    """Plot resampled non-selected mean variance distribution with observed selected mean."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    if resampled_means.size:
        ax.hist(resampled_means, bins=45, color="#A8DADC", edgecolor="white", alpha=0.85, density=True)
        if resampled_means.size > 1:
            xmin = float(np.min(resampled_means))
            xmax = float(np.max(resampled_means))
            if xmin != xmax:
                grid = np.linspace(xmin, xmax, 400)
                kde = stats.gaussian_kde(resampled_means)
                ax.plot(grid, kde(grid), color="#457B9D", linewidth=2)
    else:
        ax.text(0.5, 0.5, "No resamples", transform=ax.transAxes,
                fontsize=11, horizontalalignment="center", verticalalignment="center")

    ax.axvline(observed_selected_mean, color="#E63946", linestyle="--", linewidth=2.0,
               label=f"Selected mean = {observed_selected_mean:.4f}")

    ax.set_xlabel(f"Mean {metric_label} (non-selected resamples, n={n_selected})")
    ax.set_ylabel("Density")
    ax.set_title(title)

    if resampled_means.size:
        textstr = (
            f"Resamples: {num_resamples}\n"
            f"P(resample > selected) = {fraction_greater:.3f}"
        )
        props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
        ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top", horizontalalignment="right", bbox=props)

    ax.legend(loc="upper left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    """Run the end-to-end workflow: predict BOLD, compute variance, and compare groups."""
    parser = argparse.ArgumentParser(description=("Compute predicted BOLD from GLMsingle outputs, estimate per-voxel trial variance, "
            "and compare selected vs motor voxel variability."))
    parser.add_argument("--runs", default="1,2", help="Comma-separated run numbers (1-based).")
    parser.add_argument("--model-path", default="data/sub09_ses01/TYPED_FITHRF_GLMDENOISE_RR.npy")
    parser.add_argument("--design-path", default="data/sub09_ses01/DESIGNINFO.npy")
    parser.add_argument("--out-dir", default="data/sub09_ses01")
    parser.add_argument("--output-prefix", default="data/sub09_ses01/voxel_weights_mean_foldavg_sub09_ses1_task0.8_bold1_beta0.5_smooth1.2_gamma1.5", help="Prefix used to locate indices files and name outputs.")
    parser.add_argument("--trial-len", type=int, default=9, help="Number of TRs per trial segment.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Voxel chunk size for matrix multiplies.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for resampling.")
    parser.add_argument("--num-resamples", type=int, default=1000, help="Number of size-matched resamples for variability analysis.")
    parser.add_argument("--hist-bins", type=int, default=30, help="Histogram bin count.")
    parser.add_argument("--hist-subsample", dest="hist_subsample", action="store_true", default=True, help="Subsample motor-only voxels for histogram to match selected count.")
    parser.add_argument("--no-hist-subsample", dest="hist_subsample", action="store_false", help="Do not subsample motor-only voxels for histogram.")
    parser.add_argument("--mask-indices", default=None, help="Optional mask_indices.npy with GLMsingle voxel ordering.")
    parser.add_argument("--mask-dir", default=None, help="Directory to search for brain/csf/gray masks if needed.")
    parser.add_argument("--mask-gray-threshold", type=float, default=None, help="Gray matter PVE threshold for mask matching.")
    args = parser.parse_args()

    args.model_path = _pad_subject_id(args.model_path)
    args.design_path = _pad_subject_id(args.design_path)
    args.out_dir = _pad_subject_id(args.out_dir)
    args.output_prefix = _pad_subject_id(args.output_prefix)

    runs = _parse_runs(args.runs)
    if not runs:
        raise ValueError("No runs specified.")

    out_dir = _resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix_path = _resolve_path(args.output_prefix)
    output_tag = output_prefix_path.name

    model_path = _resolve_path(args.model_path)
    design_path = _resolve_path(args.design_path)

    design_single_list, hrflibrary = _load_design(design_path)
    betasmd, hrfindexrun = _load_model(model_path)

    betas, numtrials, volume_shape = _flatten_betas(betasmd)
    voxels = betas.shape[0]
    numruns = len(design_single_list)
    hrfindex_flat = _flatten_hrfindex(hrfindexrun, numruns, voxels)

    mask_indices = None
    mask_shape = None
    if len(volume_shape) != 3 or volume_shape[1] == 1 or volume_shape[2] == 1:
        mask_dir = Path(args.mask_dir) if args.mask_dir else model_path.parent
        if args.mask_indices:
            mask_indices = _load_mask_indices_file(_resolve_path(args.mask_indices))
            brain_path = _first_glob(mask_dir, "*T1w_brain_mask.nii*")
            if brain_path is not None:
                mask_shape = nib.load(brain_path).shape[:3]
        else:
            brain_path = _first_glob(mask_dir, "*T1w_brain_mask.nii*")
            csf_path = _first_glob(mask_dir, "*T1w_brain_pve_0.nii*")
            gray_path = _first_glob(mask_dir, "*T1w_brain_pve_1.nii*")
            if brain_path and csf_path and gray_path:
                mask_indices, mask_shape, gray_thr = _mask_indices_from_masks(
                    brain_path, csf_path, gray_path, voxels, args.mask_gray_threshold
                )
                print(
                    f"Using gray threshold {gray_thr:.2f} to match {voxels} voxels "
                    f"from masks in {mask_dir}.",
                    flush=True,
                )

    run_indices = [run - 1 for run in runs]
    if min(run_indices) < 0 or max(run_indices) >= numruns:
        raise ValueError(f"Runs must be between 1 and {numruns}, got {runs}.")

    trial_counts = [design.shape[1] for design in design_single_list]
    total_trials = int(np.sum(trial_counts))
    if numtrials == total_trials:
        run_offsets = np.cumsum([0] + trial_counts)
        slice_betas = True
    elif all(numtrials == trial_counts[idx] for idx in run_indices):
        run_offsets = None
        slice_betas = False
    else:
        raise ValueError(
            f"Betas trials ({numtrials}) do not match design trials ({trial_counts})."
        )

    run_timepoints = [design_single_list[idx].shape[0] for idx in run_indices]
    total_timepoints = int(np.sum(run_timepoints))
    if args.trial_len <= 0:
        raise ValueError("trial-len must be positive.")

    xbeta_path = out_dir / f"{output_tag}_X_beta.npy"
    xbeta = np.lib.format.open_memmap(
        xbeta_path, mode="w+", dtype=np.float32, shape=(voxels, total_timepoints)
    )

    sum_vals = np.zeros((voxels, args.trial_len), dtype=np.float64)
    sumsq_vals = np.zeros((voxels, args.trial_len), dtype=np.float64)
    count_vals = np.zeros((voxels, args.trial_len), dtype=np.int64)

    time_offset = 0
    total_used = 0
    total_skipped = 0
    for run_idx, run in zip(run_indices, runs):
        design_single = design_single_list[run_idx]
        run_trials = design_single.shape[1]
        run_timepoints = design_single.shape[0]
        if args.trial_len > run_timepoints:
            raise ValueError(
                f"trial-len ({args.trial_len}) exceeds run timepoints ({run_timepoints}) for run {run}."
            )
        if slice_betas:
            start = int(run_offsets[run_idx])
            end = int(run_offsets[run_idx + 1])
            betas_run = betas[:, start:end]
        else:
            betas_run = betas
        if betas_run.shape[1] != run_trials:
            raise ValueError(
                f"Run {run} betas trials {betas_run.shape[1]} do not match design trials {run_trials}."
            )

        hrf_idx_run = hrfindex_flat[:, run_idx]
        pred = _predict_run_timeseries(
            betas_run, hrf_idx_run, design_single, hrflibrary, args.chunk_size
        )
        xbeta[:, time_offset : time_offset + run_timepoints] = pred

        onsets = _trial_onsets_from_design(design_single)
        used = 0
        skipped = 0
        for onset in onsets:
            if onset < 0:
                skipped += 1
                continue
            end = onset + args.trial_len
            if end > run_timepoints:
                skipped += 1
                continue
            segment = pred[:, onset:end]
            _accumulate_trial_stats(sum_vals, sumsq_vals, count_vals, segment)
            used += 1

        time_offset += run_timepoints
        total_used += used
        total_skipped += skipped
        print(
            f"Run {run}: predicted {pred.shape}, trials used={used}, skipped={skipped}.",
            flush=True,
        )

    if time_offset != total_timepoints:
        raise RuntimeError("Timepoints written do not match expected total.")
    xbeta.flush()
    print(f"Saved predicted BOLD: {xbeta_path}", flush=True)

    trial_variance = _compute_trial_variance(sum_vals, sumsq_vals, count_vals)
    trial_var_path = out_dir / f"{output_tag}_trial_variance.npy"
    np.save(trial_var_path, trial_variance)
    print(f"Saved trial variance: {trial_var_path}", flush=True)

    voxel_var = np.nanmean(trial_variance, axis=1, keepdims=True).astype(np.float32, copy=False)
    voxel_var_path = out_dir / f"{output_tag}_voxel_var.npy"
    np.save(voxel_var_path, voxel_var)
    print(f"Saved voxel variance: {voxel_var_path}", flush=True)

    count_path = out_dir / f"{output_tag}_trial_counts.npy"
    np.save(count_path, count_vals.astype(np.int32, copy=False))

    motor_path = _resolve_path(f"{args.output_prefix}_motor_voxel_indicies.npz")
    selected_path = _resolve_path(f"{args.output_prefix}_selected_voxel_indicies.npz")
    if not motor_path.exists():
        raise FileNotFoundError(f"Motor indices file not found: {motor_path}")
    if not selected_path.exists():
        raise FileNotFoundError(f"Selected indices file not found: {selected_path}")

    motor_indices = _validate_indices(
        _load_indices_npz(motor_path, volume_shape, mask_indices=mask_indices, mask_shape=mask_shape),
        voxels,
        "Motor",
    )
    selected_indices = _validate_indices(
        _load_indices_npz(selected_path, volume_shape, mask_indices=mask_indices, mask_shape=mask_shape),
        voxels,
        "Selected",
    )

    overlap = np.intersect1d(selected_indices, motor_indices)
    motor_only = np.setdiff1d(motor_indices, selected_indices)
    selected_not_in_motor = np.setdiff1d(selected_indices, motor_indices)
    if motor_only.size == 0:
        raise ValueError("Motor-only voxel set is empty after removing selected voxels.")

    voxel_var_flat = voxel_var.reshape(-1)
    selected_vals = voxel_var_flat[selected_indices]
    motor_vals = voxel_var_flat[motor_only]
    selected_vals = selected_vals[np.isfinite(selected_vals)]
    motor_vals = motor_vals[np.isfinite(motor_vals)]
    if selected_vals.size == 0 or motor_vals.size == 0:
        raise ValueError("No finite voxel variance values for selected or motor voxels.")

    rng = np.random.default_rng(args.seed)
    hist_motor_vals = motor_vals
    hist_note = ""
    if args.hist_subsample and motor_vals.size > selected_vals.size:
        hist_motor_vals = rng.choice(motor_vals, size=selected_vals.size, replace=False)
        hist_note = "Motor-only distribution subsampled to match selected count."

    bins = np.histogram_bin_edges(np.concatenate([selected_vals, motor_vals]), bins=args.hist_bins)
    stats_summary, resampled_means = _size_matched_resample_summary(
        selected_vals, motor_vals, rng, args.num_resamples
    )

    stats_summary.update({
        "analysis_type": "selected_vs_motor_resample",
        "metric": "variance",
        "selected_mean": float(np.mean(selected_vals)),
        "motor_mean": float(np.mean(motor_vals)),
        "selected_median": float(np.median(selected_vals)),
        "motor_median": float(np.median(motor_vals)),
        "run_list": runs,
        "trial_len": args.trial_len,
        "total_trials_used": int(total_used),
        "total_trials_skipped": int(total_skipped),
        "total_timepoints": int(total_timepoints),
        "voxel_count": int(voxels),
        "motor_indices_path": str(motor_path),
        "selected_indices_path": str(selected_path),
        "motor_count_total": int(motor_indices.size),
        "motor_only_count": int(motor_only.size),
        "selected_overlap_with_motor": int(overlap.size),
        "selected_not_in_motor": int(selected_not_in_motor.size),
        "hist_subsample": bool(args.hist_subsample),
        "hist_motor_count": int(hist_motor_vals.size),
    })

    hist_path = out_dir / f"{output_tag}_voxel_var_hist.png"
    _plot_histogram_comparison(
        selected_vals,
        hist_motor_vals,
        bins,
        hist_path,
        stats_summary["selected_mean"],
        stats_summary["motor_mean"],
        title_note=hist_note,
    )
    print(f"Saved histogram: {hist_path}", flush=True)

    metrics_path = out_dir / f"{output_tag}_variance_comparison.npz"
    np.savez(
        metrics_path,
        selected_values=selected_vals,
        motor_values=motor_vals,
        motor_hist_values=hist_motor_vals,
        resampled_nonselected_mean_variances=resampled_means,
        observed_selected_mean_variance=stats_summary["observed_selected_mean"],
        fraction_nonselected_mean_greater=stats_summary["fraction_nonselected_mean_greater"],
        percentile_selected_in_nonselected=stats_summary["percentile_selected_in_nonselected"],
        resample_size=stats_summary["resample_size"],
        num_resamples=stats_summary["num_resamples"],
        selected_indices=selected_indices,
        motor_indices=motor_indices,
        motor_only_indices=motor_only,
    )

    summary_path = out_dir / f"{output_tag}_variance_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(stats_summary, handle, indent=2)

    print(
        "Variance comparison complete: "
        f"selected={stats_summary['selected_count']}, "
        f"motor_only={stats_summary['motor_only_count']}, "
        f"P(resampled motor mean > selected mean)={stats_summary['fraction_nonselected_mean_greater']:.3f} "
        f"({stats_summary['fraction_nonselected_mean_greater']*100:.1f}%), "
        f"selected_mean_percentile={stats_summary['percentile_selected_in_nonselected']:.1f}",
        flush=True,
    )

    # =========================================================================
    # NEW ANALYSIS: Motor Cortex Variability Comparison
    # Compare selected voxels WITHIN motor cortex vs non-selected motor voxels
    # This is the key analysis to demonstrate that optimization selects
    # low-variability voxels, not noise.
    # =========================================================================

    print("\n" + "=" * 70)
    print("MOTOR CORTEX VARIABILITY ANALYSIS (SIZE-MATCHED RESAMPLING)")
    print("Comparing selected vs non-selected voxels WITHIN motor cortex")
    print("=" * 70)

    # Compute motor cortex comparison using variance (no std)
    selected_motor_vals, nonselected_motor_vals, selected_motor_idx, nonselected_motor_idx = \
        _compute_motor_cortex_comparison(voxel_var_flat, selected_indices, motor_indices)

    if selected_motor_vals.size < 10:
        print(f"WARNING: Only {selected_motor_vals.size} selected voxels in motor cortex. "
              "Analysis may not be reliable.", flush=True)

    print(f"Selected voxels in motor cortex: {selected_motor_vals.size}")
    print(f"Non-selected voxels in motor cortex: {nonselected_motor_vals.size}")

    num_resamples = int(args.num_resamples)
    motor_summary, resampled_means = _size_matched_resample_summary(
        selected_motor_vals, nonselected_motor_vals, rng, num_resamples
    )

    motor_summary.update({
        "analysis_type": "motor_cortex_variability_resample",
        "metric": "variance",
        "run_list": runs,
        "trial_len": args.trial_len,
        "total_trials_used": int(total_used),
        "total_trials_skipped": int(total_skipped),
        "total_timepoints": int(total_timepoints),
        "selected_mean_variance": float(np.mean(selected_motor_vals)),
        "nonselected_mean_variance": float(np.mean(nonselected_motor_vals)),
        "selected_median_variance": float(np.median(selected_motor_vals)),
        "nonselected_median_variance": float(np.median(nonselected_motor_vals)),
    })

    observed_selected_mean = motor_summary["observed_selected_mean"]
    resample_median = motor_summary["nonselected_resample_mean_median"]
    resample_ci_lower = motor_summary["nonselected_resample_mean_ci_lower"]
    resample_ci_upper = motor_summary["nonselected_resample_mean_ci_upper"]
    fraction_greater = motor_summary["fraction_nonselected_mean_greater"]
    percentile = motor_summary["percentile_selected_in_nonselected"]

    print("\n--- Size-matched resampling ---")
    print(f"Observed selected mean variance: {observed_selected_mean:.4f}")
    if np.isfinite(resample_median):
        print(f"Non-selected resample mean variance median [95% CI]: {resample_median:.4f} "
              f"[{resample_ci_lower:.4f}, {resample_ci_upper:.4f}]")
    print(f"P(resampled non-selected mean > observed selected mean): "
          f"{fraction_greater:.3f} ({fraction_greater*100:.1f}%)")
    print(f"Observed selected mean percentile: {percentile:.1f}% (lower = more stable)")

    # Save motor cortex comparison results (without large index arrays for JSON)
    motor_summary_path = out_dir / f"{output_tag}_motor_variability_summary.json"
    with open(motor_summary_path, "w", encoding="utf-8") as handle:
        # Convert numpy arrays to lists for JSON serialization
        summary_for_json = {k: v for k, v in motor_summary.items()
                           if not isinstance(v, np.ndarray)}
        # Don't save huge index lists to JSON
        if "selected_motor_indices" in summary_for_json:
            summary_for_json["selected_motor_indices_count"] = len(selected_motor_idx)
            del summary_for_json["selected_motor_indices"]
        if "nonselected_motor_indices" in summary_for_json:
            summary_for_json["nonselected_motor_indices_count"] = len(nonselected_motor_idx)
            del summary_for_json["nonselected_motor_indices"]
        json.dump(summary_for_json, handle, indent=2)
    print(f"Saved motor cortex summary: {motor_summary_path}")

    # Create motor cortex figures (restore multi-panel layout + resampling panel)
    motor_fig_path = out_dir / f"{output_tag}_motor_variability_figure.png"
    _plot_comprehensive_figure(
        selected_motor_vals, nonselected_motor_vals,
        motor_summary, resampled_means,
        motor_fig_path,
    )
    print(f"Saved motor cortex figure: {motor_fig_path}")

    log_selected = _log_transform_variability(selected_motor_vals)
    log_nonselected = _log_transform_variability(nonselected_motor_vals)

    enhanced_fig_path = out_dir / f"{output_tag}_motor_variability_enhanced.png"
    _plot_enhanced_figure(
        selected_motor_vals, nonselected_motor_vals,
        motor_summary, resampled_means,
        log_selected, log_nonselected,
        enhanced_fig_path,
    )
    print(f"Saved enhanced motor cortex figure: {enhanced_fig_path}")

    summary_fig_path = out_dir / f"{output_tag}_motor_variability_statistical_summary.png"
    _plot_statistical_summary_figure(motor_summary, summary_fig_path)
    print(f"Saved statistical summary figure: {summary_fig_path}")

    # Save detailed data for reproducibility
    motor_data_path = out_dir / f"{output_tag}_motor_variability_data.npz"
    np.savez(
        motor_data_path,
        selected_motor_values=selected_motor_vals,
        nonselected_motor_values=nonselected_motor_vals,
        selected_motor_indices=selected_motor_idx,
        nonselected_motor_indices=nonselected_motor_idx,
        resampled_nonselected_mean_variances=resampled_means,
        observed_selected_mean_variance=observed_selected_mean,
        fraction_nonselected_mean_greater=fraction_greater,
        percentile_selected_in_nonselected=percentile,
        resample_size=selected_motor_vals.size,
        num_resamples=num_resamples,
    )
    print(f"Saved motor cortex data: {motor_data_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("MOTOR CORTEX ANALYSIS RESULTS (SIZE-MATCHED RESAMPLING):")
    print("=" * 70)
    print(f"\n{'BASIC STATISTICS:'}")
    print(f"  Selected motor voxels:     n={motor_summary['selected_count']}, "
          f"mean variance={motor_summary['selected_mean_variance']:.4f}")
    print(f"  Non-selected motor voxels: n={motor_summary['nonselected_count']}, "
          f"mean variance={motor_summary['nonselected_mean_variance']:.4f}")
    print(f"  P(resampled non-selected mean > selected mean): "
          f"{fraction_greater:.3f} ({fraction_greater*100:.1f}%)")
    print(f"  Selected mean percentile:  {percentile:.1f}% (lower = more stable)")

    print("\n" + "=" * 70)

    if np.isfinite(fraction_greater):
        if fraction_greater >= 0.95:
            print("\n✓ STRONG EVIDENCE: Selected motor voxels show substantially lower "
                  "trial-to-trial variance than non-selected motor voxels.")
            print("  - Observed selected mean variance is lower than most size-matched resamples")
            print("  This supports the claim that the optimization selects stable voxels.")
        elif fraction_greater >= 0.80:
            print("\n○ MODERATE EVIDENCE: Selected motor voxels tend to have lower "
                  "trial-to-trial variance than non-selected motor voxels.")
            print("  - Observed selected mean variance is lower than many size-matched resamples")
            print("  The optimization appears to favor more stable voxels.")
        else:
            print("\n⚠ LIMITED EVIDENCE: The selected mean variance is not consistently lower.")
            print("  Consider investigating potential issues with the optimization or data.")


if __name__ == "__main__":
    main()
