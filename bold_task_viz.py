#!/usr/bin/env python3
"""
Variability Analysis for Voxel Selection Validation

This script demonstrates that voxels selected by the optimization algorithm
exhibit significantly lower trial-to-trial variability than non-selected voxels
within the same anatomical region (motor cortex).

Key analyses:
1. Compare selected vs non-selected motor cortex voxels
2. Permutation test with null distribution
3. Bootstrap confidence intervals
4. Multi-panel publication-quality figure
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import nibabel as nib
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO_ROOT = Path(__file__).resolve().parent
GLMSINGLE_ROOT = REPO_ROOT / "GLMsingle"
if str(GLMSINGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(GLMSINGLE_ROOT))

from glmsingle.design.convolve_design import convolve_design


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


def _resample_control_means(control_values, target_size, num_resamples, rng):
    """Resample control means for a null distribution."""
    if num_resamples <= 0:
        return np.array([], dtype=np.float64)
    replace = control_values.size < target_size
    means = np.empty(num_resamples, dtype=np.float64)
    for idx in range(num_resamples):
        sample = rng.choice(control_values, size=target_size, replace=replace)
        means[idx] = float(np.mean(sample))
    return means


def _compare_groups(selected_vals, motor_vals, rng, num_resamples):
    """Compute statistics and resampling-based sanity checks."""
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

    ttest = stats.ttest_ind(selected_vals, motor_vals, equal_var=False, nan_policy="omit")
    summary["t_stat"] = float(ttest.statistic) if np.isfinite(ttest.statistic) else float("nan")
    summary["p_two_sided"] = float(ttest.pvalue) if np.isfinite(ttest.pvalue) else float("nan")
    if np.isfinite(summary["t_stat"]) and np.isfinite(summary["p_two_sided"]):
        summary["p_one_sided"] = summary["p_two_sided"] / 2.0 if summary["t_stat"] > 0 else 1.0 - summary["p_two_sided"] / 2.0
    else:
        summary["p_one_sided"] = float("nan")

    summary["cohens_d"] = _cohens_d(selected_vals, motor_vals)

    ks = stats.ks_2samp(selected_vals, motor_vals, alternative="two-sided", mode="auto")
    summary["ks_stat"] = float(ks.statistic)
    summary["ks_pvalue"] = float(ks.pvalue)

    resample_means = _resample_control_means(motor_vals, selected_vals.size, num_resamples, rng)
    if resample_means.size:
        resample_mean = float(np.mean(resample_means))
        resample_std = float(np.std(resample_means, ddof=1)) if resample_means.size > 1 else float("nan")
        diff = summary["selected_mean"] - summary["motor_mean"]
        if diff >= 0:
            p_one = (np.sum(resample_means >= summary["selected_mean"]) + 1) / (resample_means.size + 1)
        else:
            p_one = (np.sum(resample_means <= summary["selected_mean"]) + 1) / (resample_means.size + 1)
        p_two = (np.sum(np.abs(resample_means - summary["motor_mean"]) >= abs(diff)) + 1) / (resample_means.size + 1)
        percentile = float(stats.percentileofscore(resample_means, summary["selected_mean"], kind="mean"))
    else:
        resample_mean = float("nan")
        resample_std = float("nan")
        p_one = float("nan")
        p_two = float("nan")
        percentile = float("nan")

    summary["resample_mean"] = resample_mean
    summary["resample_std"] = resample_std
    summary["resample_p_one_sided"] = float(p_one)
    summary["resample_p_two_sided"] = float(p_two)
    summary["resample_selected_percentile"] = percentile

    motor_std = summary["motor_std"]
    if np.isfinite(motor_std) and motor_std > 0:
        summary["selected_mean_z"] = (summary["selected_mean"] - summary["motor_mean"]) / motor_std
    else:
        summary["selected_mean_z"] = float("nan")

    return summary, resample_means


def _compute_motor_cortex_comparison(voxel_var_flat, selected_indices, motor_indices, use_std=True):
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

    # Convert to standard deviation if requested (more interpretable)
    if use_std:
        selected_motor_vals = np.sqrt(np.maximum(selected_motor_vals, 0))
        nonselected_motor_vals = np.sqrt(np.maximum(nonselected_motor_vals, 0))

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


def _compare_motor_groups(selected_vals, nonselected_vals, rng, num_permutations, num_bootstrap):
    """Comprehensive comparison of selected vs non-selected motor cortex voxels."""
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

    # Difference in means
    summary["mean_diff"] = summary["selected_mean"] - summary["nonselected_mean"]
    summary["mean_diff_pct"] = 100 * summary["mean_diff"] / summary["nonselected_mean"] if summary["nonselected_mean"] != 0 else float("nan")

    # Cohen's d effect size
    summary["cohens_d"] = _cohens_d(selected_vals, nonselected_vals)

    # Welch's t-test (parametric)
    ttest = stats.ttest_ind(selected_vals, nonselected_vals, equal_var=False, nan_policy="omit")
    summary["t_stat"] = float(ttest.statistic) if np.isfinite(ttest.statistic) else float("nan")
    summary["t_pvalue_two_sided"] = float(ttest.pvalue) if np.isfinite(ttest.pvalue) else float("nan")
    # One-sided p-value (selected < nonselected)
    if np.isfinite(summary["t_stat"]):
        summary["t_pvalue_one_sided"] = summary["t_pvalue_two_sided"] / 2.0 if summary["t_stat"] < 0 else 1.0 - summary["t_pvalue_two_sided"] / 2.0
    else:
        summary["t_pvalue_one_sided"] = float("nan")

    # Mann-Whitney U test (non-parametric)
    mwu = stats.mannwhitneyu(selected_vals, nonselected_vals, alternative="less")
    summary["mannwhitney_u"] = float(mwu.statistic)
    summary["mannwhitney_pvalue"] = float(mwu.pvalue)

    # Kolmogorov-Smirnov test
    ks = stats.ks_2samp(selected_vals, nonselected_vals, alternative="two-sided")
    summary["ks_stat"] = float(ks.statistic)
    summary["ks_pvalue"] = float(ks.pvalue)

    # Permutation test
    perm_results = _permutation_test(selected_vals, nonselected_vals, num_permutations, rng, statistic="mean")
    summary["perm_p_value"] = perm_results["p_value_one_sided"]
    summary["perm_percentile"] = perm_results["percentile_rank"]

    # Bootstrap confidence intervals
    sel_lower, sel_upper, _ = _bootstrap_ci(selected_vals, num_bootstrap, rng, ci_level=0.95)
    nonsel_lower, nonsel_upper, _ = _bootstrap_ci(nonselected_vals, num_bootstrap, rng, ci_level=0.95)
    summary["selected_ci_lower"] = sel_lower
    summary["selected_ci_upper"] = sel_upper
    summary["nonselected_ci_lower"] = nonsel_lower
    summary["nonselected_ci_upper"] = nonsel_upper

    # Check if CIs overlap (no overlap = strong separation)
    summary["ci_no_overlap"] = sel_upper < nonsel_lower or nonsel_upper < sel_lower

    return summary, perm_results


def _plot_comprehensive_figure(
    selected_vals, nonselected_vals, summary, perm_results, out_path, use_std=True
):
    """
    Create a publication-quality multi-panel figure demonstrating voxel selection validity.

    Panel A: KDE density comparison
    Panel B: Permutation null distribution with observed value
    Panel C: Summary statistics box
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25, height_ratios=[1.2, 1])

    metric_label = "Standard Deviation" if use_std else "Variance"

    # Color scheme
    sel_color = "#E63946"  # Red for selected
    nonsel_color = "#457B9D"  # Blue for non-selected
    null_color = "#A8DADC"  # Light blue for null distribution

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
    ax_kde.axvline(summary["selected_mean"], color=sel_color, linestyle="--", linewidth=1.5, alpha=0.8)
    ax_kde.axvline(summary["nonselected_mean"], color=nonsel_color, linestyle="--", linewidth=1.5, alpha=0.8)

    ax_kde.set_xlabel(f"Trial-to-Trial {metric_label}", fontsize=11)
    ax_kde.set_ylabel("Density", fontsize=11)
    ax_kde.set_title("A. Variability Distributions in Motor Cortex", fontsize=12, fontweight="bold")
    ax_kde.legend(loc="upper right", fontsize=9)
    ax_kde.set_xlim(xmin, xmax)
    ax_kde.spines["top"].set_visible(False)
    ax_kde.spines["right"].set_visible(False)

    # ===== Panel B: Permutation Null Distribution =====
    ax_perm = fig.add_subplot(gs[0, 1])

    null_means = perm_results["null_selected_means"]
    observed = perm_results["observed_selected_stat"]

    ax_perm.hist(null_means, bins=50, color=null_color, edgecolor="white", alpha=0.8, density=True)
    ax_perm.axvline(observed, color=sel_color, linewidth=2.5, linestyle="-", label=f"Observed = {observed:.4f}")

    # Add percentile annotation
    percentile = perm_results["percentile_rank"]
    p_value = perm_results["p_value_one_sided"]

    ax_perm.set_xlabel(f"Mean {metric_label} of Random Samples", fontsize=11)
    ax_perm.set_ylabel("Density", fontsize=11)
    ax_perm.set_title("B. Permutation Test: Null Distribution", fontsize=12, fontweight="bold")

    # Add text annotation
    textstr = f"Percentile: {percentile:.1f}%\np = {p_value:.2e}"
    props = dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray")
    ax_perm.text(0.95, 0.95, textstr, transform=ax_perm.transAxes, fontsize=10,
                 verticalalignment="top", horizontalalignment="right", bbox=props)
    ax_perm.legend(loc="upper left", fontsize=9)
    ax_perm.spines["top"].set_visible(False)
    ax_perm.spines["right"].set_visible(False)

    # ===== Panel C: Summary Statistics (spans bottom row) =====
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis("off")

    # Create formatted statistics table
    stats_text = [
        ("Metric", "Selected Motor", "Non-Selected Motor"),
        ("─" * 15, "─" * 15, "─" * 18),
        ("Count", f"{summary['selected_motor_count']:,}", f"{summary['nonselected_motor_count']:,}"),
        ("Mean", f"{summary['selected_mean']:.4f}", f"{summary['nonselected_mean']:.4f}"),
        ("Median", f"{summary['selected_median']:.4f}", f"{summary['nonselected_median']:.4f}"),
        ("Std Dev", f"{summary['selected_std']:.4f}", f"{summary['nonselected_std']:.4f}"),
        ("95% CI", f"[{summary['selected_ci_lower']:.4f}, {summary['selected_ci_upper']:.4f}]",
         f"[{summary['nonselected_ci_lower']:.4f}, {summary['nonselected_ci_upper']:.4f}]"),
        ("", "", ""),
        ("Effect Sizes & Tests", "", ""),
        ("─" * 15, "─" * 15, "─" * 18),
        ("Mean Difference", f"{summary['mean_diff']:.4f} ({summary['mean_diff_pct']:.1f}%)", ""),
        ("Cohen's d", f"{summary['cohens_d']:.3f}", _interpret_cohens_d(summary['cohens_d'])),
        ("Mann-Whitney p", f"{summary['mannwhitney_pvalue']:.2e}", ""),
        ("Permutation p", f"{summary['perm_p_value']:.2e}", ""),
        ("CI Overlap", "No" if summary['ci_no_overlap'] else "Yes", ""),
    ]

    table_text = "\n".join([f"{row[0]:<20} {row[1]:<20} {row[2]}" for row in stats_text])

    ax_stats.text(0.5, 0.95, table_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment="top", horizontalalignment="center",
                  fontfamily="monospace",
                  bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#dee2e6"))
    ax_stats.set_title("C. Statistical Summary", fontsize=12, fontweight="bold", loc="center")

    # Overall figure title
    fig.suptitle(
        "Selected Voxels Exhibit Lower Trial-to-Trial Variability\nWithin Motor Cortex",
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
    parser.add_argument("--num-resamples", type=int, default=5000, help="Number of control resamples for sanity checks.")
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
    stats_summary, resample_means = _compare_groups(
        selected_vals, motor_vals, rng, args.num_resamples
    )

    stats_summary.update({
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
        resample_means=resample_means,
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
        f"mean_diff={stats_summary['mean_diff']:.6f}, "
        f"t={stats_summary['t_stat']:.3f}, "
        f"p_one={stats_summary['p_one_sided']:.3g}, "
        f"d={stats_summary['cohens_d']:.3f}",
        flush=True,
    )

    # =========================================================================
    # NEW ANALYSIS: Motor Cortex Variability Comparison
    # Compare selected voxels WITHIN motor cortex vs non-selected motor voxels
    # This is the key analysis to demonstrate that optimization selects
    # low-variability voxels, not noise.
    # =========================================================================

    print("\n" + "=" * 70)
    print("MOTOR CORTEX VARIABILITY ANALYSIS")
    print("Comparing selected vs non-selected voxels WITHIN motor cortex")
    print("=" * 70)

    # Compute motor cortex comparison using standard deviation (more interpretable)
    use_std = True
    selected_motor_vals, nonselected_motor_vals, selected_motor_idx, nonselected_motor_idx = \
        _compute_motor_cortex_comparison(voxel_var_flat, selected_indices, motor_indices, use_std=use_std)

    if selected_motor_vals.size < 10:
        print(f"WARNING: Only {selected_motor_vals.size} selected voxels in motor cortex. "
              "Analysis may not be reliable.", flush=True)

    print(f"Selected voxels in motor cortex: {selected_motor_vals.size}")
    print(f"Non-selected voxels in motor cortex: {nonselected_motor_vals.size}")

    # Run comprehensive comparison
    num_permutations = args.num_resamples
    num_bootstrap = min(2000, args.num_resamples)

    motor_summary, motor_perm_results = _compare_motor_groups(
        selected_motor_vals, nonselected_motor_vals, rng, num_permutations, num_bootstrap
    )

    # Add metadata to summary
    motor_summary.update({
        "analysis_type": "motor_cortex_variability_comparison",
        "metric": "standard_deviation" if use_std else "variance",
        "run_list": runs,
        "trial_len": args.trial_len,
        "total_trials_used": int(total_used),
        "num_permutations": num_permutations,
        "num_bootstrap": num_bootstrap,
        "selected_motor_indices": selected_motor_idx.tolist(),
        "nonselected_motor_indices": nonselected_motor_idx.tolist(),
    })

    # Save motor cortex comparison results
    motor_summary_path = out_dir / f"{output_tag}_motor_variability_summary.json"
    with open(motor_summary_path, "w", encoding="utf-8") as handle:
        # Convert numpy arrays to lists for JSON serialization
        summary_for_json = {k: v for k, v in motor_summary.items()
                           if not isinstance(v, np.ndarray)}
        json.dump(summary_for_json, handle, indent=2)
    print(f"Saved motor cortex summary: {motor_summary_path}")

    # Create comprehensive multi-panel figure
    motor_fig_path = out_dir / f"{output_tag}_motor_variability_figure.png"
    _plot_comprehensive_figure(
        selected_motor_vals, nonselected_motor_vals,
        motor_summary, motor_perm_results,
        motor_fig_path, use_std=use_std
    )
    print(f"Saved motor cortex figure: {motor_fig_path}")

    # Save detailed data for reproducibility
    motor_data_path = out_dir / f"{output_tag}_motor_variability_data.npz"
    np.savez(
        motor_data_path,
        selected_motor_values=selected_motor_vals,
        nonselected_motor_values=nonselected_motor_vals,
        selected_motor_indices=selected_motor_idx,
        nonselected_motor_indices=nonselected_motor_idx,
        null_diffs=motor_perm_results["null_diffs"],
        null_selected_means=motor_perm_results["null_selected_means"],
    )
    print(f"Saved motor cortex data: {motor_data_path}")

    # Print summary
    print("\n" + "-" * 50)
    print("MOTOR CORTEX ANALYSIS RESULTS:")
    print("-" * 50)
    metric_name = "Std Dev" if use_std else "Variance"
    print(f"Selected motor voxels:     n={motor_summary['selected_motor_count']}, "
          f"mean {metric_name}={motor_summary['selected_mean']:.4f}")
    print(f"Non-selected motor voxels: n={motor_summary['nonselected_motor_count']}, "
          f"mean {metric_name}={motor_summary['nonselected_mean']:.4f}")
    print(f"Mean difference: {motor_summary['mean_diff']:.4f} ({motor_summary['mean_diff_pct']:.1f}%)")
    print(f"Cohen's d: {motor_summary['cohens_d']:.3f} {_interpret_cohens_d(motor_summary['cohens_d'])}")
    print(f"Mann-Whitney U p-value: {motor_summary['mannwhitney_pvalue']:.2e}")
    print(f"Permutation test p-value: {motor_summary['perm_p_value']:.2e}")
    print(f"Percentile rank in null: {motor_summary['perm_percentile']:.1f}%")
    print(f"95% CI overlap: {'No' if motor_summary['ci_no_overlap'] else 'Yes'}")
    print("-" * 50)

    if motor_summary['perm_p_value'] < 0.05:
        print("\n✓ CONCLUSION: Selected motor voxels show significantly lower "
              "trial-to-trial variability than non-selected motor voxels.")
        print("  This supports the claim that the optimization selects stable, "
              "reliable voxels rather than noise.")
    else:
        print("\n⚠ WARNING: The difference in variability is not statistically significant.")
        print("  Consider investigating potential issues with the optimization or data.")


if __name__ == "__main__":
    main()
