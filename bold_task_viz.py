#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
GLMSINGLE_ROOT = REPO_ROOT / "GLMsingle"
if str(GLMSINGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(GLMSINGLE_ROOT))

from glmsingle.design.convolve_design import convolve_design


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


def _load_indices_npz(npz_path, volume_shape):
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
            raise ValueError(f"Cannot map 3D coords to flat indices with volume shape {volume_shape}.")
        flat = np.ravel_multi_index(indices.T, volume_shape)
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


def _plot_histogram_comparison(selected_vals, motor_vals, bins, out_path, selected_mean, motor_mean, title_note=""):
    """Plot two normalized histograms with shared bins."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.hist(selected_vals, bins=bins, density=True, alpha=0.6, color="crimson",
            label=f"Selected (n={selected_vals.size})")
    ax.hist(motor_vals, bins=bins, density=True, alpha=0.6, color="navy",
            label=f"Motor-only (n={motor_vals.size})")
    ax.axvline(selected_mean, color="crimson", linestyle="--", linewidth=1)
    ax.axvline(motor_mean, color="navy", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean variance across trials")
    ax.set_ylabel("Density")
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
    parser = argparse.ArgumentParser(
        description=(
            "Compute predicted BOLD from GLMsingle outputs, estimate per-voxel trial variance, "
            "and compare selected vs motor voxel variability."
        )
    )
    parser.add_argument("--runs", default="1,2", help="Comma-separated run numbers (1-based).")
    parser.add_argument("--model-path", default="data/sub09_ses01/TYPED_FITHRF_GLMDENOISE_RR.npy")
    parser.add_argument("--design-path", default="data/sub09_ses01/DESIGNINFO.npy")
    parser.add_argument("--out-dir", default="data/sub09_ses01")
    parser.add_argument("--output-prefix", default="voxel_weights_mean_sub09_ses01",
                        help="Prefix used to locate indices files and name outputs.")
    parser.add_argument("--trial-len", type=int, default=9, help="Number of TRs per trial segment.")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Voxel chunk size for matrix multiplies.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for resampling.")
    parser.add_argument("--num-resamples", type=int, default=5000,
                        help="Number of control resamples for sanity checks.")
    parser.add_argument("--hist-bins", type=int, default=30, help="Histogram bin count.")
    parser.add_argument("--hist-subsample", dest="hist_subsample", action="store_true", default=True,
                        help="Subsample motor-only voxels for histogram to match selected count.")
    parser.add_argument("--no-hist-subsample", dest="hist_subsample", action="store_false",
                        help="Do not subsample motor-only voxels for histogram.")
    args = parser.parse_args()

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

    motor_indices = _validate_indices(_load_indices_npz(motor_path, volume_shape), voxels, "Motor")
    selected_indices = _validate_indices(_load_indices_npz(selected_path, volume_shape), voxels, "Selected")

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
        hist_note = "Motor-only histogram subsampled to match selected count."

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


if __name__ == "__main__":
    main()
