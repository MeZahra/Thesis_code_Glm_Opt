#!/usr/bin/env python3
"""Compare selected voxels against anatomical non-selected voxels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from scipy.stats import gaussian_kde


SELECTED_COLOR = "#e74c5b"
NONSELECTED_COLOR = "#4c84a6"
NULL_COLOR = "#b8dfe3"


@dataclass
class ManifestRow:
    cleaned_beta: Path
    trial_keep_path: Path | None
    n_trials_source: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether selected voxels have lower mean absolute consecutive-trial beta "
            "difference than anatomical non-selected voxels using the full per-run cleaned beta volumes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ containing the selected voxel indices. Ignored when --selected-csv-path is provided.",
    )
    parser.add_argument(
        "--selected-csv-path",
        type=Path,
        default=None,
        help="CSV with x,y,z selected voxel coordinates in anatomy index space.",
    )
    parser.add_argument(
        "--anat-path",
        type=Path,
        default=Path("results/connectivity/data/MNI152_T1_2mm_brain.nii.gz"),
        help="MNI anatomy used to define the anatomical brain voxel pool.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv"),
        help="Group-concat manifest with cleaned_beta paths and trial_keep paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prove_hypothesis"),
        help="Directory for figure and summary outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default="trial_variability_hypothesis",
        help="Base filename for outputs.",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=20000,
        help="Number of flat voxels to process at once from each 4D beta volume.",
    )
    parser.add_argument(
        "--num-resamples",
        type=int,
        default=1000,
        help="Number of null resamples from the non-selected pool.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed for null resampling and KDE downsampling.",
    )
    parser.add_argument(
        "--kde-max-points",
        type=int,
        default=20000,
        help="Maximum values per group used to estimate KDE curves.",
    )
    parser.add_argument(
        "--percentile-thresholds",
        type=float,
        nargs="+",
        default=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        help="Percentiles used in the relative-prevalence bar plot.",
    )
    return parser.parse_args()


def _load_brain_flat_indices(anat_path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    anat_img = nib.load(str(anat_path))
    anat_data = anat_img.get_fdata(dtype=np.float32)
    brain_mask = np.isfinite(anat_data) & (anat_data != 0)
    brain_flat = np.flatnonzero(brain_mask.ravel()).astype(np.int64, copy=False)
    if brain_flat.size == 0:
        raise ValueError(f"No non-zero voxels found in {anat_path}")
    return brain_flat, tuple(int(dim) for dim in anat_img.shape[:3])


def _load_selected_flat_indices(selected_indices_path: Path, anat_shape: tuple[int, int, int]) -> np.ndarray:
    selected_pack = np.load(selected_indices_path, allow_pickle=True)
    if "selected_flat_indices" in selected_pack.files:
        selected_flat = np.asarray(selected_pack["selected_flat_indices"], dtype=np.int64).ravel()
    elif "selected_ijk" in selected_pack.files:
        selected_ijk = np.asarray(selected_pack["selected_ijk"], dtype=np.int64)
        if selected_ijk.ndim != 2 or selected_ijk.shape[1] != 3:
            raise ValueError(f"Expected selected_ijk shape (N, 3), got {selected_ijk.shape}")
        selected_flat = np.ravel_multi_index(selected_ijk.T, dims=anat_shape).astype(np.int64, copy=False)
    else:
        raise KeyError(
            "selected_voxel_indices.npz must contain 'selected_flat_indices' or 'selected_ijk'."
        )
    selected_flat = np.unique(selected_flat)
    if selected_flat.size == 0:
        raise ValueError("Selected voxel set is empty.")
    return selected_flat


def _load_selected_flat_indices_from_csv(csv_path: Path, anat_shape: tuple[int, int, int]) -> np.ndarray:
    coords = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.int64)
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected CSV coordinates shape (N, 3), got {coords.shape}")

    valid = np.ones(coords.shape[0], dtype=bool)
    for dim, size in enumerate(anat_shape):
        valid &= (coords[:, dim] >= 0) & (coords[:, dim] < size)
    if not np.all(valid):
        dropped = int(np.count_nonzero(~valid))
        print(f"Warning: dropped {dropped} CSV coordinates outside anatomy bounds.", flush=True)
        coords = coords[valid]
    if coords.size == 0:
        raise ValueError("CSV selected voxel set is empty after bounds checking.")

    selected_flat = np.ravel_multi_index(coords.T, dims=anat_shape).astype(np.int64, copy=False)
    return np.unique(selected_flat)


def _load_manifest_rows(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            cleaned_beta = Path(str(row["cleaned_beta"]).strip())
            if not cleaned_beta.exists():
                raise FileNotFoundError(f"Missing cleaned beta volume: {cleaned_beta}")
            trial_keep_text = str(row.get("trial_keep_path", "") or "").strip()
            trial_keep_path = Path(trial_keep_text) if trial_keep_text else None
            n_trials_source = int(row.get("n_trials_source", 0) or 0)
            rows.append(
                ManifestRow(
                    cleaned_beta=cleaned_beta,
                    trial_keep_path=trial_keep_path,
                    n_trials_source=n_trials_source,
                )
            )
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def _load_trial_keep_mask(row: ManifestRow, n_trials_source: int) -> np.ndarray:
    if row.trial_keep_path is None:
        return np.ones(n_trials_source, dtype=bool)
    keep = np.asarray(np.load(row.trial_keep_path), dtype=bool).ravel()
    if keep.size != n_trials_source:
        raise ValueError(
            f"trial_keep length mismatch for {row.trial_keep_path}: {keep.size} vs {n_trials_source}"
        )
    return keep


def _accumulate_consecutive_diff_metric(
    target_flat: np.ndarray,
    manifest_rows: list[ManifestRow],
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_voxels = int(target_flat.size)
    pair_counts = np.zeros(n_voxels, dtype=np.int64)
    abs_diff_sums = np.zeros(n_voxels, dtype=np.float64)

    row_chunk_size = max(1, int(row_chunk_size))
    total_runs = len(manifest_rows)

    for run_idx, row in enumerate(manifest_rows, start=1):
        volume = np.load(row.cleaned_beta, mmap_mode="r")
        if volume.ndim != 4:
            raise ValueError(f"Expected 4D cleaned beta volume, got {volume.shape} for {row.cleaned_beta}")

        flat_view = volume.reshape(-1, volume.shape[-1])
        n_trials_source = int(volume.shape[-1])
        if row.n_trials_source and row.n_trials_source != n_trials_source:
            raise ValueError(
                f"Manifest n_trials_source mismatch for {row.cleaned_beta}: "
                f"{row.n_trials_source} vs {n_trials_source}"
            )
        keep_mask = _load_trial_keep_mask(row, n_trials_source)
        print(
            f"Run {run_idx}/{total_runs}: {row.cleaned_beta.name} | kept trials = {int(np.count_nonzero(keep_mask))}",
            flush=True,
        )

        for start in range(0, n_voxels, row_chunk_size):
            stop = min(start + row_chunk_size, n_voxels)
            chunk = np.asarray(flat_view[target_flat[start:stop]][:, keep_mask], dtype=np.float32)
            if chunk.shape[1] < 2:
                continue
            prev_vals = chunk[:, :-1]
            next_vals = chunk[:, 1:]
            valid_pairs = np.isfinite(prev_vals) & np.isfinite(next_vals)
            pair_counts[start:stop] += np.sum(valid_pairs, axis=1, dtype=np.int64)
            abs_diff = np.abs(next_vals - prev_vals)
            if not np.all(valid_pairs):
                abs_diff = np.where(valid_pairs, abs_diff, 0.0)
            abs_diff_sums[start:stop] += np.sum(abs_diff, axis=1, dtype=np.float64)

    metric = np.full(n_voxels, np.nan, dtype=np.float64)
    valid = pair_counts > 0
    if np.any(valid):
        metric[valid] = abs_diff_sums[valid] / pair_counts[valid]
    return metric, pair_counts


def _subsample_for_kde(values: np.ndarray, kde_max_points: int, rng: np.random.Generator) -> np.ndarray:
    if values.size <= kde_max_points:
        return values
    chosen = rng.choice(values.size, size=int(kde_max_points), replace=False)
    return np.asarray(values[chosen], dtype=np.float64)


def _density_curve(
    values: np.ndarray,
    rng: np.random.Generator,
    kde_max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate density for an empty vector.")

    if values.size < 2 or np.allclose(values, values[0]):
        hist, edges = np.histogram(values, bins=min(80, max(10, values.size // 20)), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, hist

    sampled = _subsample_for_kde(values, kde_max_points, rng)
    lower = float(np.min(sampled))
    upper = float(np.max(sampled))
    if np.isclose(lower, upper):
        lower -= 1e-6
        upper += 1e-6
    pad = 0.05 * (upper - lower)
    grid = np.linspace(lower - pad, upper + pad, 512)
    kde = gaussian_kde(sampled)
    return grid, kde.evaluate(grid)


def _plot_density_panel(
    ax: plt.Axes,
    selected_values: np.ndarray,
    nonselected_values: np.ndarray,
    rng: np.random.Generator,
    kde_max_points: int,
    xlabel: str,
    panel_label: str,
) -> None:
    xs_sel, ys_sel = _density_curve(selected_values, rng, kde_max_points)
    xs_non, ys_non = _density_curve(nonselected_values, rng, kde_max_points)

    ax.fill_between(xs_sel, ys_sel, color=SELECTED_COLOR, alpha=0.28)
    ax.plot(xs_sel, ys_sel, color=SELECTED_COLOR, linewidth=2.0, label=f"Selected (n={selected_values.size})")
    ax.fill_between(xs_non, ys_non, color=NONSELECTED_COLOR, alpha=0.24)
    ax.plot(xs_non, ys_non, color=NONSELECTED_COLOR, linewidth=2.0, label=f"Non-selected (n={nonselected_values.size})")

    sel_mean = float(np.mean(selected_values))
    non_mean = float(np.mean(nonselected_values))
    ax.axvline(sel_mean, linestyle="--", linewidth=1.6, color=SELECTED_COLOR)
    ax.axvline(non_mean, linestyle="--", linewidth=1.6, color=NONSELECTED_COLOR)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(frameon=True, fontsize=9, loc="upper right")
    ax.text(-0.02, 1.03, panel_label, transform=ax.transAxes, fontsize=18, fontweight="bold")
    ax.text(
        0.98,
        0.83,
        f"mean_sel = {sel_mean:.4g}\nmean_non = {non_mean:.4g}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compute_prevalence_ratios(
    selected_values: np.ndarray,
    nonselected_values: np.ndarray,
    percentiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    pooled = np.concatenate([selected_values, nonselected_values]).astype(np.float64, copy=False)
    thresholds = np.percentile(pooled, np.asarray(percentiles, dtype=np.float64))
    selected_prev = np.array([(selected_values <= thr).mean() for thr in thresholds], dtype=np.float64)
    nonselected_prev = np.array([(nonselected_values <= thr).mean() for thr in thresholds], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(
            selected_prev,
            nonselected_prev,
            out=np.full_like(selected_prev, np.nan),
            where=nonselected_prev > 0,
        )
    return thresholds, ratios


def _resample_nonselected_means(
    nonselected_values: np.ndarray,
    sample_size: int,
    num_resamples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool]:
    replace = int(sample_size) > int(nonselected_values.size)
    out = np.empty(int(num_resamples), dtype=np.float64)
    for idx in range(int(num_resamples)):
        chosen = rng.choice(nonselected_values.size, size=int(sample_size), replace=replace)
        out[idx] = float(np.mean(nonselected_values[chosen]))
    return out, replace


def _plot_summary_figure(
    figure_path: Path,
    selected_metric: np.ndarray,
    nonselected_metric: np.ndarray,
    percentile_labels: list[float],
    prevalence_ratios: np.ndarray,
    resampled_means: np.ndarray,
    p_lower: float,
    num_resamples: int,
    rng: np.random.Generator,
    kde_max_points: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    selected_mean = float(np.mean(selected_metric))
    nonselected_mean = float(np.mean(nonselected_metric))

    log_eps = 1e-12
    positive = np.concatenate([selected_metric, nonselected_metric])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size:
        log_eps = max(float(np.min(positive)) * 0.5, log_eps)

    _plot_density_panel(
        ax_a,
        selected_metric,
        nonselected_metric,
        rng,
        kde_max_points,
        xlabel="Mean |Delta| Consecutive Trials",
        panel_label="a",
    )
    _plot_density_panel(
        ax_b,
        np.log10(np.clip(selected_metric, log_eps, None)),
        np.log10(np.clip(nonselected_metric, log_eps, None)),
        rng,
        kde_max_points,
        xlabel="Log10(Mean |Delta| Consecutive Trials)",
        panel_label="b",
    )

    colors = [SELECTED_COLOR if value >= 1.0 else NONSELECTED_COLOR for value in prevalence_ratios]
    tick_labels = [f"{int(round(p))}%" for p in percentile_labels]
    ax_c.bar(tick_labels, prevalence_ratios, color=colors, edgecolor="white", linewidth=1.0)
    ax_c.axhline(1.0, linestyle="--", color="0.5", linewidth=1.6)
    ax_c.set_ylabel("Relative Prevalence\n(Selected / Non-selected)")
    ax_c.set_xlabel("Variability Percentile Threshold")
    ax_c.text(-0.02, 1.03, "c", transform=ax_c.transAxes, fontsize=18, fontweight="bold")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    ax_d.hist(resampled_means, bins=40, density=True, color=NULL_COLOR, edgecolor="white", alpha=0.95)
    resample_mean = float(np.mean(resampled_means))
    ci_low, ci_high = np.percentile(resampled_means, [2.5, 97.5])
    ax_d.axvline(selected_mean, linestyle="--", linewidth=2.0, color=SELECTED_COLOR, label=f"Selected mean = {selected_mean:.4g}")
    ax_d.axvline(resample_mean, linestyle="--", linewidth=1.6, color="0.45", label=f"Resample mean = {resample_mean:.4g}")
    ax_d.axvline(ci_low, linestyle="--", linewidth=1.4, color="0.55")
    ax_d.axvline(ci_high, linestyle="--", linewidth=1.4, color="0.55", label=f"95% CI = [{ci_low:.4g}, {ci_high:.4g}]")
    ax_d.set_xlabel("Mean |Delta| Consecutive Trials (non-selected voxels)")
    ax_d.set_ylabel("Density")
    ax_d.text(-0.02, 1.03, "d", transform=ax_d.transAxes, fontsize=18, fontweight="bold")
    ax_d.legend(frameon=True, fontsize=9, loc="upper right")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        (
            "Selected vs anatomical non-selected voxels | "
            f"mean |Delta|: {selected_mean:.3f} vs {nonselected_mean:.3f} | "
            f"p={p_lower:.4g} | resamples={num_resamples}"
        ),
        fontsize=14,
        y=1.02,
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    brain_flat, anat_shape = _load_brain_flat_indices(args.anat_path)
    if args.selected_csv_path is not None:
        print(f"Loading selected voxels from CSV: {args.selected_csv_path}", flush=True)
        selected_flat = _load_selected_flat_indices_from_csv(args.selected_csv_path, anat_shape)
        selected_source = str(args.selected_csv_path)
        selected_source_type = "csv"
    else:
        print(f"Loading selected voxels from NPZ: {args.selected_indices_path}", flush=True)
        selected_flat = _load_selected_flat_indices(args.selected_indices_path, anat_shape)
        selected_source = str(args.selected_indices_path)
        selected_source_type = "npz"

    brain_selected_mask = np.isin(selected_flat, brain_flat, assume_unique=False)
    if not np.all(brain_selected_mask):
        dropped = int(np.count_nonzero(~brain_selected_mask))
        print(f"Warning: dropped {dropped} selected voxels outside the anatomical non-zero mask.", flush=True)
        selected_flat = selected_flat[brain_selected_mask]
    if selected_flat.size == 0:
        raise ValueError("No selected voxels remain inside the anatomical non-zero mask.")

    nonselected_flat = np.setdiff1d(brain_flat, selected_flat, assume_unique=False)
    if nonselected_flat.size == 0:
        raise ValueError("No anatomical non-selected voxels remain after removing the selected set.")

    manifest_rows = _load_manifest_rows(args.manifest_path)

    target_flat = np.concatenate([selected_flat, nonselected_flat]).astype(np.int64, copy=False)
    print(
        f"Selected voxels: {selected_flat.size} | anatomical non-selected voxels: {nonselected_flat.size} | "
        f"runs in manifest: {len(manifest_rows)}",
        flush=True,
    )
    metric_values, pair_counts = _accumulate_consecutive_diff_metric(
        target_flat=target_flat,
        manifest_rows=manifest_rows,
        row_chunk_size=args.row_chunk_size,
    )

    n_selected = int(selected_flat.size)
    selected_metric_all = np.asarray(metric_values[:n_selected], dtype=np.float64)
    nonselected_metric_all = np.asarray(metric_values[n_selected:], dtype=np.float64)
    selected_counts_all = np.asarray(pair_counts[:n_selected], dtype=np.int64)
    nonselected_counts_all = np.asarray(pair_counts[n_selected:], dtype=np.int64)

    selected_valid = np.isfinite(selected_metric_all)
    nonselected_valid = np.isfinite(nonselected_metric_all)
    selected_metric = selected_metric_all[selected_valid]
    nonselected_metric = nonselected_metric_all[nonselected_valid]
    selected_counts = selected_counts_all[selected_valid]
    nonselected_counts = nonselected_counts_all[nonselected_valid]
    selected_flat_valid = selected_flat[selected_valid]
    nonselected_flat_valid = nonselected_flat[nonselected_valid]

    if selected_metric.size == 0:
        raise ValueError("Selected voxels have no finite consecutive-trial difference estimates.")
    if nonselected_metric.size == 0:
        raise ValueError("Anatomical non-selected voxels have no finite consecutive-trial difference estimates.")

    thresholds, prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_metric,
        nonselected_values=nonselected_metric,
        percentiles=list(args.percentile_thresholds),
    )
    resampled_means, resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_metric,
        sample_size=selected_metric.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    selected_mean = float(np.mean(selected_metric))
    p_lower = (1.0 + float(np.count_nonzero(resampled_means <= selected_mean))) / (1.0 + float(args.num_resamples))

    png_path = args.output_dir / f"{args.output_stem}.png"
    pdf_path = args.output_dir / f"{args.output_stem}.pdf"
    summary_json_path = args.output_dir / f"{args.output_stem}_summary.json"
    prevalence_csv_path = args.output_dir / f"{args.output_stem}_prevalence.csv"
    analysis_npz_path = args.output_dir / f"{args.output_stem}_analysis_data.npz"

    print(f"Saving figure to {png_path} and {pdf_path}", flush=True)
    _plot_summary_figure(
        figure_path=png_path,
        selected_metric=selected_metric,
        nonselected_metric=nonselected_metric,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=prevalence_ratios,
        resampled_means=resampled_means,
        p_lower=p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
    )
    _plot_summary_figure(
        figure_path=pdf_path,
        selected_metric=selected_metric,
        nonselected_metric=nonselected_metric,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=prevalence_ratios,
        resampled_means=resampled_means,
        p_lower=p_lower,
        num_resamples=int(args.num_resamples),
        rng=np.random.default_rng(args.random_seed),
        kde_max_points=int(args.kde_max_points),
    )

    summary = {
        "selected_source": selected_source,
        "selected_source_type": selected_source_type,
        "selected_indices_path": str(args.selected_indices_path),
        "selected_csv_path": str(args.selected_csv_path) if args.selected_csv_path is not None else None,
        "anat_path": str(args.anat_path),
        "manifest_path": str(args.manifest_path),
        "selected_count_total": int(selected_flat.size),
        "nonselected_count_total": int(nonselected_flat.size),
        "selected_count_valid": int(selected_metric.size),
        "nonselected_count_valid": int(nonselected_metric.size),
        "selected_mean_abs_consecutive_trial_diff": float(np.mean(selected_metric)),
        "nonselected_mean_abs_consecutive_trial_diff": float(np.mean(nonselected_metric)),
        "selected_median_abs_consecutive_trial_diff": float(np.median(selected_metric)),
        "nonselected_median_abs_consecutive_trial_diff": float(np.median(nonselected_metric)),
        "selected_min_finite_consecutive_pairs": int(np.min(selected_counts)),
        "nonselected_min_finite_consecutive_pairs": int(np.min(nonselected_counts)),
        "resample_mean": float(np.mean(resampled_means)),
        "resample_std": float(np.std(resampled_means, ddof=1)),
        "resample_ci_2p5": float(np.percentile(resampled_means, 2.5)),
        "resample_ci_97p5": float(np.percentile(resampled_means, 97.5)),
        "resample_p_lower_or_equal_selected": float(p_lower),
        "resample_with_replacement": bool(resample_replace),
        "num_resamples": int(args.num_resamples),
        "random_seed": int(args.random_seed),
        "png_path": str(png_path),
        "pdf_path": str(pdf_path),
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with prevalence_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["percentile", "threshold", "relative_prevalence_selected_over_nonselected"])
        for percentile, threshold, ratio in zip(args.percentile_thresholds, thresholds, prevalence_ratios):
            writer.writerow([float(percentile), float(threshold), float(ratio)])

    np.savez_compressed(
        analysis_npz_path,
        selected_flat_indices=selected_flat_valid.astype(np.int64, copy=False),
        nonselected_flat_indices=nonselected_flat_valid.astype(np.int64, copy=False),
        selected_mean_abs_consecutive_trial_diff=selected_metric.astype(np.float32, copy=False),
        nonselected_mean_abs_consecutive_trial_diff=nonselected_metric.astype(np.float32, copy=False),
        selected_finite_consecutive_pair_counts=selected_counts.astype(np.int32, copy=False),
        nonselected_finite_consecutive_pair_counts=nonselected_counts.astype(np.int32, copy=False),
        prevalence_thresholds=thresholds.astype(np.float32, copy=False),
        prevalence_ratios=prevalence_ratios.astype(np.float32, copy=False),
        resampled_nonselected_means=resampled_means.astype(np.float32, copy=False),
    )

    print(f"Saved summary JSON: {summary_json_path}", flush=True)
    print(f"Saved prevalence CSV: {prevalence_csv_path}", flush=True)
    print(f"Saved analysis NPZ: {analysis_npz_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
