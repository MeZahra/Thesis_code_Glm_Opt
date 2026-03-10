#!/usr/bin/env python3
"""Compare selected voxels against anatomical non-selected voxels."""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None


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
            "Test whether selected voxels have lower trial-to-trial beta variance than "
            "anatomical non-selected voxels using the full per-run cleaned beta volumes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ containing the selected voxel indices.",
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
        default=Path("results/connectivity/prove_hypothesis"),
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
        raise RuntimeError(f"No non-zero voxels found in {anat_path}")
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
        raise RuntimeError("Selected voxel set is empty.")
    return selected_flat


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
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
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


def _accumulate_trial_stats(
    target_flat: np.ndarray,
    manifest_rows: list[ManifestRow],
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_voxels = int(target_flat.size)
    counts = np.zeros(n_voxels, dtype=np.int64)
    sums = np.zeros(n_voxels, dtype=np.float64)
    sumsq = np.zeros(n_voxels, dtype=np.float64)

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
            finite = np.isfinite(chunk)
            counts[start:stop] += np.sum(finite, axis=1, dtype=np.int64)
            if not np.all(finite):
                chunk = np.where(finite, chunk, 0.0)
            sums[start:stop] += np.sum(chunk, axis=1, dtype=np.float64)
            sumsq[start:stop] += np.sum(chunk * chunk, axis=1, dtype=np.float64)

    variance = np.full(n_voxels, np.nan, dtype=np.float64)
    valid = counts > 1
    if np.any(valid):
        mean = sums[valid] / counts[valid]
        centered_ss = sumsq[valid] - counts[valid] * np.square(mean)
        centered_ss = np.maximum(centered_ss, 0.0)
        variance[valid] = centered_ss / (counts[valid] - 1)
    return variance, counts


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
        raise RuntimeError("Cannot estimate density for an empty vector.")

    if gaussian_kde is None or values.size < 2 or np.allclose(values, values[0]):
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
    try:
        kde = gaussian_kde(sampled)
        return grid, kde.evaluate(grid)
    except Exception:
        hist, edges = np.histogram(values, bins=min(80, max(10, values.size // 20)), density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, hist


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


def _build_text_panel(
    selected_count_total: int,
    nonselected_count_total: int,
    selected_count_valid: int,
    nonselected_count_valid: int,
    selected_mean: float,
    nonselected_mean: float,
    p_lower: float,
    num_resamples: int,
    resample_replace: bool,
) -> str:
    lines = [
        "A Selected voxels are shifted toward lower trial-to-trial beta variance than anatomical non-selected voxels.",
        "Dashed lines mark group means.",
        "",
        "B Log10 transform reduces skew and makes the separation easier to inspect.",
        "",
        "C Bars show Selected / Non-selected prevalence below pooled percentile thresholds.",
        "Values above 1 indicate overrepresentation among low-variability voxels.",
        "",
        f"D The null in panel d comes from {num_resamples} size-matched resamples of non-selected voxels.",
        f"Observed mean variance: selected = {selected_mean:.4g}, non-selected = {nonselected_mean:.4g}.",
        f"One-sided resampling p(mean_non <= mean_sel) = {p_lower:.4g}.",
        "",
        (
            f"Selected voxels came directly from selected_voxel_indices.npz: {selected_count_valid} valid / "
            f"{selected_count_total} total."
        ),
        (
            f"Non-selected voxels were defined as non-zero MNI anatomical voxels not in the selected set: "
            f"{nonselected_count_valid} valid / {nonselected_count_total} total."
        ),
    ]
    if resample_replace:
        lines.extend(
            [
                "",
                "Panel d used replacement in the null resampling because the valid non-selected pool was smaller than the selected set.",
            ]
        )
    return "\n".join(textwrap.fill(line, width=48) if line else "" for line in lines)


def _plot_summary_figure(
    figure_path: Path,
    selected_variance: np.ndarray,
    nonselected_variance: np.ndarray,
    percentile_labels: list[float],
    prevalence_ratios: np.ndarray,
    resampled_means: np.ndarray,
    p_lower: float,
    num_resamples: int,
    selected_count_total: int,
    nonselected_count_total: int,
    resample_replace: bool,
    rng: np.random.Generator,
    kde_max_points: int,
) -> None:
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1.0, 1.0])

    ax_text = fig.add_subplot(gs[:, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[0, 2])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    selected_mean = float(np.mean(selected_variance))
    nonselected_mean = float(np.mean(nonselected_variance))

    log_eps = 1e-12
    positive = np.concatenate([selected_variance, nonselected_variance])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size:
        log_eps = max(float(np.min(positive)) * 0.5, log_eps)

    _plot_density_panel(
        ax_a,
        selected_variance,
        nonselected_variance,
        rng,
        kde_max_points,
        xlabel="Trial-to-Trial Variance",
        panel_label="a",
    )
    _plot_density_panel(
        ax_b,
        np.log10(np.clip(selected_variance, log_eps, None)),
        np.log10(np.clip(nonselected_variance, log_eps, None)),
        rng,
        kde_max_points,
        xlabel="Log10(Trial-to-Trial Variance)",
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
    ax_d.set_xlabel("Mean Variance (non-selected voxels)")
    ax_d.set_ylabel("Density")
    ax_d.text(-0.02, 1.03, "d", transform=ax_d.transAxes, fontsize=18, fontweight="bold")
    ax_d.legend(frameon=True, fontsize=9, loc="upper right")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    ax_text.axis("off")
    text_block = _build_text_panel(
        selected_count_total=selected_count_total,
        nonselected_count_total=nonselected_count_total,
        selected_count_valid=int(selected_variance.size),
        nonselected_count_valid=int(nonselected_variance.size),
        selected_mean=selected_mean,
        nonselected_mean=nonselected_mean,
        p_lower=p_lower,
        num_resamples=num_resamples,
        resample_replace=resample_replace,
    )
    ax_text.text(0.02, 0.98, text_block, va="top", ha="left", fontsize=16.5, linespacing=1.45)

    fig.suptitle("Selected voxels vs anatomical non-selected voxels: trial variability of beta values", fontsize=16, y=1.02)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.random_seed)
    brain_flat, anat_shape = _load_brain_flat_indices(args.anat_path)
    selected_flat = _load_selected_flat_indices(args.selected_indices_path, anat_shape)

    brain_selected_mask = np.isin(selected_flat, brain_flat, assume_unique=False)
    if not np.all(brain_selected_mask):
        dropped = int(np.count_nonzero(~brain_selected_mask))
        print(f"Warning: dropped {dropped} selected voxels outside the anatomical non-zero mask.", flush=True)
        selected_flat = selected_flat[brain_selected_mask]
    if selected_flat.size == 0:
        raise RuntimeError("No selected voxels remain inside the anatomical non-zero mask.")

    nonselected_flat = np.setdiff1d(brain_flat, selected_flat, assume_unique=False)
    if nonselected_flat.size == 0:
        raise RuntimeError("No anatomical non-selected voxels remain after removing the selected set.")

    manifest_rows = _load_manifest_rows(args.manifest_path)

    target_flat = np.concatenate([selected_flat, nonselected_flat]).astype(np.int64, copy=False)
    print(
        f"Selected voxels: {selected_flat.size} | anatomical non-selected voxels: {nonselected_flat.size} | "
        f"runs in manifest: {len(manifest_rows)}",
        flush=True,
    )
    variance, trial_counts = _accumulate_trial_stats(
        target_flat=target_flat,
        manifest_rows=manifest_rows,
        row_chunk_size=args.row_chunk_size,
    )

    n_selected = int(selected_flat.size)
    selected_variance_all = np.asarray(variance[:n_selected], dtype=np.float64)
    nonselected_variance_all = np.asarray(variance[n_selected:], dtype=np.float64)
    selected_counts_all = np.asarray(trial_counts[:n_selected], dtype=np.int64)
    nonselected_counts_all = np.asarray(trial_counts[n_selected:], dtype=np.int64)

    selected_valid = np.isfinite(selected_variance_all)
    nonselected_valid = np.isfinite(nonselected_variance_all)
    selected_variance = selected_variance_all[selected_valid]
    nonselected_variance = nonselected_variance_all[nonselected_valid]
    selected_counts = selected_counts_all[selected_valid]
    nonselected_counts = nonselected_counts_all[nonselected_valid]
    selected_flat_valid = selected_flat[selected_valid]
    nonselected_flat_valid = nonselected_flat[nonselected_valid]

    if selected_variance.size == 0:
        raise RuntimeError("Selected voxels have no finite variance estimates.")
    if nonselected_variance.size == 0:
        raise RuntimeError("Anatomical non-selected voxels have no finite variance estimates.")

    thresholds, prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_variance,
        nonselected_values=nonselected_variance,
        percentiles=list(args.percentile_thresholds),
    )
    resampled_means, resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_variance,
        sample_size=selected_variance.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    selected_mean = float(np.mean(selected_variance))
    p_lower = (1.0 + float(np.count_nonzero(resampled_means <= selected_mean))) / (1.0 + float(args.num_resamples))

    png_path = args.output_dir / f"{args.output_stem}.png"
    pdf_path = args.output_dir / f"{args.output_stem}.pdf"
    summary_json_path = args.output_dir / f"{args.output_stem}_summary.json"
    prevalence_csv_path = args.output_dir / f"{args.output_stem}_prevalence.csv"
    analysis_npz_path = args.output_dir / f"{args.output_stem}_analysis_data.npz"

    print(f"Saving figure to {png_path} and {pdf_path}", flush=True)
    _plot_summary_figure(
        figure_path=png_path,
        selected_variance=selected_variance,
        nonselected_variance=nonselected_variance,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=prevalence_ratios,
        resampled_means=resampled_means,
        p_lower=p_lower,
        num_resamples=int(args.num_resamples),
        selected_count_total=int(selected_flat.size),
        nonselected_count_total=int(nonselected_flat.size),
        resample_replace=bool(resample_replace),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
    )
    _plot_summary_figure(
        figure_path=pdf_path,
        selected_variance=selected_variance,
        nonselected_variance=nonselected_variance,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=prevalence_ratios,
        resampled_means=resampled_means,
        p_lower=p_lower,
        num_resamples=int(args.num_resamples),
        selected_count_total=int(selected_flat.size),
        nonselected_count_total=int(nonselected_flat.size),
        resample_replace=bool(resample_replace),
        rng=np.random.default_rng(args.random_seed),
        kde_max_points=int(args.kde_max_points),
    )

    summary = {
        "selected_indices_path": str(args.selected_indices_path),
        "anat_path": str(args.anat_path),
        "manifest_path": str(args.manifest_path),
        "selected_count_total": int(selected_flat.size),
        "nonselected_count_total": int(nonselected_flat.size),
        "selected_count_valid": int(selected_variance.size),
        "nonselected_count_valid": int(nonselected_variance.size),
        "selected_mean_variance": float(np.mean(selected_variance)),
        "nonselected_mean_variance": float(np.mean(nonselected_variance)),
        "selected_median_variance": float(np.median(selected_variance)),
        "nonselected_median_variance": float(np.median(nonselected_variance)),
        "selected_min_finite_trials": int(np.min(selected_counts)),
        "nonselected_min_finite_trials": int(np.min(nonselected_counts)),
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
        selected_variance=selected_variance.astype(np.float32, copy=False),
        nonselected_variance=nonselected_variance.astype(np.float32, copy=False),
        selected_finite_trial_counts=selected_counts.astype(np.int32, copy=False),
        nonselected_finite_trial_counts=nonselected_counts.astype(np.int32, copy=False),
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
