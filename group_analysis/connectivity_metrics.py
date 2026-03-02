#!/usr/bin/env python3
"""Compute seed-based connectivity changes across GVS conditions.

This script assumes connectivity beta files are already split by condition:
    results/connectivity/selected_beta_trials_<condition>.npy

Each file is expected to have shape (n_voxels, n_trials_for_condition). The
seed signal is defined as the mean beta across *all* selected voxels per trial.
Then each voxel's connectivity is Pearson correlation with that seed signal.

Outputs:
  - per-condition connectivity vectors (.npy)
  - per-condition summary CSV
  - per-condition NIfTI maps (optional)
  - sham-vs-GVS delta vectors and summary CSV
  - condition-map similarity matrix (CSV + heatmap PNG)
  - summary plots (PNG)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = "results/connectivity"
DEFAULT_OUTPUT_DIR = "results/connectivity"
DEFAULT_META_PATH = "results/connectivity/selected_voxel_indices.npz"
DEFAULT_REF_IMG = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"

FILE_PATTERN = "selected_beta_trials_*.npy"
FILE_PREFIX = "selected_beta_trials_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute condition-wise seed connectivity and sham-vs-GVS changes "
            "from pre-split beta files."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--meta-path", default=DEFAULT_META_PATH)
    parser.add_argument("--reference-img", default=DEFAULT_REF_IMG)
    parser.add_argument("--min-pairs", type=int, default=12, help="Min paired trials per voxel for Pearson r.")
    parser.add_argument(
        "--sham-label",
        default="sham",
        help="Condition label used as sham baseline (default: sham).",
    )
    parser.add_argument(
        "--max-voxels",
        type=int,
        default=None,
        help="Optional debug limit: use only first N voxels from each condition file.",
    )
    parser.add_argument(
        "--no-nifti",
        action="store_true",
        help="Skip NIfTI map export.",
    )
    return parser.parse_args()


def _condition_sort_key(label: str) -> Tuple[int, int | str]:
    label_str = str(label)
    if label_str.lower() == "sham":
        return (0, 0)
    match = re.match(r"^gvs(\d+)$", label_str, flags=re.IGNORECASE)
    if match:
        return (1, int(match.group(1)))
    return (2, label_str.lower())


def discover_condition_files(input_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for path in sorted(input_dir.glob(FILE_PATTERN)):
        name = path.name
        if not name.startswith(FILE_PREFIX):
            continue
        label = name[len(FILE_PREFIX) : -4]  # strip prefix and .npy
        if not label:
            continue
        mapping[label] = path
    return mapping


def _safe_pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return np.nan
    x0 = x[mask].astype(np.float64, copy=False)
    y0 = y[mask].astype(np.float64, copy=False)
    x0 -= x0.mean()
    y0 -= y0.mean()
    denom = np.sqrt(np.dot(x0, x0) * np.dot(y0, y0))
    if denom <= 0:
        return np.nan
    return float(np.dot(x0, y0) / denom)


def _vector_to_nifti(
    values: np.ndarray,
    coords_ijk: np.ndarray,
    ref_img: nib.Nifti1Image,
    out_path: Path,
) -> None:
    vol = np.full(ref_img.shape[:3], np.nan, dtype=np.float32)
    vol[tuple(coords_ijk.T)] = values.astype(np.float32, copy=False)
    out_img = nib.Nifti1Image(vol, ref_img.affine, ref_img.header)
    nib.save(out_img, str(out_path))


def compute_seed_connectivity(beta: np.ndarray, min_pairs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return voxel-wise r(seed, voxel), seed signal, and paired trial counts.

    Seed signal is the trial-wise mean across all voxels (ignoring NaNs).
    """
    n_voxels, n_trials = beta.shape

    # Trial-wise seed signal from all selected voxels.
    with np.errstate(invalid="ignore"):
        seed_signal = np.nanmean(beta, axis=0).astype(np.float64, copy=False)
    finite_seed = np.isfinite(seed_signal)
    if int(finite_seed.sum()) < min_pairs:
        raise RuntimeError(
            f"Seed signal has only {int(finite_seed.sum())} finite trials (< {min_pairs})."
        )

    # Use fast summary-statistic Pearson formulas with pairwise finite masking.
    valid_mask = np.isfinite(beta)
    if not np.all(finite_seed):
        valid_mask &= finite_seed[None, :]

    paired_counts = valid_mask.sum(axis=1).astype(np.int32, copy=False)

    x = np.nan_to_num(beta, nan=0.0, copy=True).astype(np.float64, copy=False)
    if not np.all(finite_seed):
        x[:, ~finite_seed] = 0.0

    y = np.where(finite_seed, seed_signal, 0.0).astype(np.float64, copy=False)

    sx = x.sum(axis=1)
    sy = valid_mask @ y
    sxx = np.square(x).sum(axis=1)
    syy = valid_mask @ np.square(y)
    sxy = x @ y

    n = paired_counts.astype(np.float64, copy=False)
    n_safe = np.maximum(n, 1.0)

    cov = sxy - (sx * sy / n_safe)
    varx = sxx - (sx * sx / n_safe)
    vary = syy - (sy * sy / n_safe)

    varx = np.maximum(varx, 0.0)
    vary = np.maximum(vary, 0.0)
    denom = np.sqrt(varx * vary)

    r = np.full(n_voxels, np.nan, dtype=np.float32)
    valid = (paired_counts >= int(min_pairs)) & (denom > 0)
    r_vals = cov[valid] / denom[valid]
    r[valid] = np.clip(r_vals, -1.0, 1.0).astype(np.float32, copy=False)

    return r, seed_signal.astype(np.float32, copy=False), paired_counts


def _plot_condition_means(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(summary_df.shape[0])
    ax.plot(x, summary_df["mean_r"].to_numpy(), marker="o", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["condition"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Mean voxel connectivity (r)")
    ax.set_xlabel("Condition")
    ax.set_title("Mean seed connectivity by condition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_delta_means(delta_df: pd.DataFrame, out_path: Path) -> None:
    if delta_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(delta_df.shape[0])
    ax.bar(x, delta_df["mean_delta_r"].to_numpy())
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(delta_df["condition"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Mean Δr (condition - sham)")
    ax.set_xlabel("Condition")
    ax.set_title("Connectivity change vs sham")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_map_similarity_heatmap(labels: List[str], matrix: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Condition map similarity (voxel-wise Pearson r)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_files = discover_condition_files(input_dir)
    if not condition_files:
        raise FileNotFoundError(
            f"No condition files found in {input_dir} with pattern {FILE_PATTERN}."
        )

    sham_label = str(args.sham_label)
    if sham_label not in condition_files:
        available = ", ".join(sorted(condition_files))
        raise FileNotFoundError(
            f"Sham condition '{sham_label}' not found. Available: {available}"
        )

    ordered_conditions = sorted(condition_files.keys(), key=_condition_sort_key)

    meta = np.load(str(Path(args.meta_path).resolve()), allow_pickle=False)
    coords_ijk = np.asarray(meta["selected_ijk"], dtype=np.int64)

    condition_maps: Dict[str, np.ndarray] = {}
    condition_summary_rows: List[dict] = []

    n_vox_reference = None
    n_conditions_skipped = 0

    for condition in ordered_conditions:
        beta_path = condition_files[condition]
        beta = np.asarray(np.load(beta_path, mmap_mode="r"), dtype=np.float32)
        if beta.ndim != 2:
            print(
                f"WARNING: skipping {condition} because file is not 2D: "
                f"{beta_path} shape={beta.shape}",
                flush=True,
            )
            n_conditions_skipped += 1
            continue
        if args.max_voxels is not None:
            beta = beta[: int(args.max_voxels), :]

        n_vox, n_trials = beta.shape
        if n_vox < 2 or n_trials < 3:
            print(
                f"WARNING: skipping {condition} because matrix is too small for "
                f"connectivity: shape={beta.shape}",
                flush=True,
            )
            n_conditions_skipped += 1
            continue

        if coords_ijk.shape[0] < n_vox:
            print(
                f"WARNING: truncating {condition} voxels from {n_vox} to "
                f"{coords_ijk.shape[0]} to match selected_ijk rows.",
                flush=True,
            )
            n_vox = int(coords_ijk.shape[0])
            beta = beta[:n_vox, :]

        if n_vox_reference is None:
            n_vox_reference = n_vox
        elif n_vox != n_vox_reference:
            print(
                f"WARNING: skipping {condition} due voxel-count mismatch "
                f"({n_vox} vs expected {n_vox_reference}).",
                flush=True,
            )
            n_conditions_skipped += 1
            continue

        try:
            conn_r, seed_signal, paired_counts = compute_seed_connectivity(
                beta=beta,
                min_pairs=int(args.min_pairs),
            )
        except Exception as exc:
            print(
                f"WARNING: skipping {condition} because connectivity computation failed: {exc}",
                flush=True,
            )
            n_conditions_skipped += 1
            continue
        condition_maps[condition] = conn_r

        map_path = output_dir / f"seed_connectivity_{condition}.npy"
        np.save(map_path, conn_r)

        seed_path = output_dir / f"seed_signal_{condition}.npy"
        np.save(seed_path, seed_signal)

        finite_mask = np.isfinite(conn_r)
        clipped = np.clip(conn_r[finite_mask].astype(np.float64, copy=False), -0.999999, 0.999999)
        fisher_z = np.arctanh(clipped) if clipped.size else np.array([], dtype=np.float64)

        row = {
            "condition": condition,
            "file": str(beta_path),
            "n_voxels": int(n_vox),
            "n_trials": int(n_trials),
            "n_valid_voxels": int(finite_mask.sum()),
            "n_valid_seed_trials": int(np.isfinite(seed_signal).sum()),
            "mean_pairs_per_voxel": float(np.mean(paired_counts)),
            "mean_r": float(np.nanmean(conn_r)),
            "median_r": float(np.nanmedian(conn_r)),
            "std_r": float(np.nanstd(conn_r)),
            "mean_fisher_z": float(np.mean(fisher_z)) if fisher_z.size else np.nan,
            "median_fisher_z": float(np.median(fisher_z)) if fisher_z.size else np.nan,
        }
        condition_summary_rows.append(row)
        print(
            f"[{condition}] vox={n_vox} trials={n_trials} "
            f"valid_vox={row['n_valid_voxels']} mean_r={row['mean_r']:.4f}",
            flush=True,
        )

    if not condition_maps:
        raise RuntimeError("No valid condition maps were computed.")
    if sham_label not in condition_maps:
        valid_labels = ", ".join(sorted(condition_maps.keys()))
        raise RuntimeError(
            f"Sham condition '{sham_label}' was skipped/invalid. Valid conditions: {valid_labels}"
        )

    summary_df = pd.DataFrame(condition_summary_rows)
    summary_csv = output_dir / "seed_connectivity_condition_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    sham_map = condition_maps[sham_label]
    delta_rows: List[dict] = []

    for condition in ordered_conditions:
        if condition == sham_label:
            continue
        delta = condition_maps[condition] - sham_map
        delta_path = output_dir / f"seed_connectivity_delta_{condition}_vs_{sham_label}.npy"
        np.save(delta_path, delta.astype(np.float32, copy=False))

        valid = np.isfinite(delta)
        delta_rows.append(
            {
                "condition": condition,
                "contrast": f"{condition}_minus_{sham_label}",
                "n_valid_voxels": int(valid.sum()),
                "mean_delta_r": float(np.nanmean(delta)),
                "median_delta_r": float(np.nanmedian(delta)),
                "std_delta_r": float(np.nanstd(delta)),
                "pct_positive_delta": float(100.0 * np.mean(delta[valid] > 0.0)) if np.any(valid) else np.nan,
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    delta_csv = output_dir / "seed_connectivity_delta_vs_sham_summary.csv"
    delta_df.to_csv(delta_csv, index=False)

    labels = ordered_conditions
    sim = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    for i, li in enumerate(labels):
        sim[i, i] = 1.0
        for j in range(i + 1, len(labels)):
            r = _safe_pearson_1d(condition_maps[li], condition_maps[labels[j]])
            sim[i, j] = r
            sim[j, i] = r
    sim_df = pd.DataFrame(sim, index=labels, columns=labels)
    sim_df.to_csv(output_dir / "seed_connectivity_condition_similarity.csv")

    _plot_condition_means(summary_df, output_dir / "seed_connectivity_mean_by_condition.png")
    _plot_delta_means(delta_df, output_dir / "seed_connectivity_delta_vs_sham_mean.png")
    _plot_map_similarity_heatmap(labels, sim, output_dir / "seed_connectivity_condition_similarity_heatmap.png")

    if not args.no_nifti:
        ref_img = nib.load(str(Path(args.reference_img).resolve()))
        coords_use = coords_ijk[: int(n_vox_reference)]
        for condition in ordered_conditions:
            _vector_to_nifti(
                condition_maps[condition],
                coords_use,
                ref_img,
                output_dir / f"seed_connectivity_{condition}.nii.gz",
            )
            if condition != sham_label:
                delta_arr = condition_maps[condition] - sham_map
                _vector_to_nifti(
                    delta_arr,
                    coords_use,
                    ref_img,
                    output_dir / f"seed_connectivity_delta_{condition}_vs_{sham_label}.nii.gz",
                )

    run_manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "conditions": ordered_conditions,
        "conditions_processed": sorted(condition_maps.keys(), key=_condition_sort_key),
        "conditions_skipped": int(n_conditions_skipped),
        "sham_label": sham_label,
        "min_pairs": int(args.min_pairs),
        "max_voxels": int(args.max_voxels) if args.max_voxels is not None else None,
        "nifti_exported": not bool(args.no_nifti),
        "condition_summary_csv": str(summary_csv),
        "delta_summary_csv": str(delta_csv),
    }
    with open(output_dir / "seed_connectivity_run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    print(f"Saved condition summary: {summary_csv}", flush=True)
    print(f"Saved delta summary: {delta_csv}", flush=True)


if __name__ == "__main__":
    main()
