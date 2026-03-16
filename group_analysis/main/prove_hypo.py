#!/usr/bin/env python3
"""Compare selected voxels against motor-area non-selected voxels."""

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
FSL_CEREBELLUM_MAXPROB_2MM = Path("/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz")
DEFAULT_MOTOR_LABEL_PATTERNS = [
    "precentral gyrus",
    "juxtapositional lobule cortex",
    "supplementary motor",
    "precentral",
    "postcentral gyrus",
    "frontal medial cortex",
    "paracentral lobule",
    "thalamus",
    "caudate nucleus",
    "putamen",
    "globus pallidus",
    "pallidum",
    "cerebellum",
]


@dataclass
class ManifestRow:
    sub_tag: str
    ses: int
    cleaned_beta: Path
    trial_keep_path: Path | None
    n_trials_source: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether selected voxels differ from motor-area non-selected voxels in two metrics: "
            "mean absolute consecutive-trial beta difference and within-voxel trial-to-trial variance."
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
        default=Path("results/connectivity/tmp/data/MNI152_T1_2mm_brain.nii.gz"),
        help="MNI anatomy used to define the brain/motor voxel pool.",
    )
    parser.add_argument(
        "--motor-mask-path",
        type=Path,
        default=None,
        help=(
            "Optional NIfTI mask for motor areas. Non-zero values define candidate non-selected voxels. "
            "If not provided, a mask is built from Harvard-Oxford cortical and subcortical atlases."
        ),
    )
    parser.add_argument(
        "--motor-label-patterns",
        type=str,
        default=",".join(DEFAULT_MOTOR_LABEL_PATTERNS),
        help="Comma-separated motor-region label substrings used when building atlas-based motor mask.",
    )
    parser.add_argument(
        "--motor-atlas-cache-dir",
        type=Path,
        default=Path("results/connectivity/atlas_cache"),
        help="Cache directory for Harvard-Oxford atlas downloads (if needed).",
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


def _normalize_labels(labels) -> list[str]:
    out: list[str] = []
    for value in list(labels):
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _split_csv_patterns(raw: str | None) -> list[str]:
    if raw is None:
        return []
    parts = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            parts.append(item.lower())
    return parts


def _select_label_ids(labels: list[str], include: list[str], exclude: list[str] | None = None) -> tuple[list[int], list[str]]:
    include_lower = [item.lower() for item in include]
    exclude_lower = [item.lower() for item in (exclude or [])]
    selected_ids: list[int] = []
    selected_names: list[str] = []

    for idx, raw_label in enumerate(labels):
        if idx == 0:
            continue
        label = raw_label.lower()
        if not any(p in label for p in include_lower):
            continue
        if any(p in label for p in exclude_lower):
            continue
        selected_ids.append(idx)
        selected_names.append(raw_label)

    return selected_ids, selected_names


def _load_motor_mask_from_atlas(
    anat_path: Path,
    label_patterns: list[str],
    atlas_cache_dir: Path,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    try:
        from nilearn import datasets, image
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Cannot build motor atlas: nilearn is required when --motor-mask-path is not set. "
            "Either install nilearn or pass --motor-mask-path directly."
        ) from exc

    anat_img = nib.load(str(anat_path))
    cortex = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm",
        data_dir=str(atlas_cache_dir),
    )
    subcortex = datasets.fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-2mm",
        data_dir=str(atlas_cache_dir),
    )

    cortical_img = cortex.maps if hasattr(cortex.maps, "get_fdata") else nib.load(cortex.maps)
    subcortical_img = subcortex.maps if hasattr(subcortex.maps, "get_fdata") else nib.load(subcortex.maps)
    cortical_data = image.resample_to_img(cortical_img, anat_img, interpolation="nearest", force_resample=True).get_fdata()
    subcortical_data = image.resample_to_img(subcortical_img, anat_img, interpolation="nearest", force_resample=True).get_fdata()
    cortical_data = np.rint(cortical_data).astype(np.int32, copy=False)
    subcortical_data = np.rint(subcortical_data).astype(np.int32, copy=False)

    cortical_labels = _normalize_labels(cortex.labels)
    subcortical_labels = _normalize_labels(subcortex.labels)

    include_patterns = [p for p in (label_patterns or []) if p]
    include_lower = [p.lower() for p in include_patterns]

    cortical_ids, cortical_names = _select_label_ids(cortical_labels, include_lower)
    subcortical_ids, subcortical_names = _select_label_ids(subcortical_labels, include_lower)

    region_names: list[str] = []
    region_counts: list[int] = []
    motor_mask = np.zeros(anat_img.shape[:3], dtype=bool)

    if cortical_ids:
        for lid, name in zip(cortical_ids, cortical_names):
            region = cortical_data == int(lid)
            region_count = int(np.count_nonzero(region))
            if region_count == 0:
                continue
            motor_mask |= region
            region_names.append(f"Cortical: {name}")
            region_counts.append(region_count)

    if subcortical_ids:
        for lid, name in zip(subcortical_ids, subcortical_names):
            region = subcortical_data == int(lid)
            region_count = int(np.count_nonzero(region))
            if region_count == 0:
                continue
            motor_mask |= region
            region_names.append(f"Subcortical: {name}")
            region_counts.append(region_count)

    if any("cerebellum" in item for item in include_lower) and FSL_CEREBELLUM_MAXPROB_2MM.exists():
        cereb_img = nib.load(str(FSL_CEREBELLUM_MAXPROB_2MM))
        cereb_data = image.resample_to_img(cereb_img, anat_img, interpolation="nearest", force_resample=True).get_fdata()
        region = np.asarray(cereb_data > 0, dtype=bool)
        region_count = int(np.count_nonzero(region))
        if region_count > 0:
            motor_mask |= region
            region_names.append("Cerebellum (FSL maxprob)")
            region_counts.append(region_count)

    if not np.any(motor_mask):
        raise ValueError("Motor atlas mask has no voxels. Try widening --motor-label-patterns or use --motor-mask-path.")

    flat_region_counts = np.array(region_counts, dtype=np.int64)
    return (
        np.flatnonzero(motor_mask),
        region_names,
        flat_region_counts.tolist(),
        sorted(set(include_lower)),
    )


def _load_motor_flat_indices(
    motor_mask_path: Path | None,
    anat_path: Path,
    label_patterns: list[str],
    atlas_cache_dir: Path,
) -> tuple[np.ndarray, list[str], list[int], list[str]]:
    anat_img = nib.load(str(anat_path))
    anat_shape = anat_img.shape[:3]

    if motor_mask_path is not None:
        mask_img = nib.load(str(motor_mask_path))
        if tuple(mask_img.shape[:3]) != anat_shape or not np.allclose(mask_img.affine, anat_img.affine):
            try:
                from nilearn import image as nilearn_image
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "--motor-mask-path shape differs from anatomy and nilearn is not installed for resampling. "
                    "Install nilearn or regenerate the motor mask in the same space as anatomy."
                ) from exc
            mask_img = nilearn_image.resample_to_img(mask_img, anat_img, interpolation="nearest", force_resample=True)

        mask_data = np.asarray(mask_img.get_fdata(), dtype=np.float32)
        motor_mask = np.isfinite(mask_data) & (np.abs(mask_data) > 0)
        motor_flat = np.flatnonzero(motor_mask.ravel())
        if motor_flat.size == 0:
            raise ValueError(f"No non-zero voxels in --motor-mask-path: {motor_mask_path}")
        return (
            motor_flat.astype(np.int64, copy=False),
            ["Provided motor mask: non-zero voxels"],
            [int(np.count_nonzero(motor_mask))],
            ["custom"],
        )

    motor_flat, region_names, region_counts, resolved_patterns = _load_motor_mask_from_atlas(
        anat_path=anat_path,
        label_patterns=label_patterns,
        atlas_cache_dir=atlas_cache_dir,
    )
    return motor_flat, region_names, region_counts, resolved_patterns


def _save_motor_region_figure(
    figure_path: Path,
    region_names: list[str],
    region_counts: list[int],
    selected_patterns: list[str],
) -> None:
    if len(region_names) == 0:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.02, 0.6, "No motor regions matched selected patterns", fontsize=12)
        ax.set_axis_off()
        fig.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    order = np.argsort(region_counts)[::-1]
    names = [region_names[i] for i in order]
    counts = [int(region_counts[i]) for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(names, counts, color="#4c84a6", edgecolor="white")
    ax.set_xlabel("Voxel count")
    if selected_patterns:
        title = "Motor regions selected by atlas patterns\n" + \
            f"patterns: {', '.join(selected_patterns)}"
    else:
        title = "Motor regions selected by atlas patterns"
    ax.set_title(title)
    ax.invert_yaxis()
    for y, value in enumerate(counts):
        ax.text(value + max(1.0, max(counts) * 0.02), y, f"{int(value)}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
            sub_tag = str(row.get("sub_tag", "") or "").strip()
            if not sub_tag:
                raise ValueError(f"Manifest row missing sub_tag: {manifest_path}")
            ses_text = str(row.get("ses", "") or "").strip()
            if not ses_text:
                raise ValueError(f"Manifest row missing ses for {sub_tag}: {manifest_path}")
            ses = int(ses_text)
            cleaned_beta = Path(str(row["cleaned_beta"]).strip())
            if not cleaned_beta.exists():
                raise FileNotFoundError(f"Missing cleaned beta volume: {cleaned_beta}")
            trial_keep_text = str(row.get("trial_keep_path", "") or "").strip()
            trial_keep_path = Path(trial_keep_text) if trial_keep_text else None
            n_trials_source = int(row.get("n_trials_source", 0) or 0)
            rows.append(
                ManifestRow(
                    sub_tag=sub_tag,
                    ses=ses,
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


def _group_manifest_rows_by_subject_session(manifest_rows: list[ManifestRow]) -> list[tuple[str, list[ManifestRow]]]:
    grouped: dict[tuple[str, int], list[ManifestRow]] = {}
    for row in manifest_rows:
        key = (row.sub_tag, int(row.ses))
        grouped.setdefault(key, []).append(row)
    ordered_keys = sorted(grouped, key=lambda item: (item[0], item[1]))
    return [(f"{sub_tag}-ses{ses}", grouped[(sub_tag, ses)]) for sub_tag, ses in ordered_keys]


def _accumulate_subject_nanmean(
    sum_buffer: np.ndarray,
    count_buffer: np.ndarray,
    values: np.ndarray,
) -> None:
    finite = np.isfinite(values)
    if np.any(finite):
        sum_buffer[finite] += values[finite]
        count_buffer[finite] += 1


def _finalize_subject_nanmean(
    sum_buffer: np.ndarray,
    count_buffer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(sum_buffer.shape, np.nan, dtype=np.float64)
    valid = count_buffer > 0
    if np.any(valid):
        out[valid] = sum_buffer[valid] / count_buffer[valid]
    return out, count_buffer


def _accumulate_consecutive_diff_and_variance_metrics(
    target_flat: np.ndarray,
    manifest_rows: list[ManifestRow],
    row_chunk_size: int,
    per_run_normalization: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate raw (or per-run-scaled) variability metrics across all runs.

    When per_run_normalization=True, each run's betas are divided by the run-level
    std (computed across all target voxels × kept trials in that run). This removes
    between-run amplitude differences while preserving relative voxel ordering within
    each run. Mean is NOT subtracted because Var(beta - c) = Var(beta) — subtracting
    a constant never changes variance or |delta|.
    """
    n_voxels = int(target_flat.size)
    pair_counts = np.zeros(n_voxels, dtype=np.int64)
    abs_diff_sums = np.zeros(n_voxels, dtype=np.float64)
    trial_sums = np.zeros(n_voxels, dtype=np.float64)
    trial_sq_sums = np.zeros(n_voxels, dtype=np.float64)
    trial_counts = np.zeros(n_voxels, dtype=np.int64)

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
        # Adjacency mask: True only for pairs that are consecutive in the original run.
        # Computed once per run, broadcast across voxel chunks.
        kept_idx = np.flatnonzero(keep_mask)
        is_adjacent = np.diff(kept_idx) == 1  # shape (n_kept - 1,)

        # Per-run normalization: compute a single scale factor from all (target_voxel × kept_trial)
        # entries in this run. The same divisor is applied to every voxel, so relative variability
        # differences between voxels are preserved. Mean is NOT subtracted (would not affect variance).
        run_scale = 1.0
        if per_run_normalization:
            run_sum = 0.0
            run_sq_sum = 0.0
            run_count = 0
            for start in range(0, n_voxels, row_chunk_size):
                stop = min(start + row_chunk_size, n_voxels)
                pre_chunk = flat_view[target_flat[start:stop]][:, keep_mask].astype(np.float64)
                fin = np.isfinite(pre_chunk)
                run_count += int(fin.sum())
                safe = np.where(fin, pre_chunk, 0.0)
                run_sum += float(safe.sum())
                run_sq_sum += float((safe * safe).sum())
            if run_count > 1:
                run_mean = run_sum / run_count
                run_pop_var = max(0.0, run_sq_sum / run_count - run_mean ** 2)
                run_std = float(np.sqrt(run_pop_var))
                if run_std > 0.0:
                    run_scale = run_std

        for start in range(0, n_voxels, row_chunk_size):
            stop = min(start + row_chunk_size, n_voxels)
            chunk = np.asarray(flat_view[target_flat[start:stop]][:, keep_mask], dtype=np.float32)
            if chunk.shape[1] == 0:
                continue

            if per_run_normalization and run_scale != 1.0:
                chunk = chunk / np.float32(run_scale)

            finite = np.isfinite(chunk)
            trial_counts[start:stop] += np.sum(finite, axis=1, dtype=np.int64)
            if np.any(finite):
                safe_chunk = np.where(finite, chunk, 0.0)
                trial_sums[start:stop] += np.sum(safe_chunk, axis=1, dtype=np.float64)
                trial_sq_sums[start:stop] += np.sum(safe_chunk * safe_chunk, axis=1, dtype=np.float64)

            if chunk.shape[1] < 2:
                continue
            prev_vals = chunk[:, :-1]
            next_vals = chunk[:, 1:]
            # valid_pairs combines: both values finite AND the pair is truly adjacent in the original run.
            valid_pairs = np.isfinite(prev_vals) & np.isfinite(next_vals) & is_adjacent[np.newaxis, :]
            pair_counts[start:stop] += np.sum(valid_pairs, axis=1, dtype=np.int64)
            abs_diff = np.abs(next_vals - prev_vals)
            if not np.all(valid_pairs):
                abs_diff = np.where(valid_pairs, abs_diff, 0.0)
            abs_diff_sums[start:stop] += np.sum(abs_diff, axis=1, dtype=np.float64)

    metric = np.full(n_voxels, np.nan, dtype=np.float64)
    valid = pair_counts > 0
    if np.any(valid):
        metric[valid] = abs_diff_sums[valid] / pair_counts[valid]

    variance = np.full(n_voxels, np.nan, dtype=np.float64)
    variance_valid = trial_counts > 1
    if np.any(variance_valid):
        numerator = trial_sq_sums[variance_valid] - (trial_sums[variance_valid] ** 2) / trial_counts[variance_valid]
        variance[variance_valid] = numerator / (trial_counts[variance_valid] - 1).astype(np.float64)
        variance[variance_valid] = np.where(variance[variance_valid] >= 0.0, variance[variance_valid], 0.0)

    return metric, pair_counts, variance, trial_counts


def _compute_per_voxel_mean_std(
    target_flat: np.ndarray,
    manifest_rows: list[ManifestRow],
    row_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1: compute per-voxel mean and std across all kept trials in all runs.

    Returns (mean, std), both shape (n_voxels,) float64.
    Voxels with zero std or fewer than 2 finite values get std=nan so that
    the normalization step naturally produces nan, which the isfinite guard handles.
    """
    n_voxels = int(target_flat.size)
    sums = np.zeros(n_voxels, dtype=np.float64)
    sq_sums = np.zeros(n_voxels, dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.int64)

    row_chunk_size = max(1, int(row_chunk_size))
    total_runs = len(manifest_rows)

    print("Pass 1: computing per-voxel mean and std for normalization...", flush=True)
    for run_idx, row in enumerate(manifest_rows, start=1):
        volume = np.load(row.cleaned_beta, mmap_mode="r")
        flat_view = volume.reshape(-1, volume.shape[-1])
        n_trials_source = int(volume.shape[-1])
        keep_mask = _load_trial_keep_mask(row, n_trials_source)
        print(
            f"  Pass 1 run {run_idx}/{total_runs}: {row.cleaned_beta.name} | kept = {int(np.count_nonzero(keep_mask))}",
            flush=True,
        )

        for start in range(0, n_voxels, row_chunk_size):
            stop = min(start + row_chunk_size, n_voxels)
            chunk = np.asarray(flat_view[target_flat[start:stop]][:, keep_mask], dtype=np.float32)
            if chunk.shape[1] == 0:
                continue
            finite = np.isfinite(chunk)
            counts[start:stop] += np.sum(finite, axis=1, dtype=np.int64)
            safe = np.where(finite, chunk.astype(np.float64), 0.0)
            sums[start:stop] += np.sum(safe, axis=1)
            sq_sums[start:stop] += np.sum(safe * safe, axis=1)

    per_voxel_mean = np.full(n_voxels, np.nan, dtype=np.float64)
    per_voxel_std = np.full(n_voxels, np.nan, dtype=np.float64)

    has_data = counts > 0
    per_voxel_mean[has_data] = sums[has_data] / counts[has_data]

    has_variance = counts > 1
    pop_var = np.zeros(n_voxels, dtype=np.float64)
    pop_var[has_variance] = np.maximum(
        0.0,
        sq_sums[has_variance] / counts[has_variance]
        - per_voxel_mean[has_variance] ** 2,
    )
    std_candidate = np.sqrt(pop_var)
    # Voxels with zero std cannot be normalized; set to nan so the isfinite
    # guard in the accumulation function excludes them automatically.
    nonzero_std = has_variance & (std_candidate > 0)
    per_voxel_std[nonzero_std] = std_candidate[nonzero_std]

    n_valid = int(np.count_nonzero(nonzero_std))
    print(f"Pass 1 complete: {n_valid}/{n_voxels} voxels have finite mean and non-zero std.", flush=True)
    return per_voxel_mean, per_voxel_std


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
    metric_name: str,
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
        xlabel=metric_name,
        panel_label="a",
    )
    _plot_density_panel(
        ax_b,
        np.log10(np.clip(selected_metric, log_eps, None)),
        np.log10(np.clip(nonselected_metric, log_eps, None)),
        rng,
        kde_max_points,
        xlabel=f"Log10({metric_name})",
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
    ax_d.set_xlabel(f"{metric_name} (non-selected voxels)")
    ax_d.set_ylabel("Density")
    ax_d.text(-0.02, 1.03, "d", transform=ax_d.transAxes, fontsize=18, fontweight="bold")
    ax_d.legend(frameon=True, fontsize=9, loc="upper right")
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)

    fig.suptitle(
        (
            f"Selected vs motor-area non-selected voxels | {metric_name} | "
            f"mean: {selected_mean:.3f} vs {nonselected_mean:.3f} | "
            f"p={p_lower:.4g} | resamples={num_resamples}"
        ),
        fontsize=14,
        y=1.02,
    )
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_cv_log_density_all_subjects(
    figure_path: Path,
    unit_cv_records: list[dict[str, object]],
    random_seed: int,
    kde_max_points: int,
) -> None:
    if not unit_cv_records:
        return

    n_units = len(unit_cv_records)
    ncols = 5 if n_units >= 5 else n_units
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.0 * nrows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(int(random_seed))

    for idx, record in enumerate(unit_cv_records):
        ax = axes_arr[idx]
        unit_label = str(record["unit_label"])
        n_runs = int(record["n_runs"])
        selected_cv = np.asarray(record["selected_cv"], dtype=np.float64)
        nonselected_cv = np.asarray(record["nonselected_cv"], dtype=np.float64)

        positive = np.concatenate([selected_cv, nonselected_cv])
        positive = positive[np.isfinite(positive) & (positive > 0)]
        log_eps = 1e-12
        if positive.size:
            log_eps = max(float(np.min(positive)) * 0.5, log_eps)

        selected_log = np.log10(np.clip(selected_cv, log_eps, None))
        nonselected_log = np.log10(np.clip(nonselected_cv, log_eps, None))

        xs_sel, ys_sel = _density_curve(selected_log, rng, kde_max_points)
        xs_non, ys_non = _density_curve(nonselected_log, rng, kde_max_points)
        ax.fill_between(xs_sel, ys_sel, color=SELECTED_COLOR, alpha=0.26)
        ax.plot(xs_sel, ys_sel, color=SELECTED_COLOR, linewidth=1.6)
        ax.fill_between(xs_non, ys_non, color=NONSELECTED_COLOR, alpha=0.22)
        ax.plot(xs_non, ys_non, color=NONSELECTED_COLOR, linewidth=1.6)
        ax.axvline(float(np.mean(selected_log)), linestyle="--", linewidth=1.2, color=SELECTED_COLOR)
        ax.axvline(float(np.mean(nonselected_log)), linestyle="--", linewidth=1.2, color=NONSELECTED_COLOR)
        ax.set_title(f"{unit_label} (runs={n_runs})", fontsize=9)
        ax.set_xlabel("Log10(CV)")
        ax.set_ylabel("Density")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(n_units, axes_arr.size):
        axes_arr[idx].set_axis_off()

    fig.suptitle(
        "Subject-session-wise Log10(CV) Density (Selected vs Motor Non-selected Voxels)",
        fontsize=13,
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

    manifest_rows = _load_manifest_rows(args.manifest_path)

    motor_flat, motor_region_names, motor_region_counts, motor_patterns = _load_motor_flat_indices(
        motor_mask_path=args.motor_mask_path,
        anat_path=args.anat_path,
        label_patterns=_split_csv_patterns(args.motor_label_patterns),
        atlas_cache_dir=args.motor_atlas_cache_dir,
    )
    motor_region_figure = args.output_dir / f"{args.output_stem}_motor_regions.png"
    _save_motor_region_figure(
        figure_path=motor_region_figure,
        region_names=motor_region_names,
        region_counts=motor_region_counts,
        selected_patterns=motor_patterns,
    )
    print(f"Saved motor-region report figure: {motor_region_figure}", flush=True)

    if np.setdiff1d(motor_flat, selected_flat, assume_unique=False).size == 0:
        raise ValueError("No motor voxels remain after removing selected set. Cannot form baseline non-selected pool.")
    nonselected_flat = np.setdiff1d(motor_flat, selected_flat, assume_unique=False)

    n_selected = int(selected_flat.size)
    target_flat = np.concatenate([selected_flat, nonselected_flat]).astype(np.int64, copy=False)
    print(
        f"Selected voxels: {selected_flat.size} | motor non-selected voxels: {nonselected_flat.size} | "
        f"(motor pool size: {motor_flat.size}) | "
        f"runs in manifest: {len(manifest_rows)}",
        flush=True,
    )
    session_groups = _group_manifest_rows_by_subject_session(manifest_rows)
    n_sessions = len(session_groups)
    n_voxels = int(target_flat.size)
    print(
        "Subject-session metric mode: compute per subject-session then average across subject-sessions "
        f"(units={n_sessions}).",
        flush=True,
    )

    diff_subject_sum = np.zeros(n_voxels, dtype=np.float64)
    diff_subject_n = np.zeros(n_voxels, dtype=np.int64)
    var_subject_sum = np.zeros(n_voxels, dtype=np.float64)
    var_subject_n = np.zeros(n_voxels, dtype=np.int64)
    cv_subject_sum = np.zeros(n_voxels, dtype=np.float64)
    cv_subject_n = np.zeros(n_voxels, dtype=np.int64)
    norm_diff_subject_sum = np.zeros(n_voxels, dtype=np.float64)
    norm_diff_subject_n = np.zeros(n_voxels, dtype=np.int64)
    rs_diff_subject_sum = np.zeros(n_voxels, dtype=np.float64)
    rs_diff_subject_n = np.zeros(n_voxels, dtype=np.int64)
    rs_var_subject_sum = np.zeros(n_voxels, dtype=np.float64)
    rs_var_subject_n = np.zeros(n_voxels, dtype=np.int64)

    pair_counts = np.zeros(n_voxels, dtype=np.int64)
    trial_counts = np.zeros(n_voxels, dtype=np.int64)
    session_cv_records: list[dict[str, object]] = []

    for session_idx, (session_label, session_rows) in enumerate(session_groups, start=1):
        print(
            f"Session-unit {session_idx}/{n_sessions}: {session_label} | runs = {len(session_rows)}",
            flush=True,
        )
        sub_norm_mean, sub_norm_std = _compute_per_voxel_mean_std(
            target_flat=target_flat,
            manifest_rows=session_rows,
            row_chunk_size=args.row_chunk_size,
        )
        print(f"Session-unit {session_idx}/{n_sessions}: accumulating raw metrics...", flush=True)
        sub_metric_values, sub_pair_counts, sub_variance_values, sub_trial_counts = _accumulate_consecutive_diff_and_variance_metrics(
            target_flat=target_flat,
            manifest_rows=session_rows,
            row_chunk_size=args.row_chunk_size,
            per_run_normalization=False,
        )
        print(f"Session-unit {session_idx}/{n_sessions}: accumulating per-run-scaled metrics...", flush=True)
        sub_rs_metric_values, _, sub_rs_variance_values, _ = _accumulate_consecutive_diff_and_variance_metrics(
            target_flat=target_flat,
            manifest_rows=session_rows,
            row_chunk_size=args.row_chunk_size,
            per_run_normalization=True,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            sub_cv_values = np.where(
                np.abs(sub_norm_mean) > 1e-8,
                sub_norm_std / np.abs(sub_norm_mean),
                np.nan,
            )
            sub_norm_diff_values = np.where(
                np.abs(sub_norm_mean) > 1e-8,
                sub_metric_values / np.abs(sub_norm_mean),
                np.nan,
            )
        sub_selected_cv = np.asarray(sub_cv_values[:n_selected], dtype=np.float64)
        sub_nonselected_cv = np.asarray(sub_cv_values[n_selected:], dtype=np.float64)
        sub_selected_cv = sub_selected_cv[np.isfinite(sub_selected_cv)]
        sub_nonselected_cv = sub_nonselected_cv[np.isfinite(sub_nonselected_cv)]
        if sub_selected_cv.size > 0 and sub_nonselected_cv.size > 0:
            session_cv_records.append(
                {
                    "unit_label": session_label,
                    "n_runs": len(session_rows),
                    "selected_cv": sub_selected_cv,
                    "nonselected_cv": sub_nonselected_cv,
                }
            )

        _accumulate_subject_nanmean(diff_subject_sum, diff_subject_n, sub_metric_values)
        _accumulate_subject_nanmean(var_subject_sum, var_subject_n, sub_variance_values)
        _accumulate_subject_nanmean(cv_subject_sum, cv_subject_n, sub_cv_values)
        _accumulate_subject_nanmean(norm_diff_subject_sum, norm_diff_subject_n, sub_norm_diff_values)
        _accumulate_subject_nanmean(rs_diff_subject_sum, rs_diff_subject_n, sub_rs_metric_values)
        _accumulate_subject_nanmean(rs_var_subject_sum, rs_var_subject_n, sub_rs_variance_values)

        pair_counts += sub_pair_counts
        trial_counts += sub_trial_counts

    metric_values, metric_subject_counts = _finalize_subject_nanmean(diff_subject_sum, diff_subject_n)
    variance_values, variance_subject_counts = _finalize_subject_nanmean(var_subject_sum, var_subject_n)
    cv_values, cv_subject_counts = _finalize_subject_nanmean(cv_subject_sum, cv_subject_n)
    norm_diff_values, norm_diff_subject_counts = _finalize_subject_nanmean(
        norm_diff_subject_sum, norm_diff_subject_n
    )
    rs_metric_values, rs_diff_subject_counts = _finalize_subject_nanmean(rs_diff_subject_sum, rs_diff_subject_n)
    rs_variance_values, rs_var_subject_counts = _finalize_subject_nanmean(rs_var_subject_sum, rs_var_subject_n)

    selected_diff_all = np.asarray(metric_values[:n_selected], dtype=np.float64)
    nonselected_diff_all = np.asarray(metric_values[n_selected:], dtype=np.float64)
    selected_variance_all = np.asarray(variance_values[:n_selected], dtype=np.float64)
    nonselected_variance_all = np.asarray(variance_values[n_selected:], dtype=np.float64)
    selected_counts_all = np.asarray(pair_counts[:n_selected], dtype=np.int64)
    nonselected_counts_all = np.asarray(pair_counts[n_selected:], dtype=np.int64)
    selected_trial_counts_all = np.asarray(trial_counts[:n_selected], dtype=np.int64)
    nonselected_trial_counts_all = np.asarray(trial_counts[n_selected:], dtype=np.int64)
    selected_diff_subject_counts_all = np.asarray(metric_subject_counts[:n_selected], dtype=np.int64)
    nonselected_diff_subject_counts_all = np.asarray(metric_subject_counts[n_selected:], dtype=np.int64)
    selected_variance_subject_counts_all = np.asarray(variance_subject_counts[:n_selected], dtype=np.int64)
    nonselected_variance_subject_counts_all = np.asarray(variance_subject_counts[n_selected:], dtype=np.int64)

    selected_diff_valid = np.isfinite(selected_diff_all)
    nonselected_diff_valid = np.isfinite(nonselected_diff_all)
    selected_variance_valid = np.isfinite(selected_variance_all)
    nonselected_variance_valid = np.isfinite(nonselected_variance_all)

    selected_metric = selected_diff_all[selected_diff_valid]
    nonselected_metric = nonselected_diff_all[nonselected_diff_valid]
    selected_counts = selected_counts_all[selected_diff_valid]
    nonselected_counts = nonselected_counts_all[nonselected_diff_valid]
    selected_diff_subject_counts = selected_diff_subject_counts_all[selected_diff_valid]
    nonselected_diff_subject_counts = nonselected_diff_subject_counts_all[nonselected_diff_valid]
    selected_variance = selected_variance_all[selected_variance_valid]
    nonselected_variance = nonselected_variance_all[nonselected_variance_valid]
    selected_trial_counts = selected_trial_counts_all[selected_variance_valid]
    nonselected_trial_counts = nonselected_trial_counts_all[nonselected_variance_valid]
    selected_variance_subject_counts = selected_variance_subject_counts_all[selected_variance_valid]
    nonselected_variance_subject_counts = nonselected_variance_subject_counts_all[nonselected_variance_valid]
    selected_flat_variance_valid = selected_flat[selected_variance_valid]
    nonselected_flat_variance_valid = nonselected_flat[nonselected_variance_valid]

    selected_flat_valid = selected_flat[selected_diff_valid]
    nonselected_flat_valid = nonselected_flat[nonselected_diff_valid]

    if selected_metric.size == 0:
        raise ValueError("Selected voxels have no finite consecutive-trial difference estimates.")
    if nonselected_metric.size == 0:
        raise ValueError("Motor non-selected voxels have no finite consecutive-trial difference estimates.")
    if selected_variance.size == 0:
        raise ValueError("Selected voxels have no finite trial-variance estimates.")
    if nonselected_variance.size == 0:
        raise ValueError("Motor non-selected voxels have no finite trial-variance estimates.")

    selected_cv_all = np.asarray(cv_values[:n_selected], dtype=np.float64)
    nonselected_cv_all = np.asarray(cv_values[n_selected:], dtype=np.float64)
    selected_cv_subject_counts_all = np.asarray(cv_subject_counts[:n_selected], dtype=np.int64)
    nonselected_cv_subject_counts_all = np.asarray(cv_subject_counts[n_selected:], dtype=np.int64)
    selected_cv_valid = np.isfinite(selected_cv_all)
    nonselected_cv_valid = np.isfinite(nonselected_cv_all)
    selected_cv = selected_cv_all[selected_cv_valid]
    nonselected_cv = nonselected_cv_all[nonselected_cv_valid]
    selected_cv_subject_counts = selected_cv_subject_counts_all[selected_cv_valid]
    nonselected_cv_subject_counts = nonselected_cv_subject_counts_all[nonselected_cv_valid]
    selected_flat_cv_valid = selected_flat[selected_cv_valid]
    nonselected_flat_cv_valid = nonselected_flat[nonselected_cv_valid]
    if selected_cv.size == 0:
        raise ValueError("Selected voxels have no finite coefficient-of-variation estimates.")
    if nonselected_cv.size == 0:
        raise ValueError("Motor non-selected voxels have no finite coefficient-of-variation estimates.")

    selected_norm_diff_all = np.asarray(norm_diff_values[:n_selected], dtype=np.float64)
    nonselected_norm_diff_all = np.asarray(norm_diff_values[n_selected:], dtype=np.float64)
    selected_norm_diff_subject_counts_all = np.asarray(norm_diff_subject_counts[:n_selected], dtype=np.int64)
    nonselected_norm_diff_subject_counts_all = np.asarray(norm_diff_subject_counts[n_selected:], dtype=np.int64)
    selected_norm_diff_valid = np.isfinite(selected_norm_diff_all)
    nonselected_norm_diff_valid = np.isfinite(nonselected_norm_diff_all)
    selected_norm_diff = selected_norm_diff_all[selected_norm_diff_valid]
    nonselected_norm_diff = nonselected_norm_diff_all[nonselected_norm_diff_valid]
    selected_norm_diff_subject_counts = selected_norm_diff_subject_counts_all[selected_norm_diff_valid]
    nonselected_norm_diff_subject_counts = nonselected_norm_diff_subject_counts_all[nonselected_norm_diff_valid]
    selected_flat_norm_diff_valid = selected_flat[selected_norm_diff_valid]
    nonselected_flat_norm_diff_valid = nonselected_flat[nonselected_norm_diff_valid]
    if selected_norm_diff.size == 0:
        raise ValueError("Selected voxels have no finite normalized-|delta| estimates.")
    if nonselected_norm_diff.size == 0:
        raise ValueError("Motor non-selected voxels have no finite normalized-|delta| estimates.")

    # Per-run-scaled metrics: slice from rs_metric_values and rs_variance_values.
    selected_rs_diff_all = np.asarray(rs_metric_values[:n_selected], dtype=np.float64)
    nonselected_rs_diff_all = np.asarray(rs_metric_values[n_selected:], dtype=np.float64)
    selected_rs_var_all = np.asarray(rs_variance_values[:n_selected], dtype=np.float64)
    nonselected_rs_var_all = np.asarray(rs_variance_values[n_selected:], dtype=np.float64)
    selected_rs_diff_subject_counts_all = np.asarray(rs_diff_subject_counts[:n_selected], dtype=np.int64)
    nonselected_rs_diff_subject_counts_all = np.asarray(rs_diff_subject_counts[n_selected:], dtype=np.int64)
    selected_rs_var_subject_counts_all = np.asarray(rs_var_subject_counts[:n_selected], dtype=np.int64)
    nonselected_rs_var_subject_counts_all = np.asarray(rs_var_subject_counts[n_selected:], dtype=np.int64)
    selected_rs_diff_valid = np.isfinite(selected_rs_diff_all)
    nonselected_rs_diff_valid = np.isfinite(nonselected_rs_diff_all)
    selected_rs_var_valid = np.isfinite(selected_rs_var_all)
    nonselected_rs_var_valid = np.isfinite(nonselected_rs_var_all)
    selected_rs_diff = selected_rs_diff_all[selected_rs_diff_valid]
    nonselected_rs_diff = nonselected_rs_diff_all[nonselected_rs_diff_valid]
    selected_rs_var = selected_rs_var_all[selected_rs_var_valid]
    nonselected_rs_var = nonselected_rs_var_all[nonselected_rs_var_valid]
    selected_rs_diff_subject_counts = selected_rs_diff_subject_counts_all[selected_rs_diff_valid]
    nonselected_rs_diff_subject_counts = nonselected_rs_diff_subject_counts_all[nonselected_rs_diff_valid]
    selected_rs_var_subject_counts = selected_rs_var_subject_counts_all[selected_rs_var_valid]
    nonselected_rs_var_subject_counts = nonselected_rs_var_subject_counts_all[nonselected_rs_var_valid]
    if selected_rs_diff.size == 0 or nonselected_rs_diff.size == 0:
        raise ValueError("Per-run-scaled |delta| has empty valid set for selected or non-selected voxels.")
    if selected_rs_var.size == 0 or nonselected_rs_var.size == 0:
        raise ValueError("Per-run-scaled variance has empty valid set for selected or non-selected voxels.")

    diff_thresholds, diff_prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_metric,
        nonselected_values=nonselected_metric,
        percentiles=list(args.percentile_thresholds),
    )
    diff_resampled_means, diff_resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_metric,
        sample_size=selected_metric.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    diff_selected_mean = float(np.mean(selected_metric))
    diff_p_lower = (1.0 + float(np.count_nonzero(diff_resampled_means <= diff_selected_mean))) / (1.0 + float(args.num_resamples))

    var_thresholds, var_prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_variance,
        nonselected_values=nonselected_variance,
        percentiles=list(args.percentile_thresholds),
    )
    var_resampled_means, var_resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_variance,
        sample_size=selected_variance.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    var_selected_mean = float(np.mean(selected_variance))
    var_p_lower = (1.0 + float(np.count_nonzero(var_resampled_means <= var_selected_mean))) / (1.0 + float(args.num_resamples))

    cv_thresholds, cv_prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_cv,
        nonselected_values=nonselected_cv,
        percentiles=list(args.percentile_thresholds),
    )
    cv_resampled_means, cv_resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_cv,
        sample_size=selected_cv.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    cv_selected_mean = float(np.mean(selected_cv))
    cv_p_lower = (1.0 + float(np.count_nonzero(cv_resampled_means <= cv_selected_mean))) / (1.0 + float(args.num_resamples))

    norm_diff_thresholds, norm_diff_prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_norm_diff,
        nonselected_values=nonselected_norm_diff,
        percentiles=list(args.percentile_thresholds),
    )
    norm_diff_resampled_means, norm_diff_resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_norm_diff,
        sample_size=selected_norm_diff.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    norm_diff_selected_mean = float(np.mean(selected_norm_diff))
    norm_diff_p_lower = (1.0 + float(np.count_nonzero(norm_diff_resampled_means <= norm_diff_selected_mean))) / (1.0 + float(args.num_resamples))

    rs_diff_thresholds, rs_diff_prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_rs_diff,
        nonselected_values=nonselected_rs_diff,
        percentiles=list(args.percentile_thresholds),
    )
    rs_diff_resampled_means, rs_diff_resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_rs_diff,
        sample_size=selected_rs_diff.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    rs_diff_selected_mean = float(np.mean(selected_rs_diff))
    rs_diff_p_lower = (1.0 + float(np.count_nonzero(rs_diff_resampled_means <= rs_diff_selected_mean))) / (1.0 + float(args.num_resamples))

    rs_var_thresholds, rs_var_prevalence_ratios = _compute_prevalence_ratios(
        selected_values=selected_rs_var,
        nonselected_values=nonselected_rs_var,
        percentiles=list(args.percentile_thresholds),
    )
    rs_var_resampled_means, rs_var_resample_replace = _resample_nonselected_means(
        nonselected_values=nonselected_rs_var,
        sample_size=selected_rs_var.size,
        num_resamples=args.num_resamples,
        rng=rng,
    )
    rs_var_selected_mean = float(np.mean(selected_rs_var))
    rs_var_p_lower = (1.0 + float(np.count_nonzero(rs_var_resampled_means <= rs_var_selected_mean))) / (1.0 + float(args.num_resamples))

    diff_png_path = args.output_dir / f"{args.output_stem}.png"
    var_png_path = args.output_dir / f"{args.output_stem}_variance.png"
    cv_png_path = args.output_dir / f"{args.output_stem}_cv.png"
    norm_diff_png_path = args.output_dir / f"{args.output_stem}_norm_diff.png"
    rs_diff_png_path = args.output_dir / f"{args.output_stem}_runscaled_diff.png"
    rs_var_png_path = args.output_dir / f"{args.output_stem}_runscaled_variance.png"
    cv_subjects_log_png_path = args.output_dir / f"{args.output_stem}_cv_subjects_log_panel.png"
    summary_json_path = args.output_dir / f"{args.output_stem}_summary.json"
    analysis_npz_path = args.output_dir / f"{args.output_stem}_analysis_data.npz"

    print(
        f"Saving figures to {diff_png_path}, {var_png_path}, {cv_png_path}, "
        f"{norm_diff_png_path}, {rs_diff_png_path}, {rs_var_png_path}, {cv_subjects_log_png_path}",
        flush=True,
    )
    _plot_summary_figure(
        figure_path=diff_png_path,
        selected_metric=selected_metric,
        nonselected_metric=nonselected_metric,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=diff_prevalence_ratios,
        resampled_means=diff_resampled_means,
        p_lower=diff_p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
        metric_name="Mean |Delta| Consecutive Trials",
    )
    _plot_summary_figure(
        figure_path=var_png_path,
        selected_metric=selected_variance,
        nonselected_metric=nonselected_variance,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=var_prevalence_ratios,
        resampled_means=var_resampled_means,
        p_lower=var_p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
        metric_name="Variance Across Kept Trials",
    )
    _plot_summary_figure(
        figure_path=cv_png_path,
        selected_metric=selected_cv,
        nonselected_metric=nonselected_cv,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=cv_prevalence_ratios,
        resampled_means=cv_resampled_means,
        p_lower=cv_p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
        metric_name="Coefficient of Variation (std/|mean|)",
    )
    _plot_summary_figure(
        figure_path=norm_diff_png_path,
        selected_metric=selected_norm_diff,
        nonselected_metric=nonselected_norm_diff,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=norm_diff_prevalence_ratios,
        resampled_means=norm_diff_resampled_means,
        p_lower=norm_diff_p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
        metric_name="Normalized |Delta| (raw_diff / |voxel_mean|)",
    )
    _plot_summary_figure(
        figure_path=rs_diff_png_path,
        selected_metric=selected_rs_diff,
        nonselected_metric=nonselected_rs_diff,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=rs_diff_prevalence_ratios,
        resampled_means=rs_diff_resampled_means,
        p_lower=rs_diff_p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
        metric_name="Per-Run-Scaled Mean |Delta| (beta / std_run)",
    )
    _plot_summary_figure(
        figure_path=rs_var_png_path,
        selected_metric=selected_rs_var,
        nonselected_metric=nonselected_rs_var,
        percentile_labels=list(args.percentile_thresholds),
        prevalence_ratios=rs_var_prevalence_ratios,
        resampled_means=rs_var_resampled_means,
        p_lower=rs_var_p_lower,
        num_resamples=int(args.num_resamples),
        rng=rng,
        kde_max_points=int(args.kde_max_points),
        metric_name="Per-Run-Scaled Variance (beta / std_run)",
    )
    _plot_cv_log_density_all_subjects(
        figure_path=cv_subjects_log_png_path,
        unit_cv_records=session_cv_records,
        random_seed=int(args.random_seed),
        kde_max_points=int(args.kde_max_points),
    )

    summary = {
        "selected_source": selected_source,
        "selected_source_type": selected_source_type,
        "selected_indices_path": str(args.selected_indices_path),
        "selected_csv_path": str(args.selected_csv_path) if args.selected_csv_path is not None else None,
        "anat_path": str(args.anat_path),
        "manifest_path": str(args.manifest_path),
        "aggregation_mode": "subject_sessionwise_voxelwise_nanmean",
        "subject_session_count_total": int(n_sessions),
        "motor_mask_source": str(args.motor_mask_path) if args.motor_mask_path is not None else "harvard_oxford_auto",
        "motor_label_patterns": list(_split_csv_patterns(args.motor_label_patterns)),
        "motor_region_names": motor_region_names,
        "motor_region_counts": [int(v) for v in motor_region_counts],
        "motor_pool_size": int(motor_flat.size),
        "selected_in_motor_count": int(np.count_nonzero(np.isin(selected_flat, motor_flat))),
        "selected_count_total": int(selected_flat.size),
        "nonselected_count_total": int(nonselected_flat.size),
        "selected_count_valid": int(selected_metric.size),
        "nonselected_count_valid": int(nonselected_metric.size),
        "selected_mean_abs_consecutive_trial_diff": float(np.mean(selected_metric)),
        "nonselected_mean_abs_consecutive_trial_diff": float(np.mean(nonselected_metric)),
        "selected_median_abs_consecutive_trial_diff": float(np.median(selected_metric)),
        "nonselected_median_abs_consecutive_trial_diff": float(np.median(nonselected_metric)),
        "selected_min_units_per_voxel_diff": int(np.min(selected_diff_subject_counts)),
        "nonselected_min_units_per_voxel_diff": int(np.min(nonselected_diff_subject_counts)),
        "selected_min_finite_consecutive_pairs": int(np.min(selected_counts)),
        "nonselected_min_finite_consecutive_pairs": int(np.min(nonselected_counts)),
        "selected_count_variance_valid": int(selected_variance.size),
        "nonselected_count_variance_valid": int(nonselected_variance.size),
        "selected_mean_trial_variance": float(np.mean(selected_variance)),
        "nonselected_mean_trial_variance": float(np.mean(nonselected_variance)),
        "selected_median_trial_variance": float(np.median(selected_variance)),
        "nonselected_median_trial_variance": float(np.median(nonselected_variance)),
        "selected_min_units_per_voxel_variance": int(np.min(selected_variance_subject_counts)),
        "nonselected_min_units_per_voxel_variance": int(np.min(nonselected_variance_subject_counts)),
        "selected_min_finite_trials": int(np.min(selected_trial_counts)),
        "nonselected_min_finite_trials": int(np.min(nonselected_trial_counts)),
        "diff_resample_mean": float(np.mean(diff_resampled_means)),
        "diff_resample_std": float(np.std(diff_resampled_means, ddof=1)),
        "diff_resample_ci_2p5": float(np.percentile(diff_resampled_means, 2.5)),
        "diff_resample_ci_97p5": float(np.percentile(diff_resampled_means, 97.5)),
        "diff_resample_p_lower_or_equal_selected": float(diff_p_lower),
        "diff_resample_with_replacement": bool(diff_resample_replace),
        "var_resample_mean": float(np.mean(var_resampled_means)),
        "var_resample_std": float(np.std(var_resampled_means, ddof=1)),
        "var_resample_ci_2p5": float(np.percentile(var_resampled_means, 2.5)),
        "var_resample_ci_97p5": float(np.percentile(var_resampled_means, 97.5)),
        "var_resample_p_lower_or_equal_selected": float(var_p_lower),
        "var_resample_with_replacement": bool(var_resample_replace),
        "normalization_applied": "per_run_std",
        "selected_count_cv_valid": int(selected_cv.size),
        "nonselected_count_cv_valid": int(nonselected_cv.size),
        "selected_mean_cv": float(np.mean(selected_cv)),
        "nonselected_mean_cv": float(np.mean(nonselected_cv)),
        "selected_median_cv": float(np.median(selected_cv)),
        "nonselected_median_cv": float(np.median(nonselected_cv)),
        "selected_min_units_per_voxel_cv": int(np.min(selected_cv_subject_counts)),
        "nonselected_min_units_per_voxel_cv": int(np.min(nonselected_cv_subject_counts)),
        "cv_resample_mean": float(np.mean(cv_resampled_means)),
        "cv_resample_std": float(np.std(cv_resampled_means, ddof=1)),
        "cv_resample_ci_2p5": float(np.percentile(cv_resampled_means, 2.5)),
        "cv_resample_ci_97p5": float(np.percentile(cv_resampled_means, 97.5)),
        "cv_resample_p_lower_or_equal_selected": float(cv_p_lower),
        "cv_resample_with_replacement": bool(cv_resample_replace),
        "selected_count_norm_diff_valid": int(selected_norm_diff.size),
        "nonselected_count_norm_diff_valid": int(nonselected_norm_diff.size),
        "selected_mean_norm_diff": float(np.mean(selected_norm_diff)),
        "nonselected_mean_norm_diff": float(np.mean(nonselected_norm_diff)),
        "selected_min_units_per_voxel_norm_diff": int(np.min(selected_norm_diff_subject_counts)),
        "nonselected_min_units_per_voxel_norm_diff": int(np.min(nonselected_norm_diff_subject_counts)),
        "norm_diff_resample_p_lower_or_equal_selected": float(norm_diff_p_lower),
        "norm_diff_resample_with_replacement": bool(norm_diff_resample_replace),
        "selected_count_rs_diff_valid": int(selected_rs_diff.size),
        "nonselected_count_rs_diff_valid": int(nonselected_rs_diff.size),
        "selected_mean_rs_diff": float(np.mean(selected_rs_diff)),
        "nonselected_mean_rs_diff": float(np.mean(nonselected_rs_diff)),
        "selected_min_units_per_voxel_rs_diff": int(np.min(selected_rs_diff_subject_counts)),
        "nonselected_min_units_per_voxel_rs_diff": int(np.min(nonselected_rs_diff_subject_counts)),
        "rs_diff_resample_p_lower_or_equal_selected": float(rs_diff_p_lower),
        "rs_diff_resample_with_replacement": bool(rs_diff_resample_replace),
        "selected_count_rs_var_valid": int(selected_rs_var.size),
        "nonselected_count_rs_var_valid": int(nonselected_rs_var.size),
        "selected_mean_rs_var": float(np.mean(selected_rs_var)),
        "nonselected_mean_rs_var": float(np.mean(nonselected_rs_var)),
        "selected_min_units_per_voxel_rs_var": int(np.min(selected_rs_var_subject_counts)),
        "nonselected_min_units_per_voxel_rs_var": int(np.min(nonselected_rs_var_subject_counts)),
        "rs_var_resample_p_lower_or_equal_selected": float(rs_var_p_lower),
        "rs_var_resample_with_replacement": bool(rs_var_resample_replace),
        "num_resamples": int(args.num_resamples),
        "random_seed": int(args.random_seed),
        "diff_png_path": str(diff_png_path),
        "variance_png_path": str(var_png_path),
        "cv_png_path": str(cv_png_path),
        "cv_subjects_log_panel_png_path": str(cv_subjects_log_png_path),
        "norm_diff_png_path": str(norm_diff_png_path),
        "rs_diff_png_path": str(rs_diff_png_path),
        "rs_var_png_path": str(rs_var_png_path),
        "motor_region_figure_path": str(motor_region_figure),
    }
    with summary_json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    np.savez_compressed(
        analysis_npz_path,
        selected_flat_indices=selected_flat_valid.astype(np.int64, copy=False),
        nonselected_flat_indices=nonselected_flat_valid.astype(np.int64, copy=False),
        selected_mean_abs_consecutive_trial_diff=selected_metric.astype(np.float32, copy=False),
        nonselected_mean_abs_consecutive_trial_diff=nonselected_metric.astype(np.float32, copy=False),
        selected_subject_count_diff=selected_diff_subject_counts.astype(np.int16, copy=False),
        nonselected_subject_count_diff=nonselected_diff_subject_counts.astype(np.int16, copy=False),
        selected_finite_consecutive_pair_counts=selected_counts.astype(np.int32, copy=False),
        nonselected_finite_consecutive_pair_counts=nonselected_counts.astype(np.int32, copy=False),
        prevalence_thresholds=diff_thresholds.astype(np.float32, copy=False),
        prevalence_ratios=diff_prevalence_ratios.astype(np.float32, copy=False),
        resampled_nonselected_means=diff_resampled_means.astype(np.float32, copy=False),
        selected_trial_variance=selected_variance.astype(np.float32, copy=False),
        nonselected_trial_variance=nonselected_variance.astype(np.float32, copy=False),
        selected_subject_count_variance=selected_variance_subject_counts.astype(np.int16, copy=False),
        nonselected_subject_count_variance=nonselected_variance_subject_counts.astype(np.int16, copy=False),
        selected_finite_trial_counts=selected_trial_counts.astype(np.int32, copy=False),
        nonselected_finite_trial_counts=nonselected_trial_counts.astype(np.int32, copy=False),
        selected_flat_indices_variance=selected_flat_variance_valid.astype(np.int64, copy=False),
        nonselected_flat_indices_variance=nonselected_flat_variance_valid.astype(np.int64, copy=False),
        variance_prevalence_thresholds=var_thresholds.astype(np.float32, copy=False),
        variance_prevalence_ratios=var_prevalence_ratios.astype(np.float32, copy=False),
        resampled_nonselected_variance_means=var_resampled_means.astype(np.float32, copy=False),
        motor_flat_indices=motor_flat.astype(np.int64, copy=False),
        motor_region_names=np.array(motor_region_names, dtype=object),
        motor_region_counts=np.asarray(motor_region_counts, dtype=np.int32),
        motor_region_patterns=np.array(motor_patterns, dtype=object),
        selected_cv=selected_cv.astype(np.float32, copy=False),
        nonselected_cv=nonselected_cv.astype(np.float32, copy=False),
        selected_subject_count_cv=selected_cv_subject_counts.astype(np.int16, copy=False),
        nonselected_subject_count_cv=nonselected_cv_subject_counts.astype(np.int16, copy=False),
        selected_flat_indices_cv=selected_flat_cv_valid.astype(np.int64, copy=False),
        nonselected_flat_indices_cv=nonselected_flat_cv_valid.astype(np.int64, copy=False),
        cv_prevalence_thresholds=cv_thresholds.astype(np.float32, copy=False),
        cv_prevalence_ratios=cv_prevalence_ratios.astype(np.float32, copy=False),
        resampled_nonselected_cv_means=cv_resampled_means.astype(np.float32, copy=False),
        selected_norm_diff=selected_norm_diff.astype(np.float32, copy=False),
        nonselected_norm_diff=nonselected_norm_diff.astype(np.float32, copy=False),
        selected_subject_count_norm_diff=selected_norm_diff_subject_counts.astype(np.int16, copy=False),
        nonselected_subject_count_norm_diff=nonselected_norm_diff_subject_counts.astype(np.int16, copy=False),
        selected_flat_indices_norm_diff=selected_flat_norm_diff_valid.astype(np.int64, copy=False),
        nonselected_flat_indices_norm_diff=nonselected_flat_norm_diff_valid.astype(np.int64, copy=False),
        norm_diff_prevalence_thresholds=norm_diff_thresholds.astype(np.float32, copy=False),
        norm_diff_prevalence_ratios=norm_diff_prevalence_ratios.astype(np.float32, copy=False),
        resampled_nonselected_norm_diff_means=norm_diff_resampled_means.astype(np.float32, copy=False),
        selected_runscaled_diff=selected_rs_diff.astype(np.float32, copy=False),
        nonselected_runscaled_diff=nonselected_rs_diff.astype(np.float32, copy=False),
        selected_subject_count_rs_diff=selected_rs_diff_subject_counts.astype(np.int16, copy=False),
        nonselected_subject_count_rs_diff=nonselected_rs_diff_subject_counts.astype(np.int16, copy=False),
        selected_runscaled_variance=selected_rs_var.astype(np.float32, copy=False),
        nonselected_runscaled_variance=nonselected_rs_var.astype(np.float32, copy=False),
        selected_subject_count_rs_var=selected_rs_var_subject_counts.astype(np.int16, copy=False),
        nonselected_subject_count_rs_var=nonselected_rs_var_subject_counts.astype(np.int16, copy=False),
    )

    print(f"Saved summary JSON: {summary_json_path}", flush=True)
    print(f"Saved analysis NPZ: {analysis_npz_path}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
