#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt


PARAM_KEYS = ("task", "bold", "beta", "smooth", "gamma")
_SUBJECT_RE = re.compile(r"^sub[-_]?(\d+)$", re.IGNORECASE)
_SESSION_RE = re.compile(r"^ses[-_]?(\d+)$", re.IGNORECASE)


def _parse_params(name: str) -> Tuple[dict, Optional[float]]:
    tokens = name.split("_")
    params = {}
    for key in PARAM_KEYS:
        for token in tokens:
            if token.startswith(key):
                raw = token[len(key):]
                try:
                    params[key] = float(raw)
                except ValueError:
                    pass
                break
    thr = None
    for token in tokens:
        if token.startswith("bold_thr"):
            raw = token.replace("bold_thr", "")
            try:
                thr = float(raw)
            except ValueError:
                thr = raw or None
            break
    return params, thr


def _label_from_name(name: str) -> str:
    params, thr = _parse_params(name)
    parts = []
    for key in PARAM_KEYS:
        if key in params:
            val = params[key]
            if float(val).is_integer():
                val = int(val)
            parts.append(f"{key}{val}")
    if thr is not None:
        parts.append(f"thr{thr}")
    return " ".join(parts) if parts else name


def _sort_key(path: Path) -> tuple:
    params, thr = _parse_params(path.stem.replace(".nii", ""))
    key = []
    for key_name in PARAM_KEYS:
        val = params.get(key_name)
        if val is None:
            key.append(float("inf"))
        else:
            key.append(val)
    if thr is None:
        key.append(float("inf"))
    else:
        try:
            key.append(float(thr))
        except ValueError:
            key.append(float("inf"))
    return tuple(key)


def _normalize_subject_folder(subject: str, session: Optional[str]) -> str:
    subject = subject.strip()
    if not subject:
        return subject
    lowered = subject.lower()
    if "ses" in lowered:
        return subject
    match = _SUBJECT_RE.match(subject)
    subj_id = match.group(1) if match else subject
    if subj_id.isdigit():
        subj_id = subj_id.zfill(2)
        subject_label = f"sub{subj_id}"
    else:
        subject_label = subject if subject.startswith("sub") else f"sub{subj_id}"
    if session is None:
        session = "1"
    if session == "":
        return subject_label
    ses_match = _SESSION_RE.match(session)
    ses_id = ses_match.group(1) if ses_match else session
    return f"{subject_label}-ses{ses_id}"


def _resolve_input_dir(input_dir: Path, subject: Optional[str], session: Optional[str]) -> Path:
    if not subject:
        return input_dir
    subject = subject.strip()
    if not subject:
        return input_dir
    subject_path = Path(subject)
    if subject_path.is_absolute() or len(subject_path.parts) > 1:
        return subject_path
    subject_folder = _normalize_subject_folder(subject, session)
    return input_dir / subject_folder


def _load_mask(
    path: Path, threshold: Optional[float], percentile: Optional[float], use_abs: bool
) -> Tuple[np.ndarray, tuple, tuple]:
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if use_abs:
        data = np.abs(data)
    finite = np.isfinite(data)
    data = np.where(finite, data, 0.0)

    if percentile is not None:
        if np.any(finite):
            thr = float(np.percentile(data[finite], percentile))
        else:
            thr = 0.0
        mask = data >= thr
    elif threshold is not None:
        mask = data >= threshold
    else:
        mask = data > 0
    zooms = img.header.get_zooms()[:3]
    return mask, img.shape, zooms


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    overlap = np.count_nonzero(mask_a & mask_b)
    denom = int(mask_a.sum()) + int(mask_b.sum())
    return (2.0 * overlap / denom) if denom else np.nan


def _cohens_kappa(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    total = mask_a.size
    if total == 0:
        return np.nan
    tp = np.count_nonzero(mask_a & mask_b)
    tn = np.count_nonzero(~mask_a & ~mask_b)
    fp = np.count_nonzero(~mask_a & mask_b)
    fn = np.count_nonzero(mask_a & ~mask_b)
    po = (tp + tn) / total
    p_yes_a = (tp + fn) / total
    p_yes_b = (tp + fp) / total
    p_no_a = 1.0 - p_yes_a
    p_no_b = 1.0 - p_yes_b
    pe = p_yes_a * p_yes_b + p_no_a * p_no_b
    denom = 1.0 - pe
    return (po - pe) / denom if denom else np.nan


def _distance_to_mask(mask: np.ndarray, sampling: tuple) -> Optional[np.ndarray]:
    if not np.any(mask):
        return None
    distances = distance_transform_edt(~mask, sampling=sampling)
    return distances.astype(np.float32)


def _distance_metrics(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    dist_to_a: Optional[np.ndarray],
    dist_to_b: Optional[np.ndarray],
) -> Tuple[float, float]:
    if dist_to_a is None or dist_to_b is None:
        return np.nan, np.nan
    if not (np.any(mask_a) and np.any(mask_b)):
        return np.nan, np.nan
    dists_a = dist_to_b[mask_a]
    dists_b = dist_to_a[mask_b]
    hausdorff = max(float(dists_a.max()), float(dists_b.max()))
    mmd = float(np.median(np.concatenate([dists_a, dists_b])))
    return hausdorff, mmd


def _select_paths(
    root: Path, pattern: str, include: Optional[str], exclude: Optional[str], recursive: bool
) -> list:
    if recursive:
        paths = sorted(root.rglob(pattern))
    else:
        paths = sorted(root.glob(pattern))
    if include:
        inc_re = re.compile(include)
        paths = [p for p in paths if inc_re.search(p.name)]
    if exclude:
        exc_re = re.compile(exclude)
        paths = [p for p in paths if not exc_re.search(p.name)]
    return [p for p in paths if p.is_file()]


def _write_csv(matrix: np.ndarray, labels: list, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{val:.6f}" if np.isfinite(val) else "" for val in row])


def _plot_heatmap(matrix: np.ndarray, labels: list, out_path: Path, dpi: int) -> None:
    n = len(labels)
    fig_size = max(6, min(18, 0.5 * n))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="magma")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Network (parameter set)")
    ax.set_ylabel("Network (parameter set)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Dice overlap")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def _format_heatmap_axis(ax, labels: list) -> None:
    n = len(labels)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Network (parameter set)")
    ax.set_ylabel("Network (parameter set)")
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3)
    ax.tick_params(which="minor", bottom=False, left=False)


def _safe_vmax(matrix: np.ndarray, default: float = 1.0) -> float:
    finite = np.isfinite(matrix)
    if not np.any(finite):
        return default
    vmax = float(matrix[finite].max())
    return vmax if vmax > 0 else default


def _plot_metric_heatmaps(
    dice_matrix: np.ndarray,
    kappa_matrix: np.ndarray,
    hausdorff_matrix: np.ndarray,
    mmd_matrix: np.ndarray,
    labels: list,
    out_path: Path,
    dpi: int,
) -> None:
    n = len(labels)
    base = max(6, min(16, 0.4 * n))
    fig, axes = plt.subplots(2, 2, figsize=(2 * base, 2 * base))
    axes = axes.ravel()

    cmap = "jet"

    im_dice = axes[0].imshow(dice_matrix, vmin=0, vmax=1, cmap=cmap)
    axes[0].set_title("Dice overlap", fontsize=10)
    _format_heatmap_axis(axes[0], labels)
    cbar_dice = fig.colorbar(im_dice, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_dice.set_label("Dice overlap")

    im_kappa = axes[1].imshow(kappa_matrix, vmin=-1, vmax=1, cmap=cmap)
    axes[1].set_title("Cohen's kappa", fontsize=10)
    _format_heatmap_axis(axes[1], labels)
    cbar_kappa = fig.colorbar(im_kappa, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_kappa.set_label("Cohen's kappa")

    hausdorff_max = _safe_vmax(hausdorff_matrix, default=1.0)
    im_haus = axes[2].imshow(hausdorff_matrix, vmin=0, vmax=hausdorff_max, cmap=cmap)
    axes[2].set_title("Hausdorff distance", fontsize=10)
    _format_heatmap_axis(axes[2], labels)
    cbar_haus = fig.colorbar(im_haus, ax=axes[2], fraction=0.046, pad=0.04)
    cbar_haus.set_label("Distance (spatial units)")

    mmd_max = _safe_vmax(mmd_matrix, default=1.0)
    im_mmd = axes[3].imshow(mmd_matrix, vmin=0, vmax=mmd_max, cmap=cmap)
    axes[3].set_title("Median minimal distance (MMD)", fontsize=10)
    _format_heatmap_axis(axes[3], labels)
    cbar_mmd = fig.colorbar(im_mmd, ax=axes[3], fraction=0.046, pad=0.04)
    cbar_mmd.set_label("Distance (spatial units)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise Dice overlap, Cohen's kappa, Hausdorff distance, and median minimal "
            "distance (MMD) for NIfTI networks and plot heatmaps."
        )
    )
    parser.add_argument(
        "--input-dir",
        default="Results",
        help="Results root directory containing NIfTI maps (default: Results).",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject identifier or Results subfolder (e.g., 9, sub09, sub09-ses1).",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session identifier when building the subject folder (default: 1).",
    )
    parser.add_argument(
        "--pattern",
        default="*.nii.gz",
        help="Glob pattern to match files (default: *.nii.gz).",
    )
    parser.add_argument(
        "--include",
        default=None,
        help="Regex to include filenames (e.g., 'bold_thr95').",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help="Regex to exclude filenames.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search input directory recursively.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary threshold (values >= threshold are kept).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=None,
        help="Percentile-based threshold per image (0-100).",
    )
    parser.add_argument(
        "--abs",
        action="store_true",
        help="Use absolute values before thresholding.",
    )
    parser.add_argument(
        "--label-style",
        choices=("short", "filename"),
        default="short",
        help="Label style for heatmap axes.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path for combined heatmap image (Dice, Kappa, Hausdorff, MMD). "
            "Defaults to <input-dir>/dice_kappa_hausdorff_mmd_heatmap.png."
        ),
    )
    parser.add_argument(
        "--matrix-csv",
        default=None,
        help="Path for CSV output of the DSC matrix. Defaults to <input-dir>/dice_matrix.csv.",
    )
    parser.add_argument(
        "--kappa-matrix-csv",
        default=None,
        help="Path for CSV output of the Cohen's kappa matrix. Defaults to <input-dir>/kappa_matrix.csv.",
    )
    parser.add_argument(
        "--hausdorff-matrix-csv",
        default=None,
        help="Path for CSV output of the Hausdorff distance matrix. Defaults to <input-dir>/hausdorff_matrix.csv.",
    )
    parser.add_argument(
        "--mmd-matrix-csv",
        default=None,
        help="Path for CSV output of the MMD matrix. Defaults to <input-dir>/mmd_matrix.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved heatmap.",
    )
    args = parser.parse_args()

    input_dir = _resolve_input_dir(Path(args.input_dir), args.subject, args.session)
    output_path = (
        Path(args.output)
        if args.output
        else input_dir / "dice_kappa_hausdorff_mmd_heatmap.png"
    )
    matrix_csv = Path(args.matrix_csv) if args.matrix_csv else input_dir / "dice_matrix.csv"
    kappa_matrix_csv = (
        Path(args.kappa_matrix_csv) if args.kappa_matrix_csv else input_dir / "kappa_matrix.csv"
    )
    hausdorff_matrix_csv = (
        Path(args.hausdorff_matrix_csv)
        if args.hausdorff_matrix_csv
        else input_dir / "hausdorff_matrix.csv"
    )
    mmd_matrix_csv = (
        Path(args.mmd_matrix_csv) if args.mmd_matrix_csv else input_dir / "mmd_matrix.csv"
    )
    paths = _select_paths(input_dir, args.pattern, args.include, args.exclude, args.recursive)
    if not paths:
        raise SystemExit(f"No files matched in {input_dir} with pattern {args.pattern}.")

    paths = sorted(paths, key=_sort_key)
    labels = [p.stem.replace(".nii", "") for p in paths]
    if args.label_style == "short":
        labels = [_label_from_name(label) for label in labels]

    masks = []
    shapes = []
    zooms = []
    for path in paths:
        mask, shape, img_zooms = _load_mask(path, args.threshold, args.percentile, args.abs)
        masks.append(mask)
        shapes.append(shape)
        zooms.append(img_zooms)

    if len(set(shapes)) != 1:
        raise SystemExit(f"Input volumes have different shapes: {sorted(set(shapes))}")
    if len({tuple(z) for z in zooms}) != 1:
        raise SystemExit("Input volumes have different voxel sizes; distances would be inconsistent.")

    sampling = tuple(float(z) for z in zooms[0])
    dist_to_masks = [_distance_to_mask(mask, sampling) for mask in masks]

    n = len(paths)
    matrix = np.zeros((n, n), dtype=np.float32)
    kappa_matrix = np.zeros((n, n), dtype=np.float32)
    hausdorff_matrix = np.zeros((n, n), dtype=np.float32)
    mmd_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        matrix[i, i] = 1.0
        kappa_matrix[i, i] = _cohens_kappa(masks[i], masks[i])
        hausdorff_matrix[i, i] = 0.0 if np.any(masks[i]) else np.nan
        mmd_matrix[i, i] = 0.0 if np.any(masks[i]) else np.nan
        for j in range(i + 1, n):
            dice_val = _dice(masks[i], masks[j])
            kappa_val = _cohens_kappa(masks[i], masks[j])
            hausdorff_val, mmd_val = _distance_metrics(
                masks[i], masks[j], dist_to_masks[i], dist_to_masks[j]
            )
            matrix[i, j] = dice_val
            matrix[j, i] = dice_val
            kappa_matrix[i, j] = kappa_val
            kappa_matrix[j, i] = kappa_val
            hausdorff_matrix[i, j] = hausdorff_val
            hausdorff_matrix[j, i] = hausdorff_val
            mmd_matrix[i, j] = mmd_val
            mmd_matrix[j, i] = mmd_val

    _write_csv(matrix, labels, matrix_csv)
    _write_csv(kappa_matrix, labels, kappa_matrix_csv)
    _write_csv(hausdorff_matrix, labels, hausdorff_matrix_csv)
    _write_csv(mmd_matrix, labels, mmd_matrix_csv)
    _plot_metric_heatmaps(
        matrix, kappa_matrix, hausdorff_matrix, mmd_matrix, labels, output_path, args.dpi
    )

    print(f"Wrote DSC matrix: {matrix_csv}")
    print(f"Wrote kappa matrix: {kappa_matrix_csv}")
    print(f"Wrote Hausdorff matrix: {hausdorff_matrix_csv}")
    print(f"Wrote MMD matrix: {mmd_matrix_csv}")
    print(f"Wrote heatmaps: {output_path}")


if __name__ == "__main__":
    main()
