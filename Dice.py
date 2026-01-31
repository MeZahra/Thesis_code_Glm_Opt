#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


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
) -> Tuple[np.ndarray, tuple]:
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
    return mask, img.shape


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    overlap = np.count_nonzero(mask_a & mask_b)
    denom = int(mask_a.sum()) + int(mask_b.sum())
    return (2.0 * overlap / denom) if denom else np.nan


def _voe(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    overlap = np.count_nonzero(mask_a & mask_b)
    union = np.count_nonzero(mask_a | mask_b)
    return (1.0 - (overlap / union)) if union else np.nan


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


def _plot_combined_heatmaps(
    dice_matrix: np.ndarray,
    voe_matrix: np.ndarray,
    labels: list,
    out_path: Path,
    dpi: int,
) -> None:
    n = len(labels)
    base = max(6, min(16, 0.4 * n))
    fig, axes = plt.subplots(1, 2, figsize=(2 * base, base))

    im_dice = axes[0].imshow(dice_matrix, vmin=0, vmax=1, cmap="magma")
    axes[0].set_title("Dice overlap", fontsize=10)
    _format_heatmap_axis(axes[0], labels)
    cbar_dice = fig.colorbar(im_dice, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_dice.set_label("Dice overlap")

    im_voe = axes[1].imshow(voe_matrix, vmin=0, vmax=1, cmap="magma_r")
    axes[1].set_title("Volumetric overlap error", fontsize=10)
    _format_heatmap_axis(axes[1], labels)
    cbar_voe = fig.colorbar(im_voe, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_voe.set_label("VOE (1 - Jaccard)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute pairwise Dice Similarity Coefficient (DSC) and volumetric overlap error (VOE) "
            "for NIfTI networks and plot heatmaps."
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
        help="Path for combined heatmap image (Dice + VOE). Defaults to <input-dir>/dice_voe_heatmap.png.",
    )
    parser.add_argument(
        "--matrix-csv",
        default=None,
        help="Path for CSV output of the DSC matrix. Defaults to <input-dir>/dice_matrix.csv.",
    )
    parser.add_argument(
        "--voe-matrix-csv",
        default=None,
        help="Path for CSV output of the VOE matrix. Defaults to <input-dir>/voe_matrix.csv.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved heatmap.",
    )
    args = parser.parse_args()

    input_dir = _resolve_input_dir(Path(args.input_dir), args.subject, args.session)
    output_path = Path(args.output) if args.output else input_dir / "dice_voe_heatmap.png"
    matrix_csv = Path(args.matrix_csv) if args.matrix_csv else input_dir / "dice_matrix.csv"
    voe_matrix_csv = Path(args.voe_matrix_csv) if args.voe_matrix_csv else input_dir / "voe_matrix.csv"
    paths = _select_paths(input_dir, args.pattern, args.include, args.exclude, args.recursive)
    if not paths:
        raise SystemExit(f"No files matched in {input_dir} with pattern {args.pattern}.")

    paths = sorted(paths, key=_sort_key)
    labels = [p.stem.replace(".nii", "") for p in paths]
    if args.label_style == "short":
        labels = [_label_from_name(label) for label in labels]

    masks = []
    shapes = []
    for path in paths:
        mask, shape = _load_mask(path, args.threshold, args.percentile, args.abs)
        masks.append(mask)
        shapes.append(shape)

    if len(set(shapes)) != 1:
        raise SystemExit(f"Input volumes have different shapes: {sorted(set(shapes))}")

    n = len(paths)
    matrix = np.zeros((n, n), dtype=np.float32)
    voe_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        matrix[i, i] = 1.0
        voe_matrix[i, i] = 0.0
        for j in range(i + 1, n):
            dice_val = _dice(masks[i], masks[j])
            voe_val = _voe(masks[i], masks[j])
            matrix[i, j] = dice_val
            matrix[j, i] = dice_val
            voe_matrix[i, j] = voe_val
            voe_matrix[j, i] = voe_val

    _write_csv(matrix, labels, matrix_csv)
    _write_csv(voe_matrix, labels, voe_matrix_csv)
    _plot_combined_heatmaps(matrix, voe_matrix, labels, output_path, args.dpi)

    print(f"Wrote DSC matrix: {matrix_csv}")
    print(f"Wrote VOE matrix: {voe_matrix_csv}")
    print(f"Wrote heatmaps: {output_path}")


if __name__ == "__main__":
    main()
