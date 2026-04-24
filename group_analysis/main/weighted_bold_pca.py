#!/usr/bin/env python3
"""Project weighted BOLD time series into PCA space for each subject/run."""

from __future__ import annotations

import argparse
import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import BinaryIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BOLD_DIR = Path("/Data/zahra/bold_data")
DEFAULT_WEIGHT_MAP = (
    REPO_ROOT
    / "results"
    / "ablation"
    / "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_thr90.nii.gz"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "weighted_bold_pca"
EXPECTED_BOLD_SHAPE = (91, 109, 91, 850)
EXPECTED_WEIGHT_SHAPE = (91, 109, 91)
DROP_SLICES = (slice(270, 290), slice(560, 580))
MEDICATION_LABELS = {1: "Medication OFF", 2: "Medication ON"}
RUN_COLORS = {
    (1, 1): "#1f77b4",
    (1, 2): "#ff7f0e",
    (2, 1): "#2ca02c",
    (2, 2): "#d62728",
}
FILENAME_RE = re.compile(
    r"(?P<subject>sub-[^_]+)_ses-(?P<session>\d+)_run-(?P<run>\d+)_task-mv_"
    r"bold_corrected_smoothed_mnireg-2mm\.nii(?:\.gz)?$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each BOLD NIfTI in /Data/zahra/bold_data, compute a weighted "
            "time series, remove two 20-volume periods, reshape to 90x9, run "
            "PCA(n_components=2), and save one 2x2 figure per subject."
        )
    )
    parser.add_argument("--bold-dir", type=Path, default=DEFAULT_BOLD_DIR)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def parse_bold_name(path: Path) -> tuple[str, int, int]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse subject/session/run from {path.name}")
    return (
        match.group("subject"),
        int(match.group("session")),
        int(match.group("run")),
    )


def load_weights(path: Path) -> np.ndarray:
    weights_img = nib.load(str(path))
    if weights_img.shape != EXPECTED_WEIGHT_SHAPE:
        raise ValueError(
            f"Expected weight map shape {EXPECTED_WEIGHT_SHAPE}, got {weights_img.shape}: {path}"
        )
    return weights_img.get_fdata(dtype=np.float32)


def compute_weighted_ts(bold_path: Path, weights: np.ndarray) -> np.ndarray:
    bold_img = nib.load(str(bold_path))
    if bold_img.shape != EXPECTED_BOLD_SHAPE:
        raise ValueError(f"Expected BOLD shape {EXPECTED_BOLD_SHAPE}, got {bold_img.shape}: {bold_path}")
    if bold_img.dataobj.order != "F":
        raise ValueError(f"Expected Fortran-ordered NIfTI data, got order={bold_img.dataobj.order}: {bold_path}")

    flat_weights = weights.reshape(-1, order="F")
    weight_mask = np.isfinite(flat_weights) & (flat_weights != 0)
    weight_indices = np.flatnonzero(weight_mask)
    selected_weights = flat_weights[weight_indices].astype(np.float64, copy=False)
    weights_sum = float(selected_weights.sum())

    proxy = bold_img.dataobj
    dtype = np.dtype(proxy.dtype)
    slope = 1.0 if proxy.slope is None else float(proxy.slope)
    intercept = 0.0 if proxy.inter is None else float(proxy.inter)
    n_voxels = int(np.prod(EXPECTED_WEIGHT_SHAPE))
    n_timepoints = EXPECTED_BOLD_SHAPE[-1]
    volume_nbytes = n_voxels * dtype.itemsize
    weighted_ts = np.empty(n_timepoints, dtype=np.float64)

    with open_nifti_stream(bold_path) as stream:
        header_bytes = stream.read(int(proxy.offset))
        if len(header_bytes) != int(proxy.offset):
            raise EOFError(f"Could not read NIfTI header bytes from {bold_path}")

        for time_index in range(n_timepoints):
            buffer = read_exact(stream, volume_nbytes, bold_path, time_index)
            volume = np.frombuffer(buffer, dtype=dtype, count=n_voxels)
            # Equivalent to np.sum(bold * weights[..., None], axis=(0, 1, 2)).
            weighted_ts[time_index] = slope * (selected_weights @ volume[weight_indices]) + intercept * weights_sum

    return weighted_ts


def open_nifti_stream(path: Path) -> BinaryIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def read_exact(stream: BinaryIO, n_bytes: int, path: Path, time_index: int) -> bytes:
    buffer = stream.read(n_bytes)
    if len(buffer) != n_bytes:
        raise EOFError(f"Could not read full volume {time_index} from {path}")
    return buffer


def clean_and_reshape(weighted_ts: np.ndarray) -> np.ndarray:
    keep_mask = np.ones(weighted_ts.shape[0], dtype=bool)
    for drop_slice in DROP_SLICES:
        keep_mask[drop_slice] = False
    weighted_ts_clean = weighted_ts[keep_mask]
    if weighted_ts_clean.shape != (810,):
        raise ValueError(f"Expected cleaned time series shape (810,), got {weighted_ts_clean.shape}")
    return weighted_ts_clean.reshape(90, 9)


def run_pca(weighted_ts_90x9: np.ndarray) -> np.ndarray:
    return PCA(n_components=2).fit_transform(weighted_ts_90x9)


def add_identity_line(ax: plt.Axes) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    low = min(xlim[0], ylim[0])
    high = max(xlim[1], ylim[1])
    ax.plot([low, high], [low, high], linestyle="--", color="0.35", linewidth=1.0)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)


def save_subject_figure(
    subject: str,
    run_results: list[tuple[int, int, np.ndarray]],
    out_dir: Path,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    flat_axes = axes.ravel()

    for ax, (session, run, scores) in zip(flat_axes, sorted(run_results, key=lambda item: (item[0], item[1]))):
        ax.scatter(scores[:, 0], scores[:, 1], s=18, alpha=0.8)
        ax.axhline(0, color="0.85", linewidth=0.8)
        ax.axvline(0, color="0.85", linewidth=0.8)
        add_identity_line(ax)
        ax.set_title(f"ses-{session} run-{run}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    for ax in flat_axes[len(run_results) :]:
        ax.set_visible(False)

    fig.suptitle(subject)
    out_path = out_dir / f"{subject}_weighted_bold_pca.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def concatenate_session_runs(
    session_runs: dict[int, dict[int, np.ndarray]],
) -> list[tuple[int, np.ndarray, int]]:
    concatenated: list[tuple[int, np.ndarray, int]] = []
    for session, run_matrices in sorted(session_runs.items()):
        matrices = [run_matrices[run] for run in sorted(run_matrices)]
        concatenated.append((session, np.vstack(matrices), len(matrices)))
    return concatenated


def iter_session_run_matrices(
    session_runs: dict[int, dict[int, np.ndarray]],
) -> list[tuple[int, int, np.ndarray]]:
    rows: list[tuple[int, int, np.ndarray]] = []
    for session, run_matrices in sorted(session_runs.items()):
        for run, matrix in sorted(run_matrices.items()):
            rows.append((session, run, matrix))
    return rows


def save_medication_overlay_figure(
    subject: str,
    session_runs: dict[int, dict[int, np.ndarray]],
    out_dir: Path,
    dpi: int,
) -> Path | None:
    session_matrices = concatenate_session_runs(session_runs)
    if not session_matrices:
        return None

    pca = PCA(n_components=2).fit(np.vstack([matrix for _, matrix, _ in session_matrices]))
    run_scores = [
        (session, run, pca.transform(matrix))
        for session, run, matrix in iter_session_run_matrices(session_runs)
    ]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for session, run, scores in run_scores:
        medication = MEDICATION_LABELS.get(session, f"ses-{session}")
        color = RUN_COLORS.get((session, run), "0.35")
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            s=22,
            alpha=0.75,
            color=color,
            label=f"{medication} run-{run}",
        )

    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_title(f"{subject}: concatenated-run medication PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)

    out_path = out_dir / f"{subject}_weighted_bold_pca_medication_overlay.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    weights = load_weights(args.weight_map)
    bold_paths = sorted(args.bold_dir.glob("*_bold_corrected_smoothed_mnireg-2mm.nii.gz"))
    if not bold_paths:
        raise FileNotFoundError(f"No BOLD NIfTI files found in {args.bold_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_by_subject: dict[str, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    matrices_by_subject_session: dict[str, dict[int, dict[int, np.ndarray]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    for bold_path in bold_paths:
        subject, session, run = parse_bold_name(bold_path)
        weighted_ts = compute_weighted_ts(bold_path, weights)
        weighted_ts_90x9 = clean_and_reshape(weighted_ts)
        scores = run_pca(weighted_ts_90x9)
        results_by_subject[subject].append((session, run, scores))
        matrices_by_subject_session[subject][session][run] = weighted_ts_90x9
        print(f"Processed {bold_path.name}: weighted_ts={weighted_ts.shape}, pca_scores={scores.shape}")

    for subject, run_results in sorted(results_by_subject.items()):
        out_path = save_subject_figure(subject, run_results, args.out_dir, args.dpi)
        print(f"Saved {out_path}")
        overlay_path = save_medication_overlay_figure(
            subject,
            matrices_by_subject_session[subject],
            args.out_dir,
            args.dpi,
        )
        if overlay_path is not None:
            print(f"Saved {overlay_path}")


if __name__ == "__main__":
    main()
