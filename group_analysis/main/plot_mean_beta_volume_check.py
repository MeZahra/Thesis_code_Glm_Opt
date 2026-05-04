#!/usr/bin/env python3
"""Plot the mean trial-wise beta volume as an interactive Nilearn view."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import plotting
from scipy import stats
from statsmodels.stats.multitest import multipletests


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BETA_PATH = Path(
    REPO_ROOT
    / "results_beta_preprocessed"
    / "sub-pd004"
    / "cleaned_beta_volume_sub-pd004_ses-1_run-1.npy"
)
DEFAULT_REFERENCE_IMG = (
    REPO_ROOT
    / "results"
    / "single_run_task_glm"
    / "sub-pd004_ses-1_run-1"
    / "task_z_score_fdr05_pos.nii.gz"
)
DEFAULT_BG_IMG = Path("/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz")
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "single_run_task_glm"
    / "sub-pd004_ses-1_run-1"
)


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _percent_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Average a cleaned beta volume across its last dimension and save an "
            "interactive Nilearn HTML view."
        )
    )
    parser.add_argument(
        "--beta-path",
        type=Path,
        default=DEFAULT_BETA_PATH,
        help=f"Input 4D beta .npy file. Default: {DEFAULT_BETA_PATH}",
    )
    parser.add_argument(
        "--reference-img",
        type=Path,
        default=DEFAULT_REFERENCE_IMG,
        help=(
            "Reference NIfTI image used for affine/header information. "
            f"Default: {DEFAULT_REFERENCE_IMG}"
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help=(
            "Output interactive HTML file. Default: "
            f"{DEFAULT_OUTPUT_DIR}/mean_abs_beta_volume_top{{top-percent}}_interactive.html"
        ),
    )
    parser.add_argument(
        "--output-nifti",
        type=Path,
        default=None,
        help="Optional output NIfTI file for the displayed statistic.",
    )
    parser.add_argument(
        "--bg-img",
        type=Path,
        default=DEFAULT_BG_IMG,
        help=f"Anatomical background image for the Nilearn view. Default: {DEFAULT_BG_IMG}",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title shown in the interactive viewer. Default includes --top-percent.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=5.0,
        help=(
            "Show only this top percent of absolute mean-beta voxels. "
            "Used by --stat abs_mean and --stat mean. "
            "Use 100 to show all nonzero voxels. Default: 5."
        ),
    )
    parser.add_argument(
        "--stat",
        choices=("abs_mean", "mean", "positive_z_fdr05"),
        default="abs_mean",
        help=(
            "Statistic to display. positive_z_fdr05 computes a one-sample, "
            "positive, FDR-corrected z map across trial-wise betas."
        ),
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="FDR alpha used by --stat positive_z_fdr05. Default: 0.05.",
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=99.5,
        help="Absolute-value percentile used for the colorbar maximum. Default: 99.5.",
    )
    return parser.parse_args()


def _positive_z_fdr05(beta: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, float | int]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
        mean_beta = np.nanmean(beta, axis=-1, dtype=np.float32)
        std_beta = np.nanstd(beta, axis=-1, ddof=1, dtype=np.float32)

    counts = np.sum(np.isfinite(beta), axis=-1)
    tested = (counts > 1) & np.isfinite(mean_beta) & np.isfinite(std_beta) & (std_beta > 0)
    t_map = np.zeros(mean_beta.shape, dtype=np.float32)
    t_map[tested] = mean_beta[tested] / (std_beta[tested] / np.sqrt(counts[tested]))

    p_values = np.ones(mean_beta.shape, dtype=np.float64)
    p_values[tested] = stats.t.sf(t_map[tested], counts[tested] - 1)

    z_map = np.zeros(mean_beta.shape, dtype=np.float32)
    z_map[tested] = stats.norm.isf(np.clip(p_values[tested], 1e-300, 1.0)).astype(np.float32)

    rejected = np.zeros(mean_beta.shape, dtype=bool)
    if np.any(tested):
        reject_flat, _, _, _ = multipletests(
            p_values[tested],
            alpha=alpha,
            method="fdr_bh",
        )
        rejected[tested] = reject_flat

    z_fdr = np.where(rejected & (z_map > 0), z_map, 0.0).astype(np.float32)
    summary = {
        "tested_voxels": int(np.count_nonzero(tested)),
        "active_voxels_fdr_positive": int(np.count_nonzero(z_fdr > 0)),
        "max_z": float(np.nanmax(z_map)) if np.any(tested) else 0.0,
        "max_z_fdr": float(np.nanmax(z_fdr)) if np.any(z_fdr > 0) else 0.0,
    }
    return z_fdr, summary


def main() -> None:
    args = parse_args()
    beta_path = _require_file(args.beta_path, "beta .npy file")
    reference_img_path = _require_file(args.reference_img, "reference NIfTI image")
    bg_img_path = _require_file(args.bg_img, "background anatomical image")

    if not 0 < args.top_percent <= 100:
        raise ValueError("--top-percent must be > 0 and <= 100.")
    if not 0 < args.vmax_percentile <= 100:
        raise ValueError("--vmax-percentile must be > 0 and <= 100.")
    if not 0 < args.fdr_alpha < 1:
        raise ValueError("--fdr-alpha must be > 0 and < 1.")

    if args.output_html is None:
        if args.stat == "abs_mean":
            output_name = (
                f"mean_abs_beta_volume_top{_percent_label(args.top_percent)}_interactive.html"
            )
        elif args.stat == "mean":
            output_name = f"mean_beta_volume_top{_percent_label(args.top_percent)}_interactive.html"
        else:
            output_name = "glmsingle_beta_positive_z_fdr05_interactive.html"
        output_html = (DEFAULT_OUTPUT_DIR / output_name).resolve()
    else:
        output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    output_nifti = args.output_nifti.expanduser().resolve() if args.output_nifti else None
    if output_nifti is not None:
        output_nifti.parent.mkdir(parents=True, exist_ok=True)

    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected a 4D beta array, got shape {beta.shape}")

    stat_summary: dict[str, float | int] = {}
    if args.stat in {"abs_mean", "mean"}:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            mean_beta = np.nanmean(beta, axis=-1, dtype=np.float32)
        mean_beta = np.nan_to_num(mean_beta, nan=0.0, posinf=0.0, neginf=0.0)

        display_data = np.abs(mean_beta) if args.stat == "abs_mean" else mean_beta
        abs_display_data = np.abs(display_data)
        nonzero_abs_values = abs_display_data[abs_display_data > 0]
        if nonzero_abs_values.size == 0:
            raise ValueError("Beta statistic has no nonzero voxels to display.")

        threshold_percentile = 100.0 - args.top_percent
        threshold = float(np.percentile(nonzero_abs_values, threshold_percentile))
        vmax = float(np.percentile(nonzero_abs_values, args.vmax_percentile))
        if vmax <= threshold:
            vmax = float(nonzero_abs_values.max())
        displayed_voxels = int(np.count_nonzero(abs_display_data >= threshold))
        title = args.title or (
            "Mean absolute beta volume, top "
            f"{args.top_percent:g}% voxels"
            if args.stat == "abs_mean"
            else f"Mean beta volume, top {args.top_percent:g}% voxels"
        )
        symmetric_cmap = args.stat == "mean"
        cmap = "RdBu_r" if args.stat == "mean" else "hot"
        vmin = None if args.stat == "mean" else 0
    else:
        display_data, stat_summary = _positive_z_fdr05(beta, args.fdr_alpha)
        nonzero_values = display_data[display_data > 0]
        if nonzero_values.size == 0:
            raise ValueError("No positive FDR-significant beta z values to display.")
        threshold = 1e-6
        vmax = float(np.percentile(nonzero_values, args.vmax_percentile))
        displayed_voxels = int(np.count_nonzero(display_data > 0))
        title = args.title or f"GLMsingle trial beta positive z, FDR q<{args.fdr_alpha:g}"
        symmetric_cmap = False
        cmap = "hot"
        vmin = 0

    reference_img = nib.load(str(reference_img_path))
    if display_data.shape != reference_img.shape[:3]:
        raise ValueError(
            f"Shape mismatch: beta statistic shape {display_data.shape} vs "
            f"reference image shape {reference_img.shape[:3]}"
        )

    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    mean_beta_img = nib.Nifti1Image(display_data.astype(np.float32), reference_img.affine, header)
    if output_nifti is not None:
        mean_beta_img.to_filename(output_nifti)

    view = plotting.view_img(
        mean_beta_img,
        bg_img=str(bg_img_path),
        title=title,
        threshold=threshold,
        colorbar=True,
        symmetric_cmap=symmetric_cmap,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        resampling_interpolation="nearest",
    )
    view.save_as_html(output_html)

    print(f"Loaded beta array: {beta_path}")
    print(f"Input shape: {beta.shape}")
    print(f"Displayed statistic: {args.stat}")
    print(f"Statistic shape: {display_data.shape}")
    print(f"Display threshold: {threshold:.6g}")
    print(f"Display vmin: {vmin}")
    print(f"Display vmax: {vmax:.6g}")
    print(f"Displayed voxels: {displayed_voxels}")
    for key, value in stat_summary.items():
        print(f"{key}: {value}")
    if output_nifti is not None:
        print(f"Saved NIfTI: {output_nifti}")
    print(f"Saved interactive HTML: {output_html}")


if __name__ == "__main__":
    main()
