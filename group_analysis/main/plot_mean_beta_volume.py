#!/usr/bin/env python3
"""Plot the mean trial-wise beta volume as an interactive Nilearn view."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import plotting


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BETA_PATH = Path(
    "/Data/zahra/results_beta_preprocessed/sub-pd004/"
    "cleaned_beta_volume_sub-pd004_ses-1_run-1.npy"
)
DEFAULT_REFERENCE_IMG = (
    REPO_ROOT
    / "results"
    / "single_run_task_glm"
    / "sub-pd004_ses-1_run-1"
    / "task_z_score_fdr05_pos.nii.gz"
)
DEFAULT_OUTPUT_HTML = (
    REPO_ROOT
    / "results"
    / "single_run_task_glm"
    / "sub-pd004_ses-1_run-1"
    / "mean_beta_volume_interactive.html"
)


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


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
        default=DEFAULT_OUTPUT_HTML,
        help=f"Output interactive HTML file. Default: {DEFAULT_OUTPUT_HTML}",
    )
    parser.add_argument(
        "--title",
        default="Mean beta volume",
        help="Title shown in the interactive viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    beta_path = _require_file(args.beta_path, "beta .npy file")
    reference_img_path = _require_file(args.reference_img, "reference NIfTI image")
    output_html = args.output_html.expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    beta = np.load(beta_path, mmap_mode="r")
    if beta.ndim != 4:
        raise ValueError(f"Expected a 4D beta array, got shape {beta.shape}")

    mean_beta = np.nanmean(beta, axis=-1, dtype=np.float32)
    mean_beta = np.nan_to_num(mean_beta, nan=0.0, posinf=0.0, neginf=0.0)

    reference_img = nib.load(str(reference_img_path))
    if mean_beta.shape != reference_img.shape[:3]:
        raise ValueError(
            f"Shape mismatch: mean beta shape {mean_beta.shape} vs "
            f"reference image shape {reference_img.shape[:3]}"
        )

    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    mean_beta_img = nib.Nifti1Image(mean_beta, reference_img.affine, header)

    view = plotting.view_img(
        mean_beta_img,
        title=args.title,
        threshold=1e-6,
        colorbar=True,
        symmetric_cmap=False,
        cmap="hot",
    )
    view.save_as_html(output_html)

    print(f"Loaded beta array: {beta_path}")
    print(f"Input shape: {beta.shape}")
    print(f"Mean beta shape: {mean_beta.shape}")
    print(f"Saved interactive HTML: {output_html}")


if __name__ == "__main__":
    main()
