#!/usr/bin/env python3
"""Wrapper for the requested hemisphere-specific atlas with connectivity_new defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the requested hemisphere-specific ROI atlas under results/connectivity_new."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "results" / "connectivity" / "data",
        help="Input directory containing selected beta files and voxel indices.",
    )
    parser.add_argument(
        "--anat-path",
        type=Path,
        default=REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "MNI152_T1_2mm_brain.nii.gz",
        help="Anatomy image in MNI space.",
    )
    parser.add_argument(
        "--voxel-indices-path",
        type=Path,
        default=REPO_ROOT / "results" / "connectivity" / "data" / "selected_voxel_indices.npz",
        help="Selected-voxel NPZ.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "connectivity_new" / "atlas_requested_hemi",
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "main" / "connectivity_metrics_requested_hemi.py"),
        "--data-dir",
        str(args.data_dir.expanduser().resolve()),
        "--anat-path",
        str(args.anat_path.expanduser().resolve()),
        "--voxel-indices-path",
        str(args.voxel_indices_path.expanduser().resolve()),
        "--out-dir",
        str(args.out_dir.expanduser().resolve()),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
