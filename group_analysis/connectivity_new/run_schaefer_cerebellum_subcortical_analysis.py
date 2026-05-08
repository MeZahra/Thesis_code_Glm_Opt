#!/usr/bin/env python3
"""Run the Schaefer+cerebellum+subcortical mutual-information analysis."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "connectivity_new" / "schaefer100_cerebellum_subcortical"
ATLAS_DIR = REPO_ROOT / "results" / "connectivity_new" / "atlas_schaefer100_cerebellum_subcortical"
ATLAS_IMG = ATLAS_DIR / "schaefer100_cereb_subcort_rois_fitted.nii.gz"
ATLAS_SUMMARY = ATLAS_DIR / "schaefer100_cereb_subcort_roi_summary.json"
NETWORK_DIR = RESULTS_ROOT / "roi_edge_network"
ADVANCED_ROOT = NETWORK_DIR / "advanced_metrics"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Schaefer+cerebellum+subcortical atlas and rerun the "
            "subject-session KSG mutual-information ROI-network analysis."
        )
    )
    parser.add_argument("--schaefer-n-rois", type=int, default=100)
    parser.add_argument("--skip-atlas", action="store_true")
    parser.add_argument("--skip-connectivity", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    parser.add_argument("--skip-top-edges", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def main() -> None:
    args = _parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    atlas_dir = ATLAS_DIR
    atlas_img = ATLAS_IMG
    atlas_summary = ATLAS_SUMMARY
    if int(args.schaefer_n_rois) != 100:
        atlas_dir = REPO_ROOT / "results" / "connectivity_new" / f"atlas_schaefer{int(args.schaefer_n_rois)}_cerebellum_subcortical"
        atlas_img = atlas_dir / f"schaefer{int(args.schaefer_n_rois)}_cereb_subcort_rois_fitted.nii.gz"
        atlas_summary = atlas_dir / f"schaefer{int(args.schaefer_n_rois)}_cereb_subcort_roi_summary.json"

    if not args.skip_atlas:
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "group_analysis" / "connectivity_new" / "build_schaefer_cerebellum_subcortical_atlas.py"),
                "--schaefer-n-rois",
                str(int(args.schaefer_n_rois)),
                "--out-dir",
                str(atlas_dir),
            ]
        )

    if not args.skip_connectivity:
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "group_analysis" / "connectivity_new" / "roi_edge_connectivity_requested.py"),
                "--data-dir",
                str(REPO_ROOT / "results" / "connectivity_new" / "data"),
                "--beta-pattern",
                "selected_beta_trials_sub-*_ses-*.npy",
                "--roi-img",
                str(atlas_img),
                "--roi-summary",
                str(atlas_summary),
                "--voxel-indices-path",
                str(REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "selected_voxel_indices.npz"),
                "--out-dir",
                str(NETWORK_DIR),
                "--advanced-metrics",
                "mutual_information_ksg",
                "--force-no-split-hemispheres",
                "--no-html",
                "--no-connectome-html",
                "--skip-edge-correlation-network",
            ]
        )

    if not args.skip_pairwise:
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "group_analysis" / "main" / "analyze_pairwise_metric_separation.py"),
                "--advanced-root",
                str(ADVANCED_ROOT),
                "--out-dir",
                str(NETWORK_DIR),
                "--metrics",
                "mutual_information_ksg",
                "--comparison-metrics",
                "laplacian_spectral_distance_signed",
            ]
        )

    if not args.skip_top_edges:
        _run(
            [
                sys.executable,
                str(REPO_ROOT / "group_analysis" / "main" / "extract_top_session_delta_edges.py"),
                "--advanced-root",
                str(ADVANCED_ROOT),
                "--metric",
                "mutual_information_ksg",
                "--out-dir",
                str(NETWORK_DIR / "mutual_information_ksg"),
                "--top-percentile",
                "100",
                "--top-k",
                "25",
            ]
        )

    print("\nSchaefer+cerebellum+subcortical analysis complete.", flush=True)
    print(f"Atlas summary: {atlas_summary}", flush=True)
    print(f"ROI-edge outputs: {NETWORK_DIR}", flush=True)
    print(
        "Main distribution plot: "
        f"{NETWORK_DIR / 'mutual_information_ksg' / 'cross_subject_only_laplacian_spectral_distance_signed_distribution.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
