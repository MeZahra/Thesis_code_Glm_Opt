#!/usr/bin/env python3
"""Recalculate requested connectivity analyses under results/connectivity_new."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "connectivity_new"
ATLAS_DIR = RESULTS_ROOT / "atlas_requested_hemi"
ATLAS_IMG = ATLAS_DIR / "requested_hemi_rois_fitted.nii.gz"
ATLAS_SUMMARY = ATLAS_DIR / "requested_hemi_roi_summary.json"

SESSION_OUT_ROOT = RESULTS_ROOT
SESSION_ADVANCED_ROOT = SESSION_OUT_ROOT / "roi_edge_network" / "advanced_metrics"
SESSION_FIG_ROOT = SESSION_OUT_ROOT / "roi_edge_network"

RUN_LEVEL_OUT_ROOT = RESULTS_ROOT / "GPT" / "tmp" / "run_level_partial_correlation_metrics"
RUN_LEVEL_ADVANCED_ROOT = RUN_LEVEL_OUT_ROOT / "roi_edge_network" / "advanced_metrics"
GRAPH_BASE_OUT = RESULTS_ROOT / "GPT" / "tmp" / "roi_graph_analysis_requested_atlas"
GRAPH_HEMI_OUT = RESULTS_ROOT / "GPT" / "tmp" / "roi_graph_analysis_runaveraged_anatomical_hemispheric"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the requested atlas and rerun the requested connectivity, graph, "
            "and PLS analyses into results/connectivity_new."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow downstream split/connectivity stages to overwrite existing outputs.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("\n[cmd]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _patch_requested_anatomical_systems():
    gpt_root = REPO_ROOT / "group_analysis" / "GPT"
    if str(gpt_root) not in sys.path:
        sys.path.insert(0, str(gpt_root))

    import common_io  # type: ignore
    import roi_graph_runaveraged_analysis as base_graph  # type: ignore
    import roi_graph_runaveraged_hemispheric_analysis as hemi_graph  # type: ignore

    def list_requested_runlevel_subjects(metric: str = "partial_correlation") -> list[str]:
        subjects: set[str] = set()
        for run_dir in (RUN_LEVEL_ADVANCED_ROOT).glob("sub-pd*_ses-*_run-*"):
            metric_dir = run_dir / metric
            if not metric_dir.exists():
                continue
            matrix_path = metric_dir / f"{metric}.npy"
            if not matrix_path.exists():
                continue
            subject = run_dir.name.split("_ses-")[0]
            subjects.add(subject)
        return sorted(subjects)

    def infer_requested_system(base_roi: str) -> str:
        if base_roi in {
            "Frontal Pole",
            "Insular cortex",
            "Cerebral Cortex",
        }:
            return "cognitive_control"
        if base_roi in {
            "Precentral gyrus (primary motor cortex)",
            "Frontal medial cortex (SMA/pre-SMA)",
            "Superior parietal lobule",
            "Cerebellum (lobules VIIIa, VIIb)",
            "Cerebellar Crus II",
        }:
            return "motor_sensorimotor"
        if base_roi in {"Putamen (basal ganglia)"}:
            return "subcortical_relay"
        if base_roi in {"Hippocampus", "Parahippocampal gyrus", "Amygdala"}:
            return "limbic_memory"
        if base_roi in {
            "Precuneus",
            "Temporal cortex",
            "Fusiform cortex",
            "Lateral occipital cortex",
        }:
            return "posterior_perceptual"
        return "other_relative"

    common_io.TMP_ROI_ROOT = RUN_LEVEL_OUT_ROOT / "roi_edge_network"
    common_io.infer_anatomical_system = infer_requested_system
    common_io.list_paired_subjects_for_metric = list_requested_runlevel_subjects
    base_graph.infer_anatomical_system = infer_requested_system
    base_graph.list_paired_subjects_for_metric = list_requested_runlevel_subjects
    hemi_graph.infer_anatomical_system = infer_requested_system
    hemi_graph.list_paired_subjects_for_metric = list_requested_runlevel_subjects
    return base_graph, hemi_graph


def main() -> None:
    args = _parse_args()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("Rebuilding requested connectivity analyses under results/connectivity_new", flush=True)
    print("=" * 80, flush=True)

    print("\n[1] Building requested hemisphere-specific atlas ...", flush=True)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "connectivity_new" / "build_requested_hemi_atlas.py"),
            "--out-dir",
            str(ATLAS_DIR),
        ]
    )

    print("\n[2] Rebuilding subject/session mutual-information KSG metrics ...", flush=True)
    session_cmd = [
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "connectivity_new" / "roi_edge_connectivity_requested.py"),
        "--data-dir",
        str(REPO_ROOT / "results" / "connectivity" / "data"),
        "--beta-pattern",
        "selected_beta_trials_sub-*_ses-*.npy",
        "--roi-img",
        str(ATLAS_IMG),
        "--roi-summary",
        str(ATLAS_SUMMARY),
        "--voxel-indices-path",
        str(REPO_ROOT / "results" / "connectivity" / "data" / "selected_voxel_indices.npz"),
        "--out-dir",
        str(SESSION_OUT_ROOT / "roi_edge_network"),
        "--advanced-metrics",
        "mutual_information_ksg",
        "--force-no-split-hemispheres",
    ]
    _run(session_cmd)

    print("\n[3] Regenerating mutual_information_ksg separation and top-edge figures ...", flush=True)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "main" / "analyze_pairwise_metric_separation.py"),
            "--advanced-root",
            str(SESSION_ADVANCED_ROOT),
            "--out-dir",
            str(SESSION_FIG_ROOT),
            "--metrics",
            "mutual_information_ksg",
            "--comparison-metrics",
            "laplacian_spectral_distance_signed",
        ]
    )
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "main" / "extract_top_session_delta_edges.py"),
            "--advanced-root",
            str(SESSION_ADVANCED_ROOT),
            "--metric",
            "mutual_information_ksg",
            "--out-dir",
            str(SESSION_FIG_ROOT / "mutual_information_ksg"),
        ]
    )

    print("\n[4] Rebuilding run-level partial-correlation metrics for graph analyses ...", flush=True)
    run_level_cmd = [
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "connectivity_new" / "roi_edge_connectivity_requested.py"),
        "--data-dir",
        str(REPO_ROOT / "results" / "connectivity" / "GPT" / "tmp" / "run_level_partial_correlation_metrics" / "data"),
        "--beta-pattern",
        "selected_beta_trials_sub-*_ses-*_run-*.npy",
        "--roi-img",
        str(ATLAS_IMG),
        "--roi-summary",
        str(ATLAS_SUMMARY),
        "--voxel-indices-path",
        str(REPO_ROOT / "results" / "connectivity" / "GPT" / "tmp" / "run_level_partial_correlation_metrics" / "data" / "selected_voxel_indices.npz"),
        "--out-dir",
        str(RUN_LEVEL_OUT_ROOT / "roi_edge_network"),
        "--advanced-metrics",
        "partial_correlation",
        "--force-no-split-hemispheres",
    ]
    _run(run_level_cmd)

    print("\n[5] Running requested graph reorganization analyses ...", flush=True)
    base_graph, hemi_graph = _patch_requested_anatomical_systems()
    base_graph.run_roi_graph_runaveraged(
        out_dir=GRAPH_BASE_OUT,
        metrics_root=RUN_LEVEL_ADVANCED_ROOT,
        roi_subset=None,
    )
    hemi_graph.run_roi_graph_runaveraged_hemispheric(
        out_dir=GRAPH_HEMI_OUT,
        metrics_root=RUN_LEVEL_ADVANCED_ROOT,
        roi_subset_base=None,
    )

    print("\n[6] Running requested-atlas targeted PLS search ...", flush=True)
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "connectivity_new" / "targeted_pls_requested_atlas.py"),
        ]
    )

    print("\nPipeline complete.", flush=True)
    print(f"Atlas: {ATLAS_DIR}", flush=True)
    print(f"ROI-edge figures: {SESSION_FIG_ROOT / 'mutual_information_ksg'}", flush=True)
    print(f"Hemispheric hub figure: {GRAPH_HEMI_OUT / 'hub_reorganization.png'}", flush=True)
    print(f"PLS root: {RESULTS_ROOT / 'PLS'}", flush=True)


if __name__ == "__main__":
    main()
