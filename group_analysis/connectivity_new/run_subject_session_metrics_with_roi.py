#!/usr/bin/env python3
"""Run subject/session ROI-edge metrics with an explicit ROI atlas path."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from group_analysis.main.run_subject_session_metrics import (  # noqa: E402
    build_subject_session_splits,
    ensure_voxel_indices_npz,
    write_split_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the selected beta matrix by subject/session or subject/session/run "
            "and run roi_edge_connectivity.py using a caller-specified ROI atlas."
        )
    )
    parser.add_argument(
        "--input-beta",
        type=Path,
        default=Path("results/connectivity/data/selected_beta_trials.npy"),
        help="Global beta matrix path.",
    )
    parser.add_argument(
        "--input-voxel-indices",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ with selected_ijk or selected_flat_indices.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results/connectivity/data/concat_manifest_group.tsv"),
        help="Trial manifest TSV.",
    )
    parser.add_argument(
        "--roi-img",
        type=Path,
        required=True,
        help="ROI label NIfTI to use for ROI aggregation.",
    )
    parser.add_argument(
        "--roi-summary",
        type=Path,
        default=None,
        help="Optional ROI summary JSON matching --roi-img.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root. Writes <out-root>/data and <out-root>/roi_edge_network.",
    )
    parser.add_argument(
        "--advanced-metrics",
        default="mutual_information_ksg",
        help="Comma-separated advanced metrics for roi_edge_connectivity.py.",
    )
    parser.add_argument(
        "--allow-partial-runs",
        action="store_true",
        help="Include subject/session labels with only one run.",
    )
    parser.add_argument(
        "--preserve-runs",
        action="store_true",
        help="Write one split beta file per subject/session/run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite split files and copied voxel indices if they already exist.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Row chunk size when writing split beta matrices.",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Only create split beta files and metadata.",
    )
    parser.add_argument(
        "--save-split-summary",
        action="store_true",
        help="Write subject_session_split_summary.csv under <out-root>/data.",
    )
    parser.add_argument(
        "--save-column-map",
        action="store_true",
        help="Write the split column-index NPZ under <out-root>/data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_beta = args.input_beta.expanduser().resolve()
    input_voxel_indices = args.input_voxel_indices.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()
    roi_img = args.roi_img.expanduser().resolve()
    roi_summary = (
        args.roi_summary.expanduser().resolve()
        if args.roi_summary is not None
        else roi_img.parent / "requested_hemi_roi_summary.json"
    )
    out_root = args.out_root.expanduser().resolve()

    if not input_beta.exists():
        raise FileNotFoundError(f"Input beta file not found: {input_beta}")
    if not input_voxel_indices.exists():
        raise FileNotFoundError(f"Input voxel-index file not found: {input_voxel_indices}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    if not roi_img.exists():
        raise FileNotFoundError(f"ROI image not found: {roi_img}")
    if roi_summary is not None and not roi_summary.exists():
        raise FileNotFoundError(f"ROI summary not found: {roi_summary}")

    data_dir = out_root / "data"
    roi_out_dir = out_root / "roi_edge_network"
    data_dir.mkdir(parents=True, exist_ok=True)
    roi_out_dir.mkdir(parents=True, exist_ok=True)

    beta = np.load(input_beta, mmap_mode="r")
    n_vox, n_trials = beta.shape

    splits = build_subject_session_splits(
        manifest_path=manifest_path,
        n_trials_total=int(n_trials),
        allow_partial_runs=bool(args.allow_partial_runs),
        preserve_runs=bool(args.preserve_runs),
    )
    split_df = pd.DataFrame(
        [
            {
                "label": split.label,
                "subject": split.subject,
                "session": split.session,
                "runs": ",".join(map(str, split.runs)),
                "n_runs": len(split.runs),
                "n_trials": int(split.columns.size),
            }
            for split in splits
        ]
    )
    if args.save_split_summary:
        split_df.to_csv(data_dir / "subject_session_split_summary.csv", index=False)

    print(
        "Subject-session splits: "
        f"{len(splits)} labels across {split_df['subject'].nunique()} subjects",
        flush=True,
    )

    write_split_files(
        input_beta=input_beta,
        data_dir=data_dir,
        splits=splits,
        chunk_size=int(max(1, args.chunk_size)),
        overwrite=bool(args.overwrite),
        save_column_map=bool(args.save_column_map),
    )

    voxel_indices_npz, _ = ensure_voxel_indices_npz(
        input_voxel_indices=input_voxel_indices,
        data_dir=data_dir,
        expected_n_voxels=int(n_vox),
        overwrite=bool(args.overwrite),
    )

    if not args.skip_connectivity:
        cmd = [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "connectivity_new" / "roi_edge_connectivity_requested.py"),
            "--roi-img",
            str(roi_img),
            "--roi-summary",
            str(roi_summary),
            "--data-dir",
            str(data_dir),
            "--beta-pattern",
            (
                "selected_beta_trials_sub-*_ses-*_run-*.npy"
                if args.preserve_runs
                else "selected_beta_trials_sub-*_ses-*.npy"
            ),
            "--voxel-indices-path",
            str(voxel_indices_npz),
            "--out-dir",
            str(roi_out_dir),
            "--advanced-metrics-out-subdir",
            "advanced_metrics",
            "--advanced-metrics",
            str(args.advanced_metrics),
            "--force-no-split-hemispheres",
        ]
        print("Running per-subject/session ROI-edge pipeline:", flush=True)
        print("  " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    run_manifest = {
        "input_beta": str(input_beta),
        "input_voxel_indices": str(input_voxel_indices),
        "manifest_path": str(manifest_path),
        "roi_img": str(roi_img),
        "roi_summary": str(roi_summary),
        "out_root": str(out_root),
        "n_subject_session_labels": int(len(splits)),
        "n_unique_subjects": int(split_df["subject"].nunique()),
        "advanced_metrics": str(args.advanced_metrics),
        "allow_partial_runs": bool(args.allow_partial_runs),
        "preserve_runs": bool(args.preserve_runs),
        "skip_connectivity": bool(args.skip_connectivity),
    }
    (out_root / "run_manifest_subject_session.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
