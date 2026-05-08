#!/usr/bin/env python3
"""Rebuild the ROI-edge connectivity analysis on the correct 2,674-voxel network.

Two corrections relative to the original roi_edge_network:
  1. Uses results/connectivity/data/selected_voxel_indices.npz (2,674 voxels)
     instead of results/connectivity/tmp/data/ (91,601 voxels).
  2. Removes FSL WM voxels (white prior >= 0.5) upstream and excludes the
     "Unassigned Active Voxels" ROI entirely.

All outputs go to claude_result/ and nothing in the existing analysis is touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image
from numpy.lib.format import open_memmap

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── inputs (read-only) ─────────────────────────────────────────────────────
SOURCE_DATA_DIR   = REPO_ROOT / "results" / "connectivity" / "data"
SOURCE_VOXELS     = SOURCE_DATA_DIR / "selected_voxel_indices.npz"
ROI_IMG           = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_rois_fitted.nii.gz"
ROI_SUMMARY       = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_roi_summary.json"
FSL_WHITE_PRIOR   = Path("/usr/local/fsl/data/standard/tissuepriors/avg152T1_white.img")
WHITE_THRESHOLD   = 0.5
EXCLUDED_SUBJECTS = {"sub-pd017"}   # same as original analysis

# ── outputs ────────────────────────────────────────────────────────────────
OUT_ROOT      = Path(__file__).resolve().parent
DATA_DIR      = OUT_ROOT / "data"
NETWORK_DIR   = OUT_ROOT / "roi_edge_network"
NETWORK_METRIC      = "mutual_information_ksg"
COMPARISON_METRIC   = "laplacian_spectral_distance_signed"


# ── helpers ────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    print("\n[cmd]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _resampled_prior(path: Path, ref_img: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine):
        img = image.resample_to_img(img, ref_img, interpolation="continuous",
                                    force_resample=True, copy_header=True)
    return np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))


# ── step 1: build filtered voxel set ──────────────────────────────────────

def build_filtered_data() -> tuple[np.ndarray, int]:
    """Remove FSL WM voxels from the 2,674-voxel network; write to DATA_DIR."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    roi_img    = nib.load(str(ROI_IMG))
    source_pack = np.load(SOURCE_VOXELS, allow_pickle=True)
    ijk        = np.asarray(source_pack["selected_ijk"], dtype=np.int32)   # (2674, 3)
    x, y, z   = ijk.T

    white = _resampled_prior(FSL_WHITE_PRIOR, roi_img)[x, y, z]
    keep_mask  = white < WHITE_THRESHOLD                                    # True = keep
    keep_idx   = np.flatnonzero(keep_mask).astype(np.int64)

    n_before = int(ijk.shape[0])
    n_after  = int(keep_idx.size)
    n_removed = n_before - n_after

    print(f"\n[filter] {n_before} source voxels → {n_after} kept, {n_removed} FSL WM removed",
          flush=True)

    # write filtered voxel indices
    voxel_out = DATA_DIR / "selected_voxel_indices.npz"
    payload: dict[str, np.ndarray] = {"selected_ijk": ijk[keep_idx].astype(np.int32)}
    for key in ("selected_flat_indices", "selected_active_indices", "selected_weights"):
        if key in source_pack.files:
            payload[key] = np.asarray(source_pack[key], dtype=source_pack[key].dtype)[keep_idx]
    if "weight_threshold" in source_pack.files:
        payload["weight_threshold"] = np.asarray(source_pack["weight_threshold"])
    np.savez(voxel_out, **payload)

    # write per-session beta files (filter rows)
    session_files = sorted(SOURCE_DATA_DIR.glob("selected_beta_trials_sub-*_ses-*.npy"))
    for src in session_files:
        beta = np.load(src, mmap_mode="r")             # (2674, n_trials)
        dst  = DATA_DIR / src.name
        if dst.exists():
            existing = np.load(dst, mmap_mode="r")
            if existing.shape == (n_after, beta.shape[1]):
                continue
        out = open_memmap(dst, mode="w+", dtype=beta.dtype,
                          shape=(n_after, int(beta.shape[1])))
        out[:] = np.asarray(beta[keep_idx, :], dtype=beta.dtype)
        out.flush(); del out
        print(f"  wrote {dst.name}: {n_after} × {beta.shape[1]}", flush=True)

    # write tissue QC summary
    roi_data  = roi_img.get_fdata().astype(int)
    roi_summary = json.loads(ROI_SUMMARY.read_text(encoding="utf-8"))
    names: dict[int, str] = {}
    for row in roi_summary.get("roi_rows", []):
        names[int(row["roi_id"])] = row["roi_name"]
    for idx, nm in enumerate(roi_summary.get("all_roi_names_in_order", []), start=1):
        if idx not in names:
            names[idx] = nm

    roi_labels = roi_data[x, y, z]
    rows = []
    for rid in sorted(int(v) for v in np.unique(roi_labels) if int(v) > 0):
        mask = roi_labels == rid
        wm_cnt = int(np.count_nonzero(mask & ~keep_mask))
        rows.append({
            "roi_name": names.get(rid, f"ROI_{rid}"),
            "n_voxels_before": int(mask.sum()),
            "n_wm_removed": wm_cnt,
            "n_voxels_after": int(mask.sum()) - wm_cnt,
            "pct_wm_removed": round(wm_cnt / mask.sum() * 100, 1) if mask.sum() else 0.0,
        })
    qc_df = pd.DataFrame(rows)
    qc_df.to_csv(DATA_DIR / "wm_removal_qc.csv", index=False)
    print("\n[QC] WM removal per ROI:")
    print(qc_df.to_string(index=False))

    manifest = {
        "source_voxels": str(SOURCE_VOXELS),
        "fsl_white_prior": str(FSL_WHITE_PRIOR),
        "white_threshold": WHITE_THRESHOLD,
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_removed,
    }
    (DATA_DIR / "filter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    return keep_idx, n_after


# ── step 2: per-session ROI connectivity ──────────────────────────────────

def run_connectivity() -> None:
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "connectivity_new" / "roi_edge_connectivity_requested.py"),
        "--data-dir",          str(DATA_DIR),
        "--beta-pattern",      "selected_beta_trials_sub-*_ses-*.npy",
        "--roi-img",           str(ROI_IMG),
        "--roi-summary",       str(ROI_SUMMARY),
        "--voxel-indices-path", str(DATA_DIR / "selected_voxel_indices.npz"),
        "--out-dir",           str(NETWORK_DIR),
        "--advanced-metrics-out-subdir", "advanced_metrics",
        "--advanced-metrics",  NETWORK_METRIC,
        "--exclude-rois",      "Unassigned Active Voxels",
        "--no-html",
    ])


# ── step 3: pairwise Laplacian spectral distance ───────────────────────────

def run_pairwise() -> None:
    _run([
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "main" / "analyze_pairwise_metric_separation.py"),
        "--advanced-root",      str(NETWORK_DIR / "advanced_metrics"),
        "--out-dir",            str(NETWORK_DIR),
        "--metrics",            NETWORK_METRIC,
        "--comparison-metrics", COMPARISON_METRIC,
        "--exclude-subjects",   ",".join(sorted(EXCLUDED_SUBJECTS)),
    ])


# ── step 4: summary comparison ────────────────────────────────────────────

def write_summary() -> None:
    stats_path = NETWORK_DIR / "laplacian_spectral_distance_signed_distribution_stats.csv"
    if not stats_path.exists():
        print("No stats CSV found — pairwise step may have failed.", flush=True)
        return

    df = pd.read_csv(stats_path)
    subset = df.loc[
        (df["connectivity_metric"] == NETWORK_METRIC)
        & (df["comparison_metric"] == COMPARISON_METRIC)
        & (df["cohort"] == "cross_subject_only")
    ]

    original_stats = {
        "label": "original_91601_voxel_32node",
        "off_off": 0.213,
        "on_on": 0.184,
        "delta": -0.029,
        "p": "<0.001",
        "direction": "similarity ↑",
    }

    print("\n=== COMPARISON ===")
    print(f"{'Analysis':<40} {'OFF-OFF':>8} {'ON-ON':>8} {'delta':>8} {'p':>10} {'direction'}")
    print("-" * 90)
    print(f"{original_stats['label']:<40} {original_stats['off_off']:>8.3f} "
          f"{original_stats['on_on']:>8.3f} {original_stats['delta']:>8.3f} "
          f"{original_stats['p']:>10}  {original_stats['direction']}")

    for _, row in subset.iterrows():
        off_off = float(row["mean_group_a"])
        on_on   = float(row["mean_group_b"])
        delta   = on_on - off_off
        p       = float(row["p_value_two_sided"])
        stars   = str(row["significance_stars"])
        direction = "similarity ↑" if delta < 0 else "dissimilarity ↑"
        label = f"correct_2531vox_no_unassigned"
        print(f"{label:<40} {off_off:>8.3f} {on_on:>8.3f} {delta:>8.3f} {p:>10.4f}  {stars} {direction}")

    summary = {
        "analysis": "correct_2531_voxel_no_unassigned",
        "note": "2674 network voxels minus 143 FSL WM (white>=0.5); Unassigned Active Voxels excluded",
        "stats_csv": str(stats_path),
    }
    (OUT_ROOT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70, flush=True)
    print("Correct network analysis: 2,674-voxel set, WM removed, no Unassigned")
    print("=" * 70, flush=True)

    print("\n[1] Filtering WM voxels from 2,674-voxel network ...", flush=True)
    keep_idx, n_after = build_filtered_data()

    print(f"\n[2] Running per-session ROI connectivity on {n_after} voxels ...", flush=True)
    run_connectivity()

    print("\n[3] Running pairwise Laplacian spectral distance ...", flush=True)
    run_pairwise()

    print("\n[4] Writing comparison summary ...", flush=True)
    write_summary()

    fig_path = NETWORK_DIR / "mutual_information_ksg" / \
               "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
    print(f"\nDone. Distribution figure: {fig_path}", flush=True)


if __name__ == "__main__":
    main()
