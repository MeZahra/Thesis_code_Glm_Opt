#!/usr/bin/env python3
"""Rebuild the voxel-level connectivity analysis on the correct 2,674-voxel network.

Corrections relative to the original roi_edge_network:
  1. Uses results/connectivity/data/selected_voxel_indices.npz (2,674 voxels)
     instead of results/connectivity/tmp/data/ (91,601 voxels).
  2. Removes FSL WM voxels (white prior >= 0.5) upstream. ROI labels are used
     only for QC; voxel-level MI uses every retained selected voxel.
  3. Multiplies each selected voxel beta series by the signed voxel-weight map
     before connectivity estimation.
  4. Computes mutual_information_ksg between selected voxels rather than between
     ROI-averaged nodes.

All outputs go to claude_result/ and nothing in the existing analysis is touched.
"""

from __future__ import annotations

import json
import re
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
from numba import njit, prange
from scipy.linalg import eigvalsh
from scipy.special import digamma

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── inputs (read-only) ─────────────────────────────────────────────────────
SOURCE_DATA_DIR   = REPO_ROOT / "results" / "connectivity" / "data"
SOURCE_VOXELS     = SOURCE_DATA_DIR / "selected_voxel_indices.npz"
ROI_IMG           = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_rois_fitted.nii.gz"
ROI_SUMMARY       = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_roi_summary.json"
VOXEL_WEIGHT_IMG  = (
    REPO_ROOT
    / "results"
    / "ablation"
    / "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_original.nii.gz"
)
FSL_WHITE_PRIOR   = Path("/usr/local/fsl/data/standard/tissuepriors/avg152T1_white.img")
WHITE_THRESHOLD   = 0.5
EXCLUDED_SUBJECTS = {"sub-pd017"}   # same as original analysis

# ── outputs ────────────────────────────────────────────────────────────────
OUT_ROOT      = Path(__file__).resolve().parent
DATA_DIR      = OUT_ROOT / "data"
NETWORK_DIR   = OUT_ROOT / "roi_edge_network"
NETWORK_METRIC      = "mutual_information_ksg"
COMPARISON_METRIC   = "laplacian_spectral_distance_signed"
VOXEL_ADVANCED_DIR   = NETWORK_DIR / "advanced_metrics_voxel"
VOXEL_PAIRWISE_CSV   = NETWORK_DIR / "voxel_pairwise_metric_values.csv"
MI_KSG_K             = 3
MI_KSG_JITTER        = 1e-10
MI_KSG_MAX_SUPPORTED_K = 16
LABEL_RE = re.compile(r"^(sub-[^_]+)_ses-(\d+)$")
SESSION_TO_STATE = {1: "off", 2: "on"}


# ── helpers ────────────────────────────────────────────────────────────────

def _run(cmd: list[str]) -> None:
    print("\n[cmd]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


@njit(parallel=True, fastmath=True)
def _ksg_mi_matrix_numba(x: np.ndarray, k: int, digamma_table: np.ndarray) -> np.ndarray:
    n_nodes = x.shape[0]
    n_samples = x.shape[1]
    out = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    base = digamma_table[k] + digamma_table[n_samples]

    for i in prange(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            total = 0.0
            for sample_idx in range(n_samples):
                best = np.empty(MI_KSG_MAX_SUPPORTED_K, dtype=np.float64)
                for kk in range(MI_KSG_MAX_SUPPORTED_K):
                    best[kk] = 1e308
                max_idx = 0

                for other_idx in range(n_samples):
                    if other_idx == sample_idx:
                        continue
                    dx = abs(x[i, sample_idx] - x[i, other_idx])
                    dy = abs(x[j, sample_idx] - x[j, other_idx])
                    dist = dx if dx >= dy else dy
                    if dist < best[max_idx]:
                        best[max_idx] = dist
                        max_idx = 0
                        max_val = best[0]
                        for kk in range(1, k):
                            if best[kk] > max_val:
                                max_val = best[kk]
                                max_idx = kk

                eps = np.nextafter(best[max_idx], 0.0)
                nx = 0
                ny = 0
                for other_idx in range(n_samples):
                    if other_idx == sample_idx:
                        continue
                    if abs(x[i, sample_idx] - x[i, other_idx]) < eps:
                        nx += 1
                    if abs(x[j, sample_idx] - x[j, other_idx]) < eps:
                        ny += 1

                total += digamma_table[nx + 1] + digamma_table[ny + 1]

            mi = base - total / n_samples
            if not np.isfinite(mi):
                mi = 0.0
            out[i, j] = mi
            out[j, i] = mi

    return out


def _impute_nan_by_row_median(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).copy()
    row_median = np.nanmedian(x, axis=1, keepdims=True)
    row_median = np.where(np.isfinite(row_median), row_median, 0.0)
    bad = ~np.isfinite(x)
    if np.any(bad):
        rows, _cols = np.where(bad)
        x[bad] = row_median[rows, 0]
    return x


def _compute_voxel_ksg_mi_matrix(voxel_ts: np.ndarray, k: int, jitter: float) -> np.ndarray:
    x = _impute_nan_by_row_median(voxel_ts)
    if x.ndim != 2:
        raise ValueError(f"Expected voxel time series shape (n_voxels, n_trials), got {x.shape}")
    n_nodes, n_samples = x.shape
    if n_nodes < 2:
        raise ValueError("At least two voxels are required for voxel-level mutual information.")
    if n_samples <= 3:
        raise ValueError("At least four trials are required for KSG mutual information.")

    k_eff = int(np.clip(int(k), 1, n_samples - 1))
    if k_eff > MI_KSG_MAX_SUPPORTED_K:
        raise ValueError(
            f"KSG k={k_eff} exceeds this script's supported maximum "
            f"({MI_KSG_MAX_SUPPORTED_K})."
        )

    jitter = float(max(0.0, jitter))
    if jitter > 0.0:
        rng = np.random.default_rng(0)
        row_scale = np.nanstd(x, axis=1)
        row_scale = np.where(np.isfinite(row_scale) & (row_scale > 1e-12), row_scale, 1.0)
        x = x + rng.normal(0.0, jitter * row_scale[:, None], size=x.shape)

    digamma_table = digamma(np.arange(0, n_samples + 2, dtype=np.float64))
    mat = _ksg_mi_matrix_numba(x, k_eff, digamma_table)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 0.0)
    return mat


def _signed_normalized_laplacian_spectrum(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float64)
    adjacency = np.nan_to_num(adjacency, nan=0.0, posinf=0.0, neginf=0.0)
    adjacency = 0.5 * (adjacency + adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    degrees = np.sum(np.abs(adjacency), axis=1)
    inv_sqrt = np.zeros_like(degrees)
    valid = degrees > 1e-12
    inv_sqrt[valid] = 1.0 / np.sqrt(degrees[valid])
    laplacian = np.eye(adjacency.shape[0], dtype=np.float64)
    laplacian -= adjacency * inv_sqrt[:, None] * inv_sqrt[None, :]
    laplacian[~np.isfinite(laplacian)] = 0.0
    eigvals = eigvalsh(laplacian, check_finite=False, overwrite_a=True)
    return np.sort(np.real(eigvals))


def _resampled_prior(path: Path, ref_img: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine):
        img = image.resample_to_img(img, ref_img, interpolation="continuous",
                                    force_resample=True, copy_header=True)
    return np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))


def _selected_weights_from_img(path: Path, selected_ijk: np.ndarray, ref_img: nib.Nifti1Image) -> np.ndarray:
    weight_img = nib.load(str(path))
    if weight_img.shape[:3] != ref_img.shape[:3]:
        raise ValueError(
            f"Voxel-weight image shape {weight_img.shape[:3]} does not match ROI image shape "
            f"{ref_img.shape[:3]}: {path}"
        )
    if not np.allclose(weight_img.affine, ref_img.affine):
        print(f"Warning: voxel-weight image affine differs from ROI image affine: {path}", flush=True)

    weight_data = np.asarray(weight_img.get_fdata(), dtype=np.float32)
    x, y, z = selected_ijk.T
    return np.nan_to_num(weight_data[x, y, z], nan=0.0, posinf=0.0, neginf=0.0)


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
    for key in ("selected_flat_indices", "selected_active_indices"):
        if key in source_pack.files:
            payload[key] = np.asarray(source_pack[key], dtype=source_pack[key].dtype)[keep_idx]
    payload["selected_weights"] = _selected_weights_from_img(VOXEL_WEIGHT_IMG, ijk, roi_img)[keep_idx]
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
        "voxel_weight_img": str(VOXEL_WEIGHT_IMG),
        "voxel_weighting": (
            "selected beta rows are multiplied by signed voxel weights before voxel-level MI"
        ),
        "white_threshold": WHITE_THRESHOLD,
        "n_before": n_before,
        "n_after": n_after,
        "n_removed": n_removed,
    }
    (DATA_DIR / "filter_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    return keep_idx, n_after


# ── step 2: per-session voxel mutual information ──────────────────────────

def _session_label_from_beta_path(path: Path) -> str:
    stem = path.stem
    prefix = "selected_beta_trials_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected beta filename: {path.name}")
    return stem[len(prefix):]


def _parse_subject_session(label: str) -> tuple[str, int]:
    match = LABEL_RE.match(label)
    if match is None:
        raise ValueError(f"Could not parse subject/session label: {label}")
    return match.group(1), int(match.group(2))


def _voxel_labels(selected_ijk: np.ndarray) -> list[str]:
    return [
        f"Voxel_{idx + 1:04d}_ijk_{int(i)}_{int(j)}_{int(k)}"
        for idx, (i, j, k) in enumerate(selected_ijk)
    ]


def _finite_vmin_vmax(matrix: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(matrix, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -1e-3, 1e-3
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if not np.isfinite(vmin):
        vmin = -1e-3
    if not np.isfinite(vmax):
        vmax = 1e-3
    if vmax <= vmin:
        delta = max(1e-6, 0.05 * abs(vmax))
        vmin -= delta
        vmax += delta
    return vmin, vmax


def run_voxel_mutual_information() -> list[dict]:
    NETWORK_DIR.mkdir(parents=True, exist_ok=True)
    VOXEL_ADVANCED_DIR.mkdir(parents=True, exist_ok=True)

    voxel_pack = np.load(DATA_DIR / "selected_voxel_indices.npz", allow_pickle=True)
    selected_ijk = np.asarray(voxel_pack["selected_ijk"], dtype=np.int32)
    selected_weights = np.asarray(voxel_pack["selected_weights"], dtype=np.float64)
    labels = _voxel_labels(selected_ijk)
    labels_text = "\n".join(labels)

    beta_files = sorted(DATA_DIR.glob("selected_beta_trials_sub-*_ses-*.npy"))
    if not beta_files:
        raise FileNotFoundError(f"No filtered beta files found in {DATA_DIR}")

    rows: list[dict] = []
    for beta_path in beta_files:
        label = _session_label_from_beta_path(beta_path)
        subject, session = _parse_subject_session(label)
        state = SESSION_TO_STATE.get(session)
        if state is None:
            print(f"Skipping {label}: session {session} is not mapped to medication state.", flush=True)
            continue

        metric_dir = VOXEL_ADVANCED_DIR / label / NETWORK_METRIC
        metric_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = metric_dir / f"{NETWORK_METRIC}.npy"
        spectrum_path = metric_dir / f"{NETWORK_METRIC}_laplacian_spectrum_signed.npy"
        meta_path = metric_dir / f"{NETWORK_METRIC}_meta.json"
        labels_path = metric_dir / f"{NETWORK_METRIC}_connectome.labels.txt"
        labels_path.write_text(labels_text, encoding="utf-8")

        beta = np.load(beta_path, mmap_mode="r")
        if beta.ndim != 2:
            raise ValueError(f"Expected 2D beta file, got {beta.shape}: {beta_path}")
        if beta.shape[0] != selected_ijk.shape[0]:
            raise ValueError(
                f"Voxel count mismatch for {beta_path.name}: beta rows={beta.shape[0]}, "
                f"selected voxels={selected_ijk.shape[0]}"
            )

        if matrix_path.exists() and spectrum_path.exists() and meta_path.exists():
            matrix = np.load(matrix_path, mmap_mode="r")
            spectrum = np.load(spectrum_path)
            if matrix.shape == (selected_ijk.shape[0], selected_ijk.shape[0]) and spectrum.shape[0] == selected_ijk.shape[0]:
                print(f"Reusing voxel MI for {label}: {matrix_path}", flush=True)
                rows.append(
                    {
                        "label": label,
                        "subject": subject,
                        "session": session,
                        "state": state,
                        "matrix_path": str(matrix_path),
                        "spectrum_path": str(spectrum_path),
                        "n_nodes": int(selected_ijk.shape[0]),
                        "n_trials": int(beta.shape[1]),
                    }
                )
                continue

        beta_data = np.asarray(beta, dtype=np.float64)
        beta_data = beta_data * selected_weights[:, None]
        print(
            f"Computing voxel-level {NETWORK_METRIC} for {label}: "
            f"{beta_data.shape[0]} voxels × {beta_data.shape[1]} trials",
            flush=True,
        )
        matrix = _compute_voxel_ksg_mi_matrix(beta_data, k=MI_KSG_K, jitter=MI_KSG_JITTER)
        np.save(matrix_path, matrix.astype(np.float32, copy=False))

        spectrum = _signed_normalized_laplacian_spectrum(matrix)
        np.save(spectrum_path, spectrum.astype(np.float64, copy=False))

        vmin, vmax = _finite_vmin_vmax(matrix)
        meta = {
            "label": label,
            "metric": NETWORK_METRIC,
            "description": (
                "Pairwise continuous mutual information using the "
                "Kraskov-Stogbauer-Grassberger kNN estimator between selected voxels."
            ),
            "node_level": "voxel",
            "directed": False,
            "n_nodes": int(matrix.shape[0]),
            "n_trials": int(beta.shape[1]),
            "vmin": vmin,
            "vmax": vmax,
            "kwargs": {"k": MI_KSG_K, "jitter": MI_KSG_JITTER},
            "voxel_weighting_applied": True,
            "voxel_weight_img": str(VOXEL_WEIGHT_IMG),
            "spectrum_path": str(spectrum_path),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        rows.append(
            {
                "label": label,
                "subject": subject,
                "session": session,
                "state": state,
                "matrix_path": str(matrix_path),
                "spectrum_path": str(spectrum_path),
                "n_nodes": int(matrix.shape[0]),
                "n_trials": int(beta.shape[1]),
            }
        )
        print(f"Saved voxel MI and Laplacian spectrum for {label}", flush=True)

    if not rows:
        raise RuntimeError("No voxel-level mutual-information matrices were produced.")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(VOXEL_ADVANCED_DIR / "metric_run_summary.csv", index=False)
    manifest = {
        "metric": NETWORK_METRIC,
        "node_level": "voxel",
        "advanced_root": str(VOXEL_ADVANCED_DIR),
        "n_sessions": int(len(rows)),
        "n_voxels": int(selected_ijk.shape[0]),
        "voxel_indices_path": str(DATA_DIR / "selected_voxel_indices.npz"),
        "voxel_weight_img": str(VOXEL_WEIGHT_IMG),
        "voxel_weighting_applied": True,
        "mi_ksg_k": MI_KSG_K,
        "mi_ksg_jitter": MI_KSG_JITTER,
    }
    (VOXEL_ADVANCED_DIR / "voxel_metric_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return rows


# ── step 3: pairwise Laplacian spectral distance ───────────────────────────

def _build_voxel_pairwise_csv(session_rows: list[dict]) -> Path:
    rows = []
    for idx_a, sess_a in enumerate(session_rows):
        spectrum_a = np.load(sess_a["spectrum_path"])
        for idx_b in range(idx_a + 1, len(session_rows)):
            sess_b = session_rows[idx_b]
            spectrum_b = np.load(sess_b["spectrum_path"])
            raw_score = float(
                np.linalg.norm(
                    np.asarray(spectrum_a, dtype=np.float64)
                    - np.asarray(spectrum_b, dtype=np.float64)
                )
            )
            state_a = str(sess_a["state"])
            state_b = str(sess_b["state"])
            pair_label = f"{state_a}-{state_b}" if state_a <= state_b else f"{state_b}-{state_a}"
            rows.append(
                {
                    "connectivity_metric": NETWORK_METRIC,
                    "label_a": sess_a["label"],
                    "label_b": sess_b["label"],
                    "subject_a": sess_a["subject"],
                    "subject_b": sess_b["subject"],
                    "session_a": int(sess_a["session"]),
                    "session_b": int(sess_b["session"]),
                    "state_a": state_a,
                    "state_b": state_b,
                    "pair_class": "within_condition" if state_a == state_b else "between_condition",
                    "pair_label": pair_label,
                    "same_subject": bool(sess_a["subject"] == sess_b["subject"]),
                    "comparison_metric": COMPARISON_METRIC,
                    "comparison_kind": "graph_distance",
                    "higher_is_more_similar": False,
                    "raw_score": raw_score,
                    "oriented_score": -raw_score,
                }
            )

    pairwise_df = pd.DataFrame(rows)
    pairwise_df.to_csv(VOXEL_PAIRWISE_CSV, index=False)
    return VOXEL_PAIRWISE_CSV


def run_pairwise(session_rows: list[dict]) -> None:
    pairwise_csv = _build_voxel_pairwise_csv(session_rows)
    _run([
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "main" / "analyze_pairwise_metric_separation.py"),
        "--pairwise-csv",       str(pairwise_csv),
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
        label = "correct_2531voxel_nodes_weighted"
        print(f"{label:<40} {off_off:>8.3f} {on_on:>8.3f} {delta:>8.3f} {p:>10.4f}  {stars} {direction}")

    summary = {
        "analysis": "correct_2531_voxel_nodes_weighted",
        "note": (
            "2674 network voxels minus 143 FSL WM (white>=0.5); signed voxel weights "
            "multiplied into beta rows before voxel-level mutual_information_ksg; "
            "Laplacian spectral distances are computed from voxel-node MI matrices"
        ),
        "stats_csv": str(stats_path),
        "advanced_metrics_root": str(VOXEL_ADVANCED_DIR),
        "pairwise_csv": str(VOXEL_PAIRWISE_CSV),
    }
    (OUT_ROOT / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70, flush=True)
    print("Correct network analysis: 2,674-voxel set, voxel MI, signed weights, WM removed")
    print("=" * 70, flush=True)

    print("\n[1] Filtering WM voxels from 2,674-voxel network ...", flush=True)
    _keep_idx, n_after = build_filtered_data()

    print(f"\n[2] Running per-session voxel mutual information on {n_after} voxels ...", flush=True)
    session_rows = run_voxel_mutual_information()

    print("\n[3] Running pairwise Laplacian spectral distance ...", flush=True)
    run_pairwise(session_rows)

    print("\n[4] Writing comparison summary ...", flush=True)
    write_summary()

    fig_path = NETWORK_DIR / "mutual_information_ksg" / \
               "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
    print(f"\nDone. Distribution figure: {fig_path}", flush=True)


if __name__ == "__main__":
    main()
