#!/usr/bin/env python3
"""Compute ROI-node connectivity and edge-correlation networks from selected beta files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting

ALWAYS_EXCLUDED_ROI_PATTERNS = (
    "ventricular csf",
    "ventrical csf",
    "lateral ventricle",
)

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each selected_beta_trials_* file, average voxel beta series within ROIs, "
            "compute ROI-to-ROI connectivity (edges), and compute edge-to-edge correlations."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/connectivity/data"),
        help="Directory containing selected_beta_trials_*.npy files.",
    )
    parser.add_argument(
        "--beta-pattern",
        default="selected_beta_trials_*.npy",
        help="Glob pattern for condition beta files.",
    )
    parser.add_argument(
        "--roi-img",
        type=Path,
        default=Path("results/connectivity/created_rois_fitted.nii.gz"),
        help="ROI label image used to define node membership.",
    )
    parser.add_argument(
        "--roi-summary",
        type=Path,
        default=Path("results/connectivity/created_roi_summary.json"),
        help="ROI summary JSON that contains all_roi_names_in_order (optional).",
    )
    parser.add_argument(
        "--voxel-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ with selected_ijk or selected_flat_indices for selected beta rows.",
    )
    parser.add_argument(
        "--voxel-weight-img",
        type=Path,
        default=None,
        help=(
            "Optional NIfTI image with one weight per voxel in the same space as the ROI image. "
            "When provided, each beta row is multiplied by the corresponding selected-voxel weight "
            "before ROI aggregation."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/connectivity/roi_edge_network"),
        help="Output directory.",
    )
    parser.add_argument(
        "--min-roi-voxels",
        type=int,
        default=5,
        help="Minimum selected voxels required to keep an ROI node.",
    )
    parser.add_argument(
        "--edge-threshold-percentile",
        type=float,
        default=85.0,
        help="Percentile threshold for connectome plotting.",
    )
    parser.add_argument(
        "--split-hemispheres",
        action="store_true",
        default=True,
        help="Split each ROI into left/right hemisphere nodes using MNI x coordinate.",
    )
    parser.add_argument(
        "--midline-band-mm",
        type=float,
        default=1.0,
        help="Absolute MNI x band treated as midline when splitting hemispheres.",
    )
    parser.add_argument(
        "--node-size",
        type=float,
        default=135.0,
        help="Static connectome node size.",
    )
    parser.add_argument(
        "--edge-linewidth",
        type=float,
        default=5.0,
        help="Static connectome edge linewidth.",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip interactive HTML connectome outputs.",
    )
    parser.add_argument(
        "--run-advanced-metrics",
        action="store_true",
        default=True,
        help="Run advanced ROI-network metrics after base connectivity outputs (default: enabled).",
    )
    parser.add_argument(
        "--skip-advanced-metrics",
        dest="run_advanced_metrics",
        action="store_false",
        help="Skip advanced ROI-network metric computation.",
    )
    parser.add_argument(
        "--advanced-metrics",
        default="all",
        help="Comma-separated advanced metric list or 'all'.",
    )
    parser.add_argument(
        "--advanced-metrics-out-subdir",
        default="advanced_metrics",
        help="Output subfolder under --out-dir for advanced metric files.",
    )
    parser.add_argument(
        "--mi-bins",
        type=int,
        default=8,
        help="Quantile bins for mutual information metric.",
    )
    parser.add_argument(
        "--mi-ksg-k",
        type=int,
        default=3,
        help="k-nearest neighbors for KSG continuous mutual information metric.",
    )
    parser.add_argument(
        "--mi-ksg-jitter",
        type=float,
        default=1e-10,
        help="Small Gaussian jitter scale to break ties for KSG MI estimation.",
    )
    parser.add_argument(
        "--granger-max-lag",
        type=int,
        default=3,
        help="Lag order for linear/nonlinear Granger metrics.",
    )
    parser.add_argument(
        "--granger-ridge",
        type=float,
        default=1e-6,
        help="Ridge regularization for Granger regression fits.",
    )
    parser.add_argument(
        "--kernel-granger-kernel",
        type=str,
        default="ip",
        choices=["ip", "gaussian"],
        help="Kernel type for nonlinear_granger metric.",
    )
    parser.add_argument(
        "--kernel-granger-degree",
        type=int,
        default=2,
        help="Inhomogeneous polynomial order for nonlinear_granger when using IP kernel.",
    )
    parser.add_argument(
        "--kernel-granger-sigma",
        type=float,
        default=0.0,
        help="Gaussian kernel width for nonlinear_granger (<=0 uses median-distance heuristic).",
    )
    parser.add_argument(
        "--kernel-granger-eig-frac",
        type=float,
        default=1e-6,
        help="Relative eigenvalue threshold used to retain kernel components.",
    )
    parser.add_argument(
        "--kernel-granger-alpha",
        type=float,
        default=0.05,
        help="FDR significance level for selecting kernel Granger components.",
    )
    parser.add_argument("--wavelet-min-scale", type=int, default=2, help="Minimum wavelet scale.")
    parser.add_argument("--wavelet-max-scale", type=int, default=20, help="Maximum wavelet scale.")
    parser.add_argument(
        "--wavelet-omega0",
        type=float,
        default=6.0,
        help="Morlet omega0 for wavelet coherence.",
    )
    parser.add_argument(
        "--wavelet-smooth-scale-sigma",
        type=float,
        default=1.0,
        help="Scale-axis smoothing sigma for wavelet coherence.",
    )
    parser.add_argument(
        "--wavelet-smooth-time-sigma",
        type=float,
        default=2.0,
        help="Time-axis smoothing sigma for wavelet coherence.",
    )
    parser.add_argument(
        "--wavelet-fmin-hz",
        type=float,
        default=0.01,
        help="Lower frequency bound for wavelet coherence; <=0 disables the lower bound.",
    )
    parser.add_argument(
        "--wavelet-fmax-hz",
        type=float,
        default=0.1,
        help="Upper frequency bound for wavelet coherence; <=0 disables the upper bound.",
    )
    parser.add_argument(
        "--wavelet-mask-coi",
        action="store_true",
        default=True,
        help="Mask the cone of influence for wavelet coherence (default: enabled).",
    )
    parser.add_argument(
        "--wavelet-no-mask-coi",
        dest="wavelet_mask_coi",
        action="store_false",
        help="Disable cone-of-influence masking for wavelet coherence.",
    )
    parser.add_argument(
        "--wavelet-coi-factor",
        type=float,
        default=float(np.sqrt(2.0)),
        help="Cone-of-influence half-width factor used when COI masking is enabled.",
    )
    parser.add_argument(
        "--respect-temporal-boundaries",
        action="store_true",
        help=(
            "Forwarded to advanced metric runner: compute temporal metrics per manifest run-segment "
            "and aggregate across runs. Disabled by default to preserve legacy outputs."
        ),
    )
    parser.add_argument(
        "--temporal-manifest-path",
        type=Path,
        default=Path("results/connectivity/tmp/concat_manifest_group.tsv"),
        help="Manifest TSV path used for temporal segmentation when enabled.",
    )
    parser.add_argument(
        "--temporal-condition-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_beta_trials_gvs_column_indices.npz"),
        help="Condition column-index NPZ used for temporal segmentation when enabled.",
    )
    parser.add_argument(
        "--temporal-metrics",
        default="linear_granger,nonlinear_granger",
        help=(
            "Comma-separated metrics to segment temporally when --respect-temporal-boundaries is enabled. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--temporal-min-trials",
        type=int,
        default=8,
        help="Minimum trials required per run-segment for temporal metric estimation.",
    )
    parser.add_argument(
        "--exclude-rois",
        nargs="+",
        default=[],
        metavar="NAME",
        help=(
            "Extra ROI names (or substrings) to exclude from node building (case-insensitive). "
            "Ventricular CSF is always excluded."
        ),
    )
    return parser.parse_args()


def _condition_sort_key(label: str) -> Tuple[int, int | str]:
    if label.lower() == "sham":
        return (0, 0)
    match = re.match(r"^gvs(\d+)$", label, flags=re.IGNORECASE)
    if match:
        return (1, int(match.group(1)))
    return (2, label.lower())


def _discover_beta_files(data_dir: Path, pattern: str) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    for path in sorted(data_dir.glob(pattern)):
        stem = path.stem
        if not stem.startswith("selected_beta_trials_"):
            continue
        label = stem[len("selected_beta_trials_") :]
        if not label:
            continue
        out.append((label, path))
    out.sort(key=lambda x: _condition_sort_key(x[0]))
    return out


def _load_selected_ijk(path: Path, volume_shape: Tuple[int, int, int]) -> np.ndarray:
    pack = np.load(path, allow_pickle=True)
    if "selected_ijk" in pack.files:
        ijk = np.asarray(pack["selected_ijk"], dtype=np.int32)
    elif "selected_flat_indices" in pack.files:
        flat = np.asarray(pack["selected_flat_indices"], dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, volume_shape)).astype(np.int32, copy=False)
    else:
        raise KeyError("selected_voxel_indices.npz must contain selected_ijk or selected_flat_indices.")

    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected selected_ijk shape (N, 3); got {ijk.shape}")

    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < volume_shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < volume_shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < volume_shape[2])
    )
    if not np.all(valid):
        dropped = int(np.count_nonzero(~valid))
        print(f"Warning: dropping {dropped} out-of-bounds selected voxels from metadata.", flush=True)
        ijk = ijk[valid]
    return ijk


def _load_selected_voxel_weights(
    weight_img_path: Path | None,
    selected_ijk: np.ndarray,
    volume_shape: Tuple[int, int, int],
    reference_affine: np.ndarray,
) -> np.ndarray | None:
    if weight_img_path is None:
        return None

    weight_img = nib.load(str(weight_img_path))
    weight_data = np.asarray(weight_img.get_fdata(), dtype=np.float64)
    if weight_data.shape != volume_shape:
        raise ValueError(
            f"Voxel-weight image shape {weight_data.shape} does not match expected volume shape {volume_shape}: "
            f"{weight_img_path}"
        )
    if not np.allclose(weight_img.affine, reference_affine):
        print(
            f"Warning: voxel-weight image affine differs from ROI image affine: {weight_img_path}",
            flush=True,
        )

    x, y, z = selected_ijk.T
    weights = weight_data[x, y, z].astype(np.float64, copy=False)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return weights


def _load_roi_names(summary_path: Path) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if not summary_path.exists():
        return names
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = payload.get("roi_rows")
    if isinstance(rows, list) and rows:
        for row in rows:
            roi_id = row.get("roi_id")
            roi_name = row.get("roi_name")
            if isinstance(roi_id, int) and isinstance(roi_name, str) and roi_name:
                names[int(roi_id)] = roi_name
    all_names = payload.get("all_roi_names_in_order")
    if isinstance(all_names, list):
        for idx, name in enumerate(all_names, start=1):
            if idx not in names and isinstance(name, str) and name:
                names[idx] = name
    return names
def _safe_corrcoef_rows(series: np.ndarray, min_overlap: int = 3) -> np.ndarray:
    x = np.asarray(series, dtype=np.float64)
    n_rows = x.shape[0]
    out = np.full((n_rows, n_rows), np.nan, dtype=np.float64)
    for i in range(n_rows):
        out[i, i] = 1.0
    for i in range(n_rows):
        xi = x[i]
        for j in range(i + 1, n_rows):
            xj = x[j]
            valid = np.isfinite(xi) & np.isfinite(xj)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < min_overlap:
                continue
            xi0 = xi[valid]
            xj0 = xj[valid]
            xi0 = xi0 - np.mean(xi0)
            xj0 = xj0 - np.mean(xj0)
            denom = np.sqrt(np.dot(xi0, xi0) * np.dot(xj0, xj0))
            if denom <= 1e-12:
                val = np.nan
            else:
                val = float(np.dot(xi0, xj0) / denom)
                val = float(np.clip(val, -1.0, 1.0))
            out[i, j] = val
            out[j, i] = val
    return out
def _zscore_rows_nan(series: np.ndarray) -> np.ndarray:
    x = np.asarray(series, dtype=np.float64)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    sd = np.where(sd <= 1e-12, np.nan, sd)
    return (x - mu) / sd
def _write_matrix_csv(path: Path, matrix: np.ndarray, row_labels: List[str], col_labels: List[str]) -> None:
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_csv(path, index=True)
def _plot_connectome(conn: np.ndarray, node_coords_mm: np.ndarray, node_labels: List[str], out_png: Path,
                     out_html: Path | None, percentile: float, node_size: float, edge_linewidth: float, title: str):
    finite_edges = np.abs(conn[np.triu_indices(conn.shape[0], k=1)])
    finite_edges = finite_edges[np.isfinite(finite_edges)]
    if finite_edges.size == 0:
        edge_threshold = "99%"
    else:
        percentile = float(np.clip(percentile, 0.0, 100.0))
        edge_threshold = f"{percentile:.1f}%"

    plotting.plot_connectome(np.nan_to_num(conn, nan=0.0), node_coords_mm, node_color="#1f78b4", node_size=float(node_size),
                             edge_cmap="jet", edge_vmin=-1.0, edge_vmax=1.0, edge_threshold=edge_threshold,
                             edge_kwargs={"linewidth": float(edge_linewidth), "alpha": 0.85}, node_kwargs={"alpha": 0.95}, colorbar=True, 
                             title=title, output_file=str(out_png))
    
    if out_html is not None:
        view = plotting.view_connectome(np.nan_to_num(conn, nan=0.0), node_coords_mm, edge_threshold=edge_threshold, edge_cmap="jet",
                                        symmetric_cmap=False, linewidth=max(1.5, float(edge_linewidth) * 1.8), colorbar=True,
                                        node_size=max(3.0, float(node_size) / 16.0), title=title)
        view.save_as_html(str(out_html))

    labels_path = out_png.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(node_labels), encoding="utf-8")
def _plot_heatmap(matrix: np.ndarray, labels: List[str], out_png: Path, title: str, cmap: str = "jet", vmin: float = -1.0, vmax: float = 1.0,
                  cbar_label: str = "Correlation") -> None:
    values = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    n = values.shape[0]
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(values, cmap=cmap, vmin=float(vmin), vmax=float(vmax))
    ax.set_title(title)
    ax.set_xlabel("Edges")
    ax.set_ylabel("Edges")
    if n <= 32:
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
    else:
        n_ticks = min(24, n)
        tick_idx = np.linspace(0, n - 1, num=n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_yticks(tick_idx)
        ax.set_xticklabels([labels[i] for i in tick_idx], rotation=90, fontsize=6)
        ax.set_yticklabels([labels[i] for i in tick_idx], fontsize=6)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
def _offdiag_abs_vmax(matrix: np.ndarray, percentile: float = 99.0, fallback: float = 0.10) -> float:
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return float(fallback)
    iu = np.triu_indices(m.shape[0], k=1)
    vals = np.abs(m[iu])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(fallback)
    vmax = float(np.percentile(vals, float(np.clip(percentile, 50.0, 100.0))))
    return float(max(vmax, 1e-6))


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    roi_img_path = args.roi_img.expanduser().resolve()
    roi_summary_path = args.roi_summary.expanduser().resolve()
    voxel_indices_path = args.voxel_indices_path.expanduser().resolve()
    voxel_weight_img_path = (
        args.voxel_weight_img.expanduser().resolve() if args.voxel_weight_img is not None else None
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    beta_files = _discover_beta_files(data_dir, args.beta_pattern)
    roi_img = nib.load(str(roi_img_path))
    roi_data = roi_img.get_fdata().astype(np.int32)

    selected_ijk = _load_selected_ijk(voxel_indices_path, roi_data.shape)
    selected_voxel_weights = _load_selected_voxel_weights(
        voxel_weight_img_path,
        selected_ijk,
        roi_data.shape,
        roi_img.affine,
    )
    n_selected = selected_ijk.shape[0]
    x, y, z = selected_ijk.T
    roi_labels_at_selected = roi_data[x, y, z]

    roi_names_lookup = _load_roi_names(roi_summary_path)
    unique_ids, counts = np.unique(roi_labels_at_selected[roi_labels_at_selected > 0], return_counts=True)
    unique_ids = unique_ids.astype(np.int32, copy=False)
    counts = counts.astype(np.int64, copy=False)
    keep = counts >= int(max(1, args.min_roi_voxels))
    kept_roi_ids = unique_ids[keep]
    selected_coords_mm = nib.affines.apply_affine(roi_img.affine, selected_ijk)
    selected_x = selected_coords_mm[:, 0]
    midline_band = float(max(0.0, args.midline_band_mm))
    min_node_vox = int(max(1, args.min_roi_voxels))

    roi_member_indices: List[np.ndarray] = []
    roi_labels: List[str] = []
    node_coords_mm: List[np.ndarray] = []
    node_base_roi_ids: List[int] = []
    node_hemisphere: List[str] = []

    exclude_lower = sorted(
        {
            str(e).strip().lower()
            for e in [*ALWAYS_EXCLUDED_ROI_PATTERNS, *list(args.exclude_rois)]
            if str(e).strip()
        }
    )

    for roi_id in kept_roi_ids.tolist():
        base_members = np.flatnonzero(roi_labels_at_selected == roi_id).astype(np.int64, copy=False)
        roi_name = roi_names_lookup.get(int(roi_id), f"ROI_{int(roi_id)}")

        if exclude_lower and any(ex in roi_name.lower() for ex in exclude_lower):
            continue

        if args.split_hemispheres:
            roi_x = selected_x[base_members]
            left = base_members[roi_x < -midline_band]
            right = base_members[roi_x > midline_band]
            mid = base_members[np.abs(roi_x) <= midline_band]

            if mid.size > 0:
                if left.size >= right.size and left.size > 0:
                    left = np.concatenate([left, mid])
                elif right.size > 0:
                    right = np.concatenate([right, mid])

            added = 0
            if left.size >= min_node_vox:
                roi_member_indices.append(left)
                roi_labels.append(f"L {roi_name}")
                node_coords_mm.append(np.mean(selected_coords_mm[left], axis=0))
                node_base_roi_ids.append(int(roi_id))
                node_hemisphere.append("L")
                added += 1
            if right.size >= min_node_vox:
                roi_member_indices.append(right)
                roi_labels.append(f"R {roi_name}")
                node_coords_mm.append(np.mean(selected_coords_mm[right], axis=0))
                node_base_roi_ids.append(int(roi_id))
                node_hemisphere.append("R")
                added += 1
            if added == 0 and base_members.size >= min_node_vox:
                roi_member_indices.append(base_members)
                roi_labels.append(f"M {roi_name}")
                node_coords_mm.append(np.mean(selected_coords_mm[base_members], axis=0))
                node_base_roi_ids.append(int(roi_id))
                node_hemisphere.append("M")
        else:
            if base_members.size < min_node_vox:
                continue
            roi_member_indices.append(base_members)
            roi_labels.append(roi_name)
            node_coords_mm.append(np.mean(selected_coords_mm[base_members], axis=0))
            node_base_roi_ids.append(int(roi_id))
            node_hemisphere.append("B")

    node_coords_mm_arr = np.vstack(node_coords_mm).astype(np.float64, copy=False)
    node_counts = np.asarray([int(m.size) for m in roi_member_indices], dtype=np.int64)

    roi_meta = pd.DataFrame({"node_id": np.arange(1, len(roi_member_indices) + 1, dtype=int),
            "base_roi_id": np.asarray(node_base_roi_ids, dtype=int),
            "hemisphere": node_hemisphere,
            "node_name": roi_labels,
            "n_selected_voxels": node_counts.astype(int),
            "x_mm": node_coords_mm_arr[:, 0],
            "y_mm": node_coords_mm_arr[:, 1],
            "z_mm": node_coords_mm_arr[:, 2]})
    roi_meta.to_csv(out_dir / "roi_nodes.csv", index=False)

    n_rois = int(len(roi_member_indices))
    iu = np.triu_indices(n_rois, k=1)
    edge_pairs = list(zip(iu[0].tolist(), iu[1].tolist()))
    edge_labels = [f"{roi_labels[i]}__{roi_labels[j]}" for i, j in edge_pairs]

    connectivity_vec_rows: List[np.ndarray] = []
    connectivity_vec_labels: List[str] = []
    edge_corr_by_label: Dict[str, np.ndarray] = {}
    per_file_summary = []

    for label, beta_path in beta_files:
        beta = np.load(beta_path, mmap_mode="r")
        if beta.ndim != 2:
            print(f"Skipping {beta_path.name}: expected 2D, got {beta.shape}", flush=True)
            continue
        if beta.shape[0] != n_selected:
            print(
                f"Skipping {beta_path.name}: voxel count {beta.shape[0]} does not match selected_ijk {n_selected}.",
                flush=True,
            )
            continue

        beta_data = np.asarray(beta, dtype=np.float64)
        if selected_voxel_weights is not None:
            beta_data = beta_data * selected_voxel_weights[:, None]

        roi_ts = np.full((n_rois, beta.shape[1]), np.nan, dtype=np.float64)
        for idx, members in enumerate(roi_member_indices):
            roi_ts[idx] = np.nanmean(beta_data[members, :], axis=0)

        node_conn = _safe_corrcoef_rows(roi_ts)
        node_conn = np.clip(node_conn, -1.0, 1.0, out=node_conn)
        np.fill_diagonal(node_conn, 1.0)

        z_roi_ts = _zscore_rows_nan(roi_ts)
        edge_ts = z_roi_ts[iu[0], :] * z_roi_ts[iu[1], :]
        edge_corr = _safe_corrcoef_rows(edge_ts)
        edge_corr = np.clip(edge_corr, -1.0, 1.0, out=edge_corr)
        np.fill_diagonal(edge_corr, 1.0)

        conn_vec = node_conn[iu]
        connectivity_vec_rows.append(conn_vec)
        connectivity_vec_labels.append(label)

        cond_dir = out_dir / label
        cond_dir.mkdir(parents=True, exist_ok=True)

        np.save(cond_dir / f"roi_timeseries_{label}.npy", roi_ts.astype(np.float32, copy=False))
        np.save(cond_dir / f"roi_connectivity_corr_{label}.npy", node_conn.astype(np.float32, copy=False))
        np.save(cond_dir / f"edge_timeseries_{label}.npy", edge_ts.astype(np.float32, copy=False))
        np.save(cond_dir / f"edge_correlation_{label}.npy", edge_corr.astype(np.float32, copy=False))

        _write_matrix_csv(cond_dir / f"roi_connectivity_corr_{label}.csv", node_conn, row_labels=roi_labels, col_labels=roi_labels)
        _write_matrix_csv(cond_dir / f"edge_correlation_{label}.csv", edge_corr, row_labels=edge_labels, col_labels=edge_labels)

        edge_df = pd.DataFrame({
                "edge_id": np.arange(len(edge_pairs), dtype=int),
                "roi_i_idx": [i for i, _ in edge_pairs],
                "roi_j_idx": [j for _, j in edge_pairs],
                "roi_i_name": [roi_labels[i] for i, _ in edge_pairs],
                "roi_j_name": [roi_labels[j] for _, j in edge_pairs],
                "roi_pair_corr": conn_vec.astype(np.float64)})
        edge_df.to_csv(cond_dir / f"roi_edges_{label}.csv", index=False)

        connectome_png = cond_dir / f"connectome_{label}.png"
        connectome_html = None if args.no_html else cond_dir / f"connectome_{label}.html"
        _plot_connectome(conn=node_conn, node_coords_mm=node_coords_mm_arr, node_labels=roi_labels, out_png=connectome_png,
                         out_html=connectome_html, percentile=float(args.edge_threshold_percentile), node_size=float(args.node_size),
                         edge_linewidth=float(args.edge_linewidth), title=f"ROI connectome ({label})")

        _plot_heatmap(edge_corr, labels=edge_labels, out_png=cond_dir / f"edge_correlation_{label}.png", title=f"Edge correlation matrix ({label})")
        edge_corr_by_label[label] = edge_corr.copy()

        per_file_summary.append({"label": label, "beta_file": str(beta_path), "n_trials": int(beta.shape[1]), "n_rois": n_rois, "n_edges": int(len(edge_pairs)),
             "roi_ts_finite_fraction": float(np.mean(np.isfinite(roi_ts))), "conn_finite_fraction": float(np.mean(np.isfinite(node_conn))),
             "edge_corr_finite_fraction": float(np.mean(np.isfinite(edge_corr))),
             "voxel_weighting_applied": bool(selected_voxel_weights is not None)})
        print(f"Processed {label}: trials={beta.shape[1]}, rois={n_rois}, edges={len(edge_pairs)}", flush=True)

    edge_strength = np.vstack(connectivity_vec_rows).astype(np.float64, copy=False)  # files x edges
    np.save(out_dir / "edge_strength_by_file.npy", edge_strength.astype(np.float32, copy=False))
    _write_matrix_csv(out_dir / "edge_strength_by_file.csv", edge_strength, row_labels=connectivity_vec_labels, col_labels=edge_labels)

    file_similarity = _safe_corrcoef_rows(edge_strength)
    np.save(out_dir / "file_similarity_from_edges.npy", file_similarity.astype(np.float32, copy=False))
    _write_matrix_csv(out_dir / "file_similarity_from_edges.csv", file_similarity, row_labels=connectivity_vec_labels, col_labels=connectivity_vec_labels)

    edge_strength_corr = _safe_corrcoef_rows(edge_strength.T)
    np.save(out_dir / "edge_strength_correlation_across_files.npy", edge_strength_corr.astype(np.float32, copy=False))
    _write_matrix_csv(out_dir / "edge_strength_correlation_across_files.csv", edge_strength_corr, row_labels=edge_labels, col_labels=edge_labels)
    _plot_heatmap(edge_strength_corr, edge_labels, out_png=out_dir / "edge_strength_correlation_across_files.png",
                  title="Edge-strength correlation across selected_beta_trials_* files")

    if "sham" in edge_corr_by_label:
        sham_edge_corr = edge_corr_by_label["sham"]
        diff_rows = []
        for label in connectivity_vec_labels:
            if label == "sham" or label not in edge_corr_by_label:
                continue
            delta = edge_corr_by_label[label] - sham_edge_corr
            cond_dir = out_dir / label
            np.save(cond_dir / f"edge_correlation_delta_vs_sham_{label}.npy", delta.astype(np.float32, copy=False))
            _write_matrix_csv(
                cond_dir / f"edge_correlation_delta_vs_sham_{label}.csv",
                delta,
                row_labels=edge_labels,
                col_labels=edge_labels,
            )
            delta_vmax = _offdiag_abs_vmax(delta, percentile=99.0, fallback=0.10)
            _plot_heatmap(
                delta,
                labels=edge_labels,
                out_png=cond_dir / f"edge_correlation_delta_vs_sham_{label}.png",
                title=f"Edge correlation delta vs sham ({label} - sham)",
                cmap="jet",
                vmin=-delta_vmax,
                vmax=delta_vmax,
                cbar_label="Delta correlation",
            )
            diff_rows.append(
                {
                    "label": label,
                    "fro_norm_delta_vs_sham": float(np.linalg.norm(np.nan_to_num(delta), ord="fro")),
                    "mean_abs_delta_vs_sham": float(np.nanmean(np.abs(delta))),
                    "max_abs_delta_vs_sham": float(np.nanmax(np.abs(delta))),
                }
            )
        if diff_rows:
            pd.DataFrame(diff_rows).to_csv(out_dir / "edge_correlation_delta_vs_sham_summary.csv", index=False)

    summary = {
        "data_dir": str(data_dir),
        "roi_img": str(roi_img_path),
        "voxel_indices_path": str(voxel_indices_path),
        "voxel_weight_img": str(voxel_weight_img_path) if voxel_weight_img_path is not None else None,
        "voxel_weighting_applied": bool(selected_voxel_weights is not None),
        "voxel_weight_summary": (
            {
                "min": float(np.min(selected_voxel_weights)),
                "max": float(np.max(selected_voxel_weights)),
                "mean": float(np.mean(selected_voxel_weights)),
                "nonzero": int(np.count_nonzero(selected_voxel_weights)),
            }
            if selected_voxel_weights is not None
            else None
        ),
        "beta_pattern": args.beta_pattern,
        "min_roi_voxels": int(args.min_roi_voxels),
        "split_hemispheres": bool(args.split_hemispheres),
        "midline_band_mm": float(args.midline_band_mm),
        "node_size": float(args.node_size),
        "edge_linewidth": float(args.edge_linewidth),
        "edge_threshold_percentile": float(args.edge_threshold_percentile),
        "excluded_roi_patterns": exclude_lower,
        "n_files_processed": int(len(connectivity_vec_labels)),
        "file_labels": connectivity_vec_labels,
        "n_rois": n_rois,
        "n_edges": int(len(edge_pairs)),
        "has_sham_reference": bool("sham" in edge_corr_by_label),
        "roi_labels": roi_labels,
        "edge_labels": edge_labels,
        "per_file_summary": per_file_summary,
    }

    advanced_info = None
    if args.run_advanced_metrics:
        from group_analysis.main.roi_metric_runner import run_metric_pipeline

        advanced_info = run_metric_pipeline(
            network_dir=out_dir,
            out_subdir=args.advanced_metrics_out_subdir,
            metrics=args.advanced_metrics,
            labels=connectivity_vec_labels,
            args=args,
        )
        summary["advanced_metrics"] = {
            "enabled": True,
            "selected_metrics": advanced_info["selected_metrics"],
            "selected_labels": advanced_info["selected_labels"],
            "n_outputs": int(advanced_info["n_outputs"]),
            "summary_csv": str(advanced_info["summary_csv"]),
            "out_root": str(advanced_info["out_root"]),
        }
    else:
        summary["advanced_metrics"] = {"enabled": False}

    (out_dir / "roi_edge_network_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Saved outputs to: {out_dir}", flush=True)
    if advanced_info is not None:
        print(f"Saved advanced metrics to: {advanced_info['out_root']}", flush=True)


if __name__ == "__main__":
    main()
