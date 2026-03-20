#!/usr/bin/env python3
"""Compare observed subject-session graph structure against random non-selected ROI controls."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = next(path for path in [HERE, *HERE.parents] if (path / "group_analysis").is_dir())
GROUP_ANALYSIS_DIR = REPO_ROOT / "group_analysis"
GROUP_ANALYSIS_MAIN_DIR = GROUP_ANALYSIS_DIR / "main"
if str(GROUP_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(GROUP_ANALYSIS_DIR))
if str(GROUP_ANALYSIS_MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(GROUP_ANALYSIS_MAIN_DIR))

from roi_metrics import METRIC_REGISTRY, normalize_metric_list  # noqa: E402
from roi_metric_runner import _metric_kwargs  # noqa: E402
from analyze_pairwise_metric_separation import (  # noqa: E402
    _laplacian_spectral_distance,
    _signed_normalized_laplacian_spectrum,
)


LABEL_RE = re.compile(r"^(sub-[^_]+)_ses-(\d+)$")
SESSION_TO_STATE = {1: "off", 2: "on"}
PAIR_LABEL_ORDER = ("off-off", "on-on", "off-on")
NULL_METRIC_EXCLUDE = {"linear_granger", "nonlinear_granger"}
EXCLUDED_NODE_PATTERNS = ("brain-stem", "cerebral white matter")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate random non-selected ROI control networks, compute graph distances for each "
            "connectivity metric, and compare the observed selected-network contrasts against the "
            "empirical null distribution from random controls."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "results/connectivity/data/concat_manifest_group.tsv"
        ),
        help="Subject/session manifest TSV with cleaned_beta and trial_keep paths.",
    )
    parser.add_argument(
        "--selected-ts-root",
        type=Path,
        default=Path(
            "results/connectivity/tmp/tmp-roi_edge_network_from_data/all"
        ),
        help=(
            "Root containing the saved selected ROI average time series per subject/session. "
            "This must match --selected-voxel-indices."
        ),
    )
    parser.add_argument(
        "--selected-voxel-indices",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ with the objective-selected voxel coordinates.",
    )
    parser.add_argument(
        "--roi-img",
        type=Path,
        default=Path(
            "results/connectivity/atlas figure/created_rois_fitted.nii.gz"
        ),
        help="ROI atlas image.",
    )
    parser.add_argument(
        "--brain-img",
        type=Path,
        default=Path(
            "results/connectivity/tmp/data/MNI152_T1_2mm_brain.nii.gz"
        ),
        help="Brain image used to restrict atlas voxels to the analysis volume.",
    )
    parser.add_argument(
        "--reference-node-csv",
        type=Path,
        default=None,
        help=(
            "ROI-node CSV used to preserve node names and geometry. "
            "If omitted, defaults to <selected-ts-root>/../roi_nodes.csv."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/connectivity/roi_edge_network/advanced_metrics/random_graph_distance_null"
        ),
        help="Output directory for the matched random-control null analysis.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated connectivity metrics to analyze or 'all'. Granger metrics are excluded by default.",
    )
    parser.add_argument(
        "--exclude-subjects",
        default="sub-pd017",
        help="Comma-separated subject IDs to exclude.",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=100,
        help="How many random control draws to generate.",
    )
    parser.add_argument(
        "--n-control-voxels",
        type=int,
        default=0,
        help=(
            "Number of non-selected voxels to draw per random control. "
            "If <= 0, uses the selected-network voxel count from selected_voxel_indices.npz."
        ),
    )
    parser.add_argument(
        "--min-node-voxels",
        type=int,
        default=1,
        help=(
            "Minimum number of usable non-selected voxels required in every included subject/session "
            "to keep a node in the random-control graph."
        ),
    )
    parser.add_argument(
        "--min-finite-trial-fraction",
        type=float,
        default=0.0,
        help=(
            "Per-subject/session minimum fraction of finite beta values required for a voxel "
            "to be eligible for the random-control pool."
        ),
    )
    parser.add_argument(
        "--midline-band-mm",
        type=float,
        default=1.0,
        help="Absolute MNI x band treated as midline when splitting left/right control nodes.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for voxel sampling.",
    )
    parser.add_argument(
        "--graph-distance",
        choices=["frobenius", "laplacian_spectral_distance_signed"],
        default="frobenius",
        help="Whole-graph distance used to compare subject/session matrices.",
    )

    parser.add_argument("--mi-bins", type=int, default=8)
    parser.add_argument("--mi-ksg-k", type=int, default=3)
    parser.add_argument("--mi-ksg-jitter", type=float, default=1e-10)
    parser.add_argument("--granger-max-lag", type=int, default=3)
    parser.add_argument("--granger-ridge", type=float, default=1e-6)
    parser.add_argument(
        "--kernel-granger-kernel",
        type=str,
        default="ip",
        choices=["ip", "gaussian"],
    )
    parser.add_argument("--kernel-granger-degree", type=int, default=2)
    parser.add_argument("--kernel-granger-sigma", type=float, default=0.0)
    parser.add_argument("--kernel-granger-eig-frac", type=float, default=1e-6)
    parser.add_argument("--kernel-granger-alpha", type=float, default=0.05)
    parser.add_argument("--wavelet-min-scale", type=int, default=2)
    parser.add_argument("--wavelet-max-scale", type=int, default=20)
    parser.add_argument("--wavelet-omega0", type=float, default=6.0)
    parser.add_argument("--wavelet-smooth-scale-sigma", type=float, default=1.0)
    parser.add_argument("--wavelet-smooth-time-sigma", type=float, default=2.0)
    parser.add_argument("--wavelet-fmin-hz", type=float, default=0.01)
    parser.add_argument("--wavelet-fmax-hz", type=float, default=0.1)
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
    )
    return parser.parse_args()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _normalize_subject_exclusions(raw_value: str) -> set[str]:
    return {item.strip() for item in str(raw_value).split(",") if item.strip()}


def _parse_label(label: str) -> tuple[str, int]:
    match = LABEL_RE.match(str(label))
    if match is None:
        raise ValueError(f"Unrecognized subject/session label: {label}")
    return str(match.group(1)), int(match.group(2))


def _is_excluded_node_name(node_name: str) -> bool:
    lower = str(node_name).lower()
    return any(pattern in lower for pattern in EXCLUDED_NODE_PATTERNS)


def _load_selected_flat_indices(path: Path, volume_shape: tuple[int, int, int]) -> np.ndarray:
    pack = np.load(path, allow_pickle=True)
    if "selected_flat_indices" in pack.files:
        flat = np.asarray(pack["selected_flat_indices"], dtype=np.int64).ravel()
    elif "selected_ijk" in pack.files:
        ijk = np.asarray(pack["selected_ijk"], dtype=np.int64)
        flat = np.ravel_multi_index(ijk.T, volume_shape)
    else:
        raise KeyError("Selected voxel NPZ must contain selected_flat_indices or selected_ijk.")
    if flat.size > 1 and np.any(flat[1:] < flat[:-1]):
        flat = np.unique(flat)
    return flat.astype(np.int64, copy=False)


def _load_manifest(manifest_path: Path, excluded_subjects: set[str]) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, Any]]]:
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    required = {
        "sub_tag",
        "ses",
        "run",
        "n_trials",
        "cleaned_beta",
        "trial_keep_path",
        "offset_start",
        "offset_end",
    }
    missing = required.difference(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    manifest_df = manifest_df.loc[~manifest_df["sub_tag"].astype(str).isin(excluded_subjects)].copy()
    manifest_df["ses"] = pd.to_numeric(manifest_df["ses"], errors="raise").astype(int)
    manifest_df["run"] = pd.to_numeric(manifest_df["run"], errors="raise").astype(int)
    manifest_df["n_trials"] = pd.to_numeric(manifest_df["n_trials"], errors="raise").astype(int)
    manifest_df["offset_start"] = pd.to_numeric(manifest_df["offset_start"], errors="raise").astype(int)
    manifest_df["offset_end"] = pd.to_numeric(manifest_df["offset_end"], errors="raise").astype(int)
    manifest_df["label"] = [
        f"{sub_tag}_ses-{ses}"
        for sub_tag, ses in zip(manifest_df["sub_tag"].astype(str), manifest_df["ses"].astype(int), strict=False)
    ]
    manifest_df = manifest_df.sort_values(["sub_tag", "ses", "offset_start", "run"]).reset_index(drop=True)
    labels = (
        manifest_df.loc[:, ["sub_tag", "ses", "label"]]
        .drop_duplicates()
        .sort_values(["sub_tag", "ses"])
        ["label"]
        .astype(str)
        .tolist()
    )
    label_meta = {
        label: {
            "subject": subject,
            "session": session,
            "state": SESSION_TO_STATE[session],
        }
        for label, subject, session in (
            (label, *_parse_label(label))
            for label in labels
        )
    }
    return manifest_df, labels, label_meta


def _load_trial_keep(path: Path, expected_len: int, cache: dict[Path, np.ndarray]) -> np.ndarray:
    cached = cache.get(path)
    if cached is not None:
        if cached.shape[0] != expected_len:
            raise ValueError(f"Cached trial_keep length mismatch for {path}: {cached.shape[0]} vs {expected_len}")
        return cached
    keep = np.asarray(np.load(path), dtype=bool).ravel()
    if keep.shape[0] != expected_len:
        raise ValueError(f"trial_keep length mismatch for {path}: {keep.shape[0]} vs {expected_len}")
    cache[path] = keep
    return keep


def _load_session_voxel_data(
    session_rows: pd.DataFrame,
    voxel_ijk: np.ndarray,
    trial_keep_cache: dict[Path, np.ndarray],
) -> np.ndarray:
    x, y, z = voxel_ijk.T
    blocks: list[np.ndarray] = []
    for row in session_rows.sort_values(["offset_start", "run"]).itertuples(index=False):
        beta = np.load(Path(row.cleaned_beta), mmap_mode="r")
        keep = _load_trial_keep(Path(row.trial_keep_path), beta.shape[-1], trial_keep_cache)
        block = np.asarray(beta[x, y, z, :], dtype=np.float32)
        block = block[:, keep]
        if block.shape[1] != int(row.n_trials):
            raise ValueError(
                f"Kept-trial mismatch for {row.cleaned_beta}: got {block.shape[1]}, expected {int(row.n_trials)}"
            )
        blocks.append(block)
    if not blocks:
        raise RuntimeError("Session had no run blocks in the manifest.")
    return np.concatenate(blocks, axis=1)


def _finite_fraction_mask(finite_fraction: np.ndarray, min_fraction: float) -> np.ndarray:
    frac = np.asarray(finite_fraction, dtype=np.float64)
    threshold = float(max(0.0, min(1.0, min_fraction)))
    if threshold <= 0.0:
        return frac > 0.0
    return frac >= threshold


def _nanmean_rows(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(data)
    counts = np.sum(valid, axis=0)
    summed = np.sum(np.where(valid, data, 0.0), axis=0, dtype=np.float64)
    out = np.full(data.shape[1], np.nan, dtype=np.float32)
    keep = counts > 0
    out[keep] = (summed[keep] / counts[keep]).astype(np.float32, copy=False)
    return out


def _build_control_candidate_pool(
    roi_img_path: Path,
    brain_img_path: Path,
    selected_voxel_indices_path: Path,
    reference_node_csv: Path,
    midline_band_mm: float,
) -> tuple[int, np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    roi_img = nib.load(str(roi_img_path))
    roi_data = np.asarray(roi_img.get_fdata(), dtype=np.int32)
    brain_data = np.asarray(nib.load(str(brain_img_path)).get_fdata(), dtype=np.float32)
    if brain_data.shape != roi_data.shape:
        raise ValueError(
            f"Brain image shape {brain_data.shape} does not match ROI image shape {roi_data.shape}"
        )

    selected_flat = _load_selected_flat_indices(selected_voxel_indices_path, roi_data.shape)
    selected_count = int(selected_flat.size)
    brain_mask = np.isfinite(brain_data.ravel()) & (brain_data.ravel() != 0)
    brain_flat = np.flatnonzero(brain_mask)
    candidate_flat = np.setdiff1d(brain_flat, selected_flat, assume_unique=False).astype(np.int64, copy=False)
    if candidate_flat.size == 0:
        raise RuntimeError("No non-selected brain voxels remain after removing selected_flat indices.")
    candidate_ijk = np.column_stack(np.unravel_index(candidate_flat, roi_data.shape)).astype(np.int32, copy=False)
    candidate_coords_mm = nib.affines.apply_affine(roi_img.affine, candidate_ijk)
    candidate_x_mm = candidate_coords_mm[:, 0]
    candidate_roi_ids = roi_data.ravel()[candidate_flat].astype(np.int32, copy=False)

    reference_nodes = pd.read_csv(reference_node_csv).sort_values("node_id").reset_index(drop=True)
    required_cols = {"node_id", "base_roi_id", "hemisphere", "node_name", "x_mm", "y_mm", "z_mm"}
    missing = required_cols.difference(reference_nodes.columns)
    if missing:
        raise ValueError(f"Reference node CSV is missing required columns: {sorted(missing)}")

    node_union_indices: dict[str, np.ndarray] = {}
    inventory_rows: list[dict[str, Any]] = []
    for base_roi_id, roi_df in reference_nodes.groupby("base_roi_id", sort=False):
        base_idx = np.flatnonzero(candidate_roi_ids == int(base_roi_id)).astype(np.int64, copy=False)
        roi_x = candidate_x_mm[base_idx]
        left = base_idx[roi_x < -float(midline_band_mm)]
        right = base_idx[roi_x > float(midline_band_mm)]
        mid = base_idx[np.abs(roi_x) <= float(midline_band_mm)]
        if mid.size:
            if left.size == 0 and right.size == 0:
                left = np.concatenate([left, mid])
            elif left.size >= right.size:
                left = np.concatenate([left, mid])
            else:
                right = np.concatenate([right, mid])

        hemi_members = {
            "L": left,
            "R": right,
            "M": base_idx,
            "B": base_idx,
        }

        for row in roi_df.itertuples(index=False):
            members = np.asarray(hemi_members.get(str(row.hemisphere), np.asarray([], dtype=np.int64)), dtype=np.int64)
            node_union_indices[str(row.node_name)] = members
            inventory_rows.append(
                {
                    "node_id": int(row.node_id),
                    "base_roi_id": int(row.base_roi_id),
                    "hemisphere": str(row.hemisphere),
                    "node_name": str(row.node_name),
                    "x_mm": float(row.x_mm),
                    "y_mm": float(row.y_mm),
                    "z_mm": float(row.z_mm),
                    "global_candidate_voxels": int(members.size),
                }
            )
    inventory_df = pd.DataFrame(inventory_rows).sort_values("node_id").reset_index(drop=True)
    return selected_count, candidate_flat, candidate_ijk, inventory_df, node_union_indices


def _collect_control_counts(
    manifest_df: pd.DataFrame,
    labels: list[str],
    candidate_ijk: np.ndarray,
    node_union_indices: dict[str, np.ndarray],
    min_finite_trial_fraction: float,
) -> pd.DataFrame:
    trial_keep_cache: dict[Path, np.ndarray] = {}
    count_rows: list[dict[str, Any]] = []
    for label in labels:
        session_rows = manifest_df.loc[manifest_df["label"] == label].copy()
        session_data = _load_session_voxel_data(session_rows, candidate_ijk, trial_keep_cache)
        finite_fraction = np.mean(np.isfinite(session_data), axis=1)
        row = {
            "label": label,
            "subject": str(session_rows["sub_tag"].iloc[0]),
            "session": int(session_rows["ses"].iloc[0]),
            "n_trials": int(session_data.shape[1]),
        }
        for node_name, node_idx in node_union_indices.items():
            members = np.asarray(node_idx, dtype=np.int64)
            if members.size == 0:
                row[node_name] = 0
            else:
                keep = _finite_fraction_mask(finite_fraction[members], min_fraction=min_finite_trial_fraction)
                row[node_name] = int(np.count_nonzero(keep))
        count_rows.append(row)
    return pd.DataFrame(count_rows).sort_values(["subject", "session"]).reset_index(drop=True)


def _choose_keep_nodes(
    node_inventory_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    min_node_voxels: int,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    stat_rows: list[dict[str, Any]] = []
    for node_name in node_inventory_df["node_name"].astype(str).tolist():
        values = counts_df[node_name].to_numpy(dtype=np.int64)
        stat_rows.append(
            {
                "node_name": node_name,
                "min_available_voxels": int(np.min(values)) if values.size else 0,
                "median_available_voxels": float(np.median(values)) if values.size else 0.0,
                "max_available_voxels": int(np.max(values)) if values.size else 0,
            }
        )
    stat_df = pd.DataFrame(stat_rows)
    merged = node_inventory_df.merge(stat_df, on="node_name", how="left")
    merged["sampled_voxels_per_draw"] = merged["min_available_voxels"].astype(int)
    keep_df = merged.loc[
        (~merged["node_name"].astype(str).map(_is_excluded_node_name))
        & (merged["min_available_voxels"].astype(int) >= int(max(1, min_node_voxels)))
    ].copy()
    keep_df = keep_df.sort_values("node_id").reset_index(drop=True)
    keep_nodes = keep_df["node_name"].astype(str).tolist()
    sample_sizes = {
        str(row.node_name): int(row.sampled_voxels_per_draw)
        for row in keep_df.itertuples(index=False)
    }
    return merged, keep_nodes, sample_sizes


def _load_available_control_node_data(
    manifest_df: pd.DataFrame,
    labels: list[str],
    candidate_ijk: np.ndarray,
    node_union_indices: dict[str, np.ndarray],
    keep_nodes: list[str],
    min_finite_trial_fraction: float,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, Any]]]:
    trial_keep_cache: dict[Path, np.ndarray] = {}
    data_by_label: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict[str, Any]] = []
    for label in labels:
        session_rows = manifest_df.loc[manifest_df["label"] == label].copy()
        label_store: dict[str, np.ndarray] = {}
        n_trials = None
        for node_name in keep_nodes:
            node_ijk = candidate_ijk[np.asarray(node_union_indices[node_name], dtype=np.int64)]
            node_data = _load_session_voxel_data(session_rows, node_ijk, trial_keep_cache)
            finite_fraction = np.mean(np.isfinite(node_data), axis=1)
            keep = _finite_fraction_mask(finite_fraction, min_fraction=min_finite_trial_fraction)
            available_data = node_data[keep].astype(np.float32, copy=False)
            label_store[node_name] = available_data
            n_trials = int(available_data.shape[1]) if n_trials is None else n_trials
            summary_rows.append(
                {
                    "label": label,
                    "node_name": node_name,
                    "available_voxels": int(available_data.shape[0]),
                    "n_trials": int(available_data.shape[1]),
                }
            )
        data_by_label[label] = label_store
    return data_by_label, summary_rows


def _sample_control_draws(
    control_data_by_label: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    keep_nodes: list[str],
    n_control_voxels: int,
    n_draws: int,
    random_seed: int,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(int(random_seed))
    draws: list[dict[str, np.ndarray]] = []
    if int(n_control_voxels) <= 0:
        raise ValueError(f"n_control_voxels must be > 0, got {n_control_voxels}")
    if not keep_nodes:
        raise RuntimeError("No keep nodes were selected; cannot sample control draws.")
    if int(n_control_voxels) < len(keep_nodes):
        raise ValueError(
            f"n_control_voxels ({n_control_voxels}) must be at least the number of kept nodes ({len(keep_nodes)})."
        )

    for draw_idx in range(int(n_draws)):
        draw_map: dict[str, np.ndarray] = {}
        for label in labels:
            n_nodes = len(keep_nodes)
            first_node_data = control_data_by_label[label][keep_nodes[0]]
            if first_node_data.ndim != 2:
                raise ValueError(
                    f"Expected 2D node data for label {label}, got {first_node_data.ndim} dims."
                )
            n_trials = int(first_node_data.shape[1])
            if n_trials <= 0:
                raise RuntimeError(f"No trials available for label {label}.")

            node_rows = np.zeros(0, dtype=np.int32)
            node_ranges: list[tuple[int, int]] = []
            control_pool: list[np.ndarray] = []
            for node_idx, node_name in enumerate(keep_nodes):
                available = control_data_by_label[label][node_name]
                if available.ndim != 2:
                    raise ValueError(
                        f"Expected 2D node data for label {label}, node {node_name}, got {available.ndim} dims."
                    )
                if available.shape[1] != n_trials:
                    raise ValueError(
                        f"Trial count mismatch for label {label}, node {node_name}: "
                        f"{available.shape[1]} vs {n_trials}"
                    )
                if available.shape[0] <= 0:
                    raise RuntimeError(
                        f"No usable control voxels for label {label}, node {node_name}; cannot enforce one voxel per node."
                    )
                control_pool.append(available.astype(np.float32, copy=False))
                start = int(node_rows.shape[0])
                end = start + int(available.shape[0])
                node_ranges.append((start, end))
                node_rows = np.concatenate(
                    (node_rows, np.full(available.shape[0], node_idx, dtype=np.int32)),
                    axis=0,
                )

            if len(control_pool) == 0:
                raise RuntimeError(f"No available control voxels for label {label}.")
            control_data = np.concatenate(control_pool, axis=0)
            if control_data.shape[0] != node_rows.shape[0]:
                raise RuntimeError(f"Control pool size mismatch for label {label}.")
            if control_data.shape[0] == int(n_control_voxels):
                chosen = np.arange(control_data.shape[0], dtype=np.int64)
            elif control_data.shape[0] < int(n_control_voxels):
                mandatory: list[np.int64] = []
                for start, end in node_ranges:
                    mandatory.append(rng.choice(np.arange(start, end, dtype=np.int64), size=1)[0])
                remaining_to_draw = int(n_control_voxels - len(mandatory))
                if remaining_to_draw > 0:
                    extras = rng.choice(
                        np.arange(control_data.shape[0], dtype=np.int64),
                        size=remaining_to_draw,
                        replace=True,
                    )
                    chosen = np.concatenate((np.array(mandatory, dtype=np.int64), extras), axis=0)
                else:
                    chosen = np.array(mandatory, dtype=np.int64)
            else:
                mandatory: list[np.int64] = []
                for start, end in node_ranges:
                    mandatory.append(rng.choice(np.arange(start, end, dtype=np.int64), size=1)[0])
                remaining = np.ones(control_data.shape[0], dtype=bool)
                remaining[np.array(mandatory, dtype=np.int64)] = False
                remaining_indices = np.flatnonzero(remaining)
                remaining_to_draw = int(n_control_voxels - n_nodes)
                if remaining_to_draw > 0:
                    extras = rng.choice(
                        remaining_indices,
                        size=remaining_to_draw,
                        replace=False,
                    )
                    chosen = np.concatenate((np.array(mandatory, dtype=np.int64), extras), axis=0)
                else:
                    chosen = np.array(mandatory, dtype=np.int64)

            picked_data = control_data[chosen]
            picked_node = node_rows[chosen]

            roi_ts = np.full((n_nodes, n_trials), np.nan, dtype=np.float32)
            for node_idx in range(n_nodes):
                node_mask = picked_node == node_idx
                if np.any(node_mask):
                    roi_ts[node_idx] = _nanmean_rows(picked_data[node_mask])
            draw_map[label] = roi_ts
        draws.append(draw_map)
    return draws


def _prepare_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"Expected square matrix, got {out.shape}")
    np.fill_diagonal(out, 0.0)
    return out


def _frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = _prepare_matrix(a) - _prepare_matrix(b)
    scale = max(diff.shape[0] * diff.shape[1], 1)
    return float(np.linalg.norm(diff, ord="fro") / np.sqrt(scale))


def _graph_distance(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    if kind == "frobenius":
        return _frobenius_distance(a, b)
    if kind == "laplacian_spectral_distance_signed":
        return float(
            _laplacian_spectral_distance(
                _signed_normalized_laplacian_spectrum(_prepare_matrix(a)),
                _signed_normalized_laplacian_spectrum(_prepare_matrix(b)),
            )
        )
    raise ValueError(f"Unsupported graph distance: {kind}")


def _metric_output_dir_name(metric_name: str) -> str:
    return str(metric_name)


def _load_selected_roi_timeseries(
    selected_ts_root: Path,
    keep_nodes: list[str],
    label_meta: dict[str, dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[str]]:
    node_csv_parent = selected_ts_root.parent / "roi_nodes.csv"
    node_csv_local = selected_ts_root / "roi_nodes.csv"
    node_csv = node_csv_parent if node_csv_parent.exists() else node_csv_local
    if not node_csv.exists():
        raise FileNotFoundError(
            "Selected ROI node CSV not found. Checked: "
            f"{node_csv_parent} and {node_csv_local}"
        )
    all_nodes = pd.read_csv(node_csv)["node_name"].astype(str).tolist()
    filtered_nodes = [node for node in all_nodes if not _is_excluded_node_name(node)]
    missing_nodes = [node for node in keep_nodes if node not in filtered_nodes]
    if missing_nodes:
        raise RuntimeError(f"Selected ROI time series are missing required nodes: {missing_nodes}")
    keep_idx = np.asarray([filtered_nodes.index(node) for node in keep_nodes], dtype=np.int64)

    selected_ts: dict[str, np.ndarray] = {}
    for label in sorted(label_meta.keys()):
        path = selected_ts_root / label / f"roi_timeseries_{label}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing selected ROI time series: {path}")
        roi_ts = np.asarray(np.load(path), dtype=np.float32)
        if roi_ts.ndim != 2 or roi_ts.shape[0] != len(filtered_nodes):
            raise ValueError(f"Unexpected selected ROI time series shape for {path}: {roi_ts.shape}")
        selected_ts[label] = roi_ts[keep_idx].astype(np.float32, copy=False)
    return selected_ts, list(keep_nodes)


def _selected_kept_voxel_count(
    selected_ts_root: Path,
    keep_nodes: list[str],
) -> int | None:
    node_csv_parent = selected_ts_root.parent / "roi_nodes.csv"
    node_csv_local = selected_ts_root / "roi_nodes.csv"
    node_csv = node_csv_parent if node_csv_parent.exists() else node_csv_local
    if not node_csv.exists():
        return None

    node_df = pd.read_csv(node_csv)
    if "node_name" not in node_df.columns or "n_selected_voxels" not in node_df.columns:
        return None

    keep = set(str(node) for node in keep_nodes)
    matched = node_df.loc[node_df["node_name"].astype(str).isin(keep), "n_selected_voxels"]
    if matched.empty:
        return None
    return int(np.sum(pd.to_numeric(matched, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)))


def _compute_control_metric_matrices(
    draws: list[dict[str, np.ndarray]],
    metric_name: str,
    args: argparse.Namespace,
) -> list[dict[str, np.ndarray]]:
    metric_fn = METRIC_REGISTRY[metric_name]
    kwargs = _metric_kwargs(metric_name, args)
    out: list[dict[str, np.ndarray]] = []
    for draw_map in draws:
        draw_mats: dict[str, np.ndarray] = {}
        for label, roi_ts in draw_map.items():
            result = metric_fn(roi_ts, **kwargs)
            draw_mats[label] = np.asarray(result["matrix"], dtype=np.float64)
        out.append(draw_mats)
    return out


def _compute_selected_metric_matrices(
    selected_ts_by_label: dict[str, np.ndarray],
    metric_name: str,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    metric_fn = METRIC_REGISTRY[metric_name]
    kwargs = _metric_kwargs(metric_name, args)
    out: dict[str, np.ndarray] = {}
    for label, roi_ts in selected_ts_by_label.items():
        result = metric_fn(roi_ts, **kwargs)
        out[label] = np.asarray(result["matrix"], dtype=np.float64)
    return out


def _pairwise_distance_matrix(
    matrices_by_label: dict[str, np.ndarray],
    label_meta: dict[str, dict[str, Any]],
    graph_distance: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    labels = sorted(matrices_by_label.keys())
    display_labels = [f"{label_meta[label]['subject']}-ses{label_meta[label]['session']}" for label in labels]
    pairwise_matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    spectra_by_label: dict[str, np.ndarray] | None = None
    if graph_distance == "laplacian_spectral_distance_signed":
        spectra_by_label = {
            label: _signed_normalized_laplacian_spectrum(_prepare_matrix(matrices_by_label[label]))
            for label in labels
        }
    rows: list[dict[str, Any]] = []
    for idx_a, label_a in enumerate(labels):
        meta_a = label_meta[label_a]
        for idx_b in range(idx_a + 1, len(labels)):
            label_b = labels[idx_b]
            meta_b = label_meta[label_b]
            if spectra_by_label is None:
                value = _graph_distance(matrices_by_label[label_a], matrices_by_label[label_b], kind=graph_distance)
            else:
                value = float(
                    _laplacian_spectral_distance(
                        spectra_by_label[label_a],
                        spectra_by_label[label_b],
                    )
                )
            pairwise_matrix[idx_a, idx_b] = value
            pairwise_matrix[idx_b, idx_a] = value
            pair_label = (
                f"{meta_a['state']}-{meta_b['state']}"
                if meta_a["state"] <= meta_b["state"]
                else f"{meta_b['state']}-{meta_a['state']}"
            )
            rows.append(
                {
                    "label_a": label_a,
                    "label_b": label_b,
                    "subject_a": meta_a["subject"],
                    "subject_b": meta_b["subject"],
                    "session_a": int(meta_a["session"]),
                    "session_b": int(meta_b["session"]),
                    "state_a": meta_a["state"],
                    "state_b": meta_b["state"],
                    "same_subject": bool(meta_a["subject"] == meta_b["subject"]),
                    "pair_label": pair_label,
                    "distance": float(value),
                }
            )
    return pd.DataFrame(rows), pairwise_matrix, display_labels


def _summarize_pair_contrasts(pairwise_df: pd.DataFrame) -> dict[str, float]:
    cross_df = pairwise_df.loc[~pairwise_df["same_subject"]].copy()
    mu_oo = float(cross_df.loc[cross_df["pair_label"] == "off-off", "distance"].mean())
    mu_nn = float(cross_df.loc[cross_df["pair_label"] == "on-on", "distance"].mean())
    mu_on = float(cross_df.loc[cross_df["pair_label"] == "off-on", "distance"].mean())
    delta_sep = float(mu_on - 0.5 * (mu_oo + mu_nn))
    delta_within = float(mu_nn - mu_oo)
    return {
        "n_cross_subject_pairs": int(cross_df.shape[0]),
        "n_off_off": int((cross_df["pair_label"] == "off-off").sum()),
        "n_on_on": int((cross_df["pair_label"] == "on-on").sum()),
        "n_off_on": int((cross_df["pair_label"] == "off-on").sum()),
        "mu_off_off": mu_oo,
        "mu_on_on": mu_nn,
        "mu_off_on": mu_on,
        "delta_sep": delta_sep,
        "delta_within": delta_within,
    }


def _empirical_right_tail(null_values: np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    return float((1 + np.count_nonzero(values >= float(observed))) / (values.size + 1))


def _empirical_two_sided(null_values: np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    center = float(np.mean(values))
    observed_abs = abs(float(observed) - center)
    null_abs = np.abs(values - center)
    return float((1 + np.count_nonzero(null_abs >= observed_abs)) / (values.size + 1))


def _z_score_against_null(null_values: np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    sd = float(np.std(values, ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return float((float(observed) - float(np.mean(values))) / sd)


def _percentile_against_null(null_values: np.ndarray, observed: float) -> float:
    values = np.asarray(null_values, dtype=np.float64)
    return float(100.0 * (np.count_nonzero(values <= float(observed)) / max(values.size, 1)))


def _save_square_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(path)


def _plot_heatmap(matrix: np.ndarray, labels: list[str], out_png: Path, title: str, cbar_label: str) -> None:
    values = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    finite = values[np.isfinite(values)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.imshow(values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    ticks = np.arange(len(labels), dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.84, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_pair_means(observed_stats: dict[str, float], out_png: Path, title: str) -> None:
    labels = ["OFF-OFF", "ON-ON", "OFF-ON"]
    values = [
        float(observed_stats["mu_off_off"]),
        float(observed_stats["mu_on_on"]),
        float(observed_stats["mu_off_on"]),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(np.arange(3), values, marker="o", linewidth=2.0, color="#1f77b4")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean graph distance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_null_distribution(
    null_values: np.ndarray,
    observed: float,
    out_png: Path,
    title: str,
    xlabel: str,
    annotation_label: str,
) -> None:
    values = np.asarray(null_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        fig, ax = plt.subplots(figsize=(6.4, 4.3))
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Random draws")
        ax.text(
            0.5,
            0.5,
            "No finite null samples available",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(out_png, dpi=190, bbox_inches="tight")
        plt.close(fig)
        return

    z_score = _z_score_against_null(values, observed)
    p_value = _empirical_two_sided(values, observed)
    null_mean = float(np.mean(values))
    null_sd = float(np.std(values, ddof=1))
    percentile = _percentile_against_null(values, observed)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.hist(values, bins=min(30, max(10, values.size // 4)), color="#9ecae1", edgecolor="white")
    ax.axvline(float(null_mean), color="#08519c", linestyle="--", linewidth=1.5, label="Null mean")
    ax.axvline(float(observed), color="#cb181d", linewidth=2.0, label="Observed selected")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Random draws")
    stat_text = (
        f"{annotation_label}\n"
        f"Observed={observed:.4g}\n"
        f"Null mean={null_mean:.4g}, SD={null_sd:.4g}\n"
        f"z={z_score:.3g}, p(two-sided)={p_value:.2g}\n"
        f"percentile={percentile:.2f}%"
    )
    ax.text(
        0.02,
        0.98,
        stat_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "#ffffff", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    selected_ts_root = args.selected_ts_root.expanduser().resolve()
    selected_voxel_indices_path = args.selected_voxel_indices.expanduser().resolve()
    roi_img_path = args.roi_img.expanduser().resolve()
    brain_img_path = args.brain_img.expanduser().resolve()
    reference_node_csv = (
        args.reference_node_csv.expanduser().resolve()
        if args.reference_node_csv is not None
        else (selected_ts_root.parent / "roi_nodes.csv").resolve()
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    excluded_subjects = _normalize_subject_exclusions(args.exclude_subjects)
    manifest_df, labels, label_meta = _load_manifest(manifest_path, excluded_subjects=excluded_subjects)
    selected_metrics = [
        name for name in normalize_metric_list(args.metrics)
        if name not in NULL_METRIC_EXCLUDE
    ]
    if not selected_metrics:
        raise ValueError("No metrics selected after excluding unsupported metrics for null analysis.")

    selected_voxel_count, candidate_flat, candidate_ijk, node_inventory_df, node_union_indices = _build_control_candidate_pool(
        roi_img_path=roi_img_path,
        brain_img_path=brain_img_path,
        selected_voxel_indices_path=selected_voxel_indices_path,
        reference_node_csv=reference_node_csv,
        midline_band_mm=float(args.midline_band_mm),
    )
    counts_df = _collect_control_counts(
        manifest_df=manifest_df,
        labels=labels,
        candidate_ijk=candidate_ijk,
        node_union_indices=node_union_indices,
        min_finite_trial_fraction=float(args.min_finite_trial_fraction),
    )
    all_nodes_df, keep_nodes, sample_sizes = _choose_keep_nodes(
        node_inventory_df=node_inventory_df,
        counts_df=counts_df,
        min_node_voxels=int(args.min_node_voxels),
    )
    if len(keep_nodes) < 2:
        top_nodes = all_nodes_df.nlargest(10, "min_available_voxels").loc[
            :,
            ["node_name", "min_available_voxels", "median_available_voxels", "max_available_voxels"],
        ]
        raise RuntimeError(
            "Fewer than two nodes satisfy the random-control availability constraint. "
            f"Best available nodes:\n{top_nodes.to_string(index=False)}"
        )

    keep_nodes_df = all_nodes_df.loc[all_nodes_df["node_name"].astype(str).isin(keep_nodes)].copy()
    keep_nodes_df = keep_nodes_df.sort_values("node_id").reset_index(drop=True)
    counts_df.to_csv(out_dir / "available_voxel_counts_by_label.csv", index=False)
    all_nodes_df.to_csv(out_dir / "control_node_inventory_all.csv", index=False)
    keep_nodes_df.to_csv(out_dir / "control_node_inventory_kept.csv", index=False)

    control_data_by_label, control_data_rows = _load_available_control_node_data(
        manifest_df=manifest_df,
        labels=labels,
        candidate_ijk=candidate_ijk,
        node_union_indices=node_union_indices,
        keep_nodes=keep_nodes,
        min_finite_trial_fraction=float(args.min_finite_trial_fraction),
    )
    pd.DataFrame(control_data_rows).to_csv(out_dir / "control_available_node_data_summary.csv", index=False)

    n_control_voxels = int(args.n_control_voxels) if int(args.n_control_voxels) > 0 else int(selected_voxel_count)

    control_draws = _sample_control_draws(
        control_data_by_label=control_data_by_label,
        labels=labels,
        keep_nodes=keep_nodes,
        n_control_voxels=int(n_control_voxels),
        n_draws=int(args.n_draws),
        random_seed=int(args.random_seed),
    )

    selected_ts_by_label, ordered_nodes = _load_selected_roi_timeseries(
        selected_ts_root=selected_ts_root,
        keep_nodes=keep_nodes,
        label_meta=label_meta,
    )
    selected_kept_voxel_count = _selected_kept_voxel_count(
        selected_ts_root=selected_ts_root,
        keep_nodes=keep_nodes,
    )
    if selected_kept_voxel_count is not None:
        ratio = float(selected_kept_voxel_count) / max(float(n_control_voxels), 1.0)
        if ratio < 0.5 or ratio > 2.0:
            raise RuntimeError(
                "Selected-vs-null voxel scale mismatch detected. "
                f"Selected kept-node voxels={selected_kept_voxel_count}, "
                f"null voxels per draw={int(n_control_voxels)}, ratio={ratio:.3f}. "
                "Use a selected_ts_root/roi_nodes pair built from the same selected_voxel_indices "
                "as this null run."
            )

    summary_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    total_metrics = len(selected_metrics)
    for metric_idx, metric_name in enumerate(selected_metrics, start=1):
        print(
            f"[{metric_idx}/{total_metrics}] Starting metric: {metric_name}",
            flush=True,
        )
        metric_out = out_dir / metric_name
        metric_out.mkdir(parents=True, exist_ok=True)

        try:
            selected_matrices = _compute_selected_metric_matrices(
                selected_ts_by_label=selected_ts_by_label,
                metric_name=metric_name,
                args=args,
            )
        except Exception as exc:
            skipped_rows.append(
                {
                    "metric": metric_name,
                    "reason": str(exc),
                }
            )
            continue
        selected_pairwise_df, selected_pairwise_matrix, display_labels = _pairwise_distance_matrix(
            matrices_by_label=selected_matrices,
            label_meta=label_meta,
            graph_distance=str(args.graph_distance),
        )
        selected_pairwise_df.to_csv(metric_out / "selected_pairwise_graph_distance.csv", index=False)
        _save_square_matrix_csv(
            metric_out / "selected_pairwise_graph_distance_matrix.csv",
            selected_pairwise_matrix,
            display_labels,
        )
        _plot_heatmap(
            selected_pairwise_matrix,
            display_labels,
            metric_out / "selected_pairwise_graph_distance_matrix.png",
            title=f"{metric_name} | selected graph distance",
            cbar_label="Graph distance",
        )
        observed_stats = _summarize_pair_contrasts(selected_pairwise_df)
        pd.DataFrame([observed_stats]).to_csv(metric_out / "selected_contrast_summary.csv", index=False)
        _plot_pair_means(
            observed_stats,
            metric_out / "selected_pair_type_means.png",
            title=f"{metric_name} | selected mean graph distances",
        )

        control_metric_draws = _compute_control_metric_matrices(
            draws=control_draws,
            metric_name=metric_name,
            args=args,
        )
        null_rows: list[dict[str, Any]] = []
        for draw_idx, draw_mats in enumerate(control_metric_draws):
            pairwise_df, _pairwise_matrix, _display_labels = _pairwise_distance_matrix(
                matrices_by_label=draw_mats,
                label_meta=label_meta,
                graph_distance=str(args.graph_distance),
            )
            stats_row = _summarize_pair_contrasts(pairwise_df)
            stats_row["draw"] = int(draw_idx)
            null_rows.append(stats_row)
        null_df = pd.DataFrame(null_rows).sort_values("draw").reset_index(drop=True)
        null_df.to_csv(metric_out / "random_draw_contrast_summary.csv", index=False)

        null_class_specs = [
            ("OFF-OFF", "mu_off_off"),
            ("ON-ON", "mu_on_on"),
            ("OFF-ON", "mu_off_on"),
        ]
        for class_label, null_key in null_class_specs:
            _plot_null_distribution(
                null_df[null_key].to_numpy(dtype=np.float64),
                float(observed_stats[null_key]),
                metric_out / f"{null_key}_selected_vs_null_distribution.png",
                title=f"{metric_name} | {class_label} mean selected vs null",
                xlabel=f"{class_label} mean graph distance",
                annotation_label=class_label,
            )

        delta_sep_null = null_df["delta_sep"].to_numpy(dtype=np.float64)
        delta_within_null = null_df["delta_within"].to_numpy(dtype=np.float64)

        summary_rows.append(
            {
                "metric": metric_name,
                "graph_distance": str(args.graph_distance),
                **observed_stats,
                "mu_off_off_z_vs_null": _z_score_against_null(
                    null_df["mu_off_off"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_off_off"]),
                ),
                "mu_off_off_p_empirical_two_sided": _empirical_two_sided(
                    null_df["mu_off_off"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_off_off"]),
                ),
                "mu_off_off_percentile_vs_null": _percentile_against_null(
                    null_df["mu_off_off"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_off_off"]),
                ),
                "mu_on_on_z_vs_null": _z_score_against_null(
                    null_df["mu_on_on"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_on_on"]),
                ),
                "mu_on_on_p_empirical_two_sided": _empirical_two_sided(
                    null_df["mu_on_on"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_on_on"]),
                ),
                "mu_on_on_percentile_vs_null": _percentile_against_null(
                    null_df["mu_on_on"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_on_on"]),
                ),
                "mu_off_on_z_vs_null": _z_score_against_null(
                    null_df["mu_off_on"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_off_on"]),
                ),
                "mu_off_on_p_empirical_two_sided": _empirical_two_sided(
                    null_df["mu_off_on"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_off_on"]),
                ),
                "mu_off_on_percentile_vs_null": _percentile_against_null(
                    null_df["mu_off_on"].to_numpy(dtype=np.float64),
                    float(observed_stats["mu_off_on"]),
                ),
                "null_mean_delta_sep": float(np.mean(delta_sep_null)),
                "null_sd_delta_sep": float(np.std(delta_sep_null, ddof=1)),
                "delta_sep_z_vs_null": _z_score_against_null(delta_sep_null, float(observed_stats["delta_sep"])),
                "delta_sep_p_empirical_right": _empirical_right_tail(
                    delta_sep_null,
                    float(observed_stats["delta_sep"]),
                ),
                "delta_sep_percentile_vs_null": _percentile_against_null(
                    delta_sep_null,
                    float(observed_stats["delta_sep"]),
                ),
                "null_mean_delta_within": float(np.mean(delta_within_null)),
                "null_sd_delta_within": float(np.std(delta_within_null, ddof=1)),
                "delta_within_z_vs_null": _z_score_against_null(
                    delta_within_null,
                    float(observed_stats["delta_within"]),
                ),
                "delta_within_p_empirical_two_sided": _empirical_two_sided(
                    delta_within_null,
                    float(observed_stats["delta_within"]),
                ),
                "delta_within_percentile_vs_null": _percentile_against_null(
                    delta_within_null,
                    float(observed_stats["delta_within"]),
                ),
                "kept_nodes": json.dumps(ordered_nodes),
                "n_control_voxels_per_draw": int(n_control_voxels),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("delta_sep_z_vs_null", ascending=False).reset_index(drop=True)
    summary_df.to_csv(out_dir / "observed_vs_random_null_summary.csv", index=False)
    pd.DataFrame(skipped_rows).to_csv(out_dir / "skipped_metrics.csv", index=False)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "out_dir": str(out_dir),
        "manifest_path": str(manifest_path),
        "selected_ts_root": str(selected_ts_root),
        "selected_voxel_indices": str(selected_voxel_indices_path),
        "roi_img": str(roi_img_path),
        "brain_img": str(brain_img_path),
        "reference_node_csv": str(reference_node_csv),
        "excluded_subjects": sorted(excluded_subjects),
        "metrics": selected_metrics,
        "graph_distance": str(args.graph_distance),
        "n_draws": int(args.n_draws),
        "candidate_union_voxels": int(candidate_flat.size),
        "kept_nodes": keep_nodes,
        "comparison_note": (
            "Observed selected-network metrics are recomputed from the saved selected ROI-average time series "
            "using the same retained ROI node set as the random non-selected controls. Random controls are "
            "sampled from brain voxels with selected voxels removed, then assigned into control ROIs and "
            "averaged within node before metric computation."
        ),
        "args": _json_safe(vars(args)),
        "n_control_voxels_per_draw": int(n_control_voxels),
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2),
        encoding="utf-8",
    )

    print(
        f"Null analysis complete. Kept {len(keep_nodes)} comparable nodes and evaluated "
        f"{len(summary_rows)} metric(s) across {int(args.n_draws)} random draws. "
        f"Outputs written to {out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
