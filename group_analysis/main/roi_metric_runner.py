#!/usr/bin/env python3
"""Compute advanced ROI-network metrics from roi_edge_connectivity outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting

_HERE = Path(__file__).resolve().parent
_GROUP_ANALYSIS_DIR = _HERE.parent
_REPO_ROOT = _GROUP_ANALYSIS_DIR.parent
if str(_GROUP_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from group_analysis.roi_metrics import METRIC_REGISTRY, normalize_metric_list
except ModuleNotFoundError:
    from roi_metrics import METRIC_REGISTRY, normalize_metric_list

ALWAYS_EXCLUDED_ROI_PATTERNS = (
    "ventricular csf",
    "ventrical csf",
    "lateral ventricle",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute advanced ROI network metrics from per-condition ROI time series "
            "produced by roi_edge_connectivity.py"
        )
    )
    parser.add_argument(
        "--network-dir",
        type=Path,
        default=Path("results/connectivity/roi_edge_network"),
        help="Directory produced by roi_edge_connectivity.py.",
    )
    parser.add_argument(
        "--out-subdir",
        default="advanced_metrics",
        help="Output subfolder inside --network-dir.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated metric list or 'all'.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional comma-separated condition labels. Empty uses summary file_labels.",
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

    parser.add_argument("--wavelet-min-scale", type=int, default=2)
    parser.add_argument("--wavelet-max-scale", type=int, default=20)
    parser.add_argument("--wavelet-omega0", type=float, default=6.0)
    parser.add_argument("--wavelet-smooth-scale-sigma", type=float, default=1.0)
    parser.add_argument("--wavelet-smooth-time-sigma", type=float, default=2.0)
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
        "--connectome-edge-threshold-percentile",
        type=float,
        default=85.0,
        help="Percentile threshold used for per-metric connectome visualization.",
    )
    parser.add_argument(
        "--connectome-node-size",
        type=float,
        default=135.0,
        help="Node size for per-metric connectome visualization.",
    )
    parser.add_argument(
        "--connectome-linewidth",
        type=float,
        default=5.0,
        help="Edge linewidth for per-metric connectome visualization.",
    )
    parser.add_argument(
        "--no-connectome-html",
        action="store_true",
        help="Skip interactive HTML connectome output for advanced metrics.",
    )
    parser.add_argument(
        "--respect-temporal-boundaries",
        action="store_true",
        help=(
            "For temporal metrics, compute per-run matrices using manifest-defined run boundaries "
            "and aggregate across runs. Disabled by default to preserve legacy behavior."
        ),
    )
    parser.add_argument(
        "--temporal-manifest-path",
        type=Path,
        default=Path("results/connectivity/tmp/concat_manifest_group.tsv"),
        help="Run-boundary manifest TSV used when --respect-temporal-boundaries is enabled.",
    )
    parser.add_argument(
        "--temporal-condition-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_beta_trials_gvs_column_indices.npz"),
        help=(
            "NPZ with per-condition column indices into selected_beta_trials.npy "
            "used for run-wise temporal segmentation."
        ),
    )
    parser.add_argument(
        "--temporal-metrics",
        default="linear_granger,nonlinear_granger",
        help=(
            "Comma-separated metrics to compute with temporal segmentation when "
            "--respect-temporal-boundaries is enabled. Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--temporal-min-trials",
        type=int,
        default=8,
        help=(
            "Minimum trials required in a run-segment for segmented temporal metric computation. "
            "Segments below this are skipped."
        ),
    )
    parser.add_argument(
        "--exclude-rois",
        nargs="+",
        default=[],
        metavar="NAME",
        help=(
            "Extra ROI names (or substrings) to exclude from metric computation (case-insensitive). "
            "Ventricular CSF is always excluded."
        ),
    )

    return parser.parse_args()


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if np.isfinite(val) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def _auto_labels(n_nodes: int) -> list[str]:
    return [f"Node_{idx:03d}" for idx in range(1, n_nodes + 1)]


def _build_exclude_patterns(extra: list[str] | None) -> list[str]:
    return sorted(
        {
            str(v).strip().lower()
            for v in [*ALWAYS_EXCLUDED_ROI_PATTERNS, *((extra or []))]
            if str(v).strip()
        }
    )


def _plot_metric_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    out_png: Path,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
) -> None:
    values = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = values.shape[0]

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    im = ax.imshow(values, cmap=cmap, vmin=float(vmin), vmax=float(vmax))
    ax.set_title(title)
    ax.set_xlabel("ROI nodes")
    ax.set_ylabel("ROI nodes")

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


def _load_summary(network_dir: Path) -> dict:
    summary_path = network_dir / "roi_edge_network_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _load_node_geometry(network_dir: Path) -> tuple[np.ndarray | None, list[str] | None]:
    node_path = network_dir / "roi_nodes.csv"
    if not node_path.exists():
        return None, None

    df = pd.read_csv(node_path)

    required = {"x_mm", "y_mm", "z_mm"}
    if not required.issubset(set(df.columns)):
        return None, None

    if "node_id" in df.columns:
        df = df.sort_values("node_id")

    coords = df[["x_mm", "y_mm", "z_mm"]].to_numpy(dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        return None, None

    if "node_name" in df.columns:
        labels = [str(v) for v in df["node_name"].tolist()]
    elif "roi_name" in df.columns:
        labels = [str(v) for v in df["roi_name"].tolist()]
    else:
        labels = _auto_labels(coords.shape[0])
    return coords, labels


def _plot_metric_connectome(
    matrix: np.ndarray,
    node_coords_mm: np.ndarray,
    node_labels: list[str],
    out_png: Path,
    out_html: Path | None,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    edge_threshold_percentile: float,
    node_size: float,
    edge_linewidth: float,
    directed: bool,
) -> dict:
    conn = np.asarray(matrix, dtype=np.float64)
    if directed:
        conn_plot = 0.5 * (conn + conn.T)
        mode = "symmetrized_mean"
    else:
        conn_plot = conn
        mode = "as_is"

    conn_plot = np.nan_to_num(conn_plot, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(conn_plot, 0.0)
    edge_threshold = f"{float(np.clip(edge_threshold_percentile, 0.0, 100.0)):.1f}%"
    offdiag = np.abs(conn_plot[~np.eye(conn_plot.shape[0], dtype=bool)])
    offdiag = offdiag[np.isfinite(offdiag)]
    if offdiag.size == 0:
        has_visible_edges = False
    else:
        thr_val = float(
            np.percentile(
                offdiag,
                float(np.clip(edge_threshold_percentile, 0.0, 100.0)),
            )
        )
        has_visible_edges = bool(np.any(offdiag > max(thr_val, 0.0)))

    plotting.plot_connectome(
        conn_plot,
        node_coords_mm,
        node_color="#1f78b4",
        node_size=float(node_size),
        edge_cmap=str(cmap),
        edge_vmin=float(vmin),
        edge_vmax=float(vmax),
        edge_threshold=edge_threshold,
        edge_kwargs={"linewidth": float(edge_linewidth), "alpha": 0.85},
        node_kwargs={"alpha": 0.95},
        colorbar=has_visible_edges,
        title=title,
        output_file=str(out_png),
    )

    html_path = None
    if out_html is not None:
        view = plotting.view_connectome(
            conn_plot,
            node_coords_mm,
            edge_threshold=edge_threshold,
            edge_cmap=str(cmap),
            symmetric_cmap=False,
            linewidth=max(1.5, float(edge_linewidth) * 1.8),
            colorbar=has_visible_edges,
            node_size=max(3.0, float(node_size) / 16.0),
            title=title,
        )
        view.save_as_html(str(out_html))
        html_path = str(out_html)

    out_png.with_suffix(".labels.txt").write_text("\n".join(node_labels), encoding="utf-8")
    return {
        "matrix_for_connectome": mode,
        "edge_threshold": edge_threshold,
        "connectome_has_visible_edges": bool(has_visible_edges),
        "connectome_html": html_path,
    }


def _resolve_labels(network_dir: Path, requested: str, summary: dict) -> list[str]:
    if requested.strip():
        return [item.strip() for item in requested.split(",") if item.strip()]

    labels = summary.get("file_labels")
    if isinstance(labels, list) and labels:
        return [str(x) for x in labels]

    discovered = []
    for sub in sorted(network_dir.iterdir()):
        if not sub.is_dir():
            continue
        candidate = sub / f"roi_timeseries_{sub.name}.npy"
        if candidate.exists():
            discovered.append(sub.name)
    return discovered


def _metric_kwargs(metric_name: str, args: argparse.Namespace) -> dict:
    if metric_name == "mutual_information":
        return {"n_bins": int(args.mi_bins)}
    if metric_name == "mutual_information_ksg":
        return {
            "k": int(args.mi_ksg_k),
            "jitter": float(args.mi_ksg_jitter),
        }
    if metric_name == "linear_granger":
        return {
            "max_lag": int(args.granger_max_lag),
            "ridge": float(args.granger_ridge),
        }
    if metric_name == "nonlinear_granger":
        sigma_raw = float(getattr(args, "kernel_granger_sigma", 0.0))
        return {
            "max_lag": int(args.granger_max_lag),
            "ridge": float(args.granger_ridge),
            "kernel": str(getattr(args, "kernel_granger_kernel", "ip")),
            "degree": int(getattr(args, "kernel_granger_degree", 2)),
            "sigma": None if sigma_raw <= 0.0 else sigma_raw,
            "eig_frac": float(getattr(args, "kernel_granger_eig_frac", 1e-6)),
            "alpha": float(getattr(args, "kernel_granger_alpha", 0.05)),
        }
    if metric_name == "wavelet_transform_coherence":
        fmin_hz = float(getattr(args, "wavelet_fmin_hz", 0.01))
        fmax_hz = float(getattr(args, "wavelet_fmax_hz", 0.1))
        return {
            "min_scale": int(args.wavelet_min_scale),
            "max_scale": int(args.wavelet_max_scale),
            "omega0": float(args.wavelet_omega0),
            "smooth_scale_sigma": float(args.wavelet_smooth_scale_sigma),
            "smooth_time_sigma": float(args.wavelet_smooth_time_sigma),
            "fmin_hz": None if fmin_hz <= 0.0 else fmin_hz,
            "fmax_hz": None if fmax_hz <= 0.0 else fmax_hz,
            "mask_coi": bool(getattr(args, "wavelet_mask_coi", True)),
            "coi_factor": float(getattr(args, "wavelet_coi_factor", np.sqrt(2.0))),
        }
    return {}


def _parse_temporal_metric_set(metric_arg: str | list[str] | None) -> set[str]:
    default = {"linear_granger", "nonlinear_granger", "kernel_granger"}
    if metric_arg is None:
        return default

    if isinstance(metric_arg, (list, tuple)):
        requested = [str(v).strip() for v in metric_arg if str(v).strip()]
    else:
        text = str(metric_arg).strip()
        if not text:
            return default
        lowered = text.lower()
        if lowered in {"none", "off", "false", "0"}:
            return set()
        if lowered == "all":
            return set(METRIC_REGISTRY.keys())
        requested = [item.strip() for item in text.split(",") if item.strip()]

    unknown = [name for name in requested if name not in METRIC_REGISTRY]
    if unknown:
        valid = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise ValueError(f"Unknown temporal metrics: {unknown}. Valid values include: {valid}")
    return set(requested)


def _load_temporal_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Temporal manifest not found: {manifest_path}")

    manifest_df = pd.read_csv(manifest_path, sep="\t")
    required_cols = {"offset_start", "offset_end", "sub_tag", "ses", "run"}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        raise ValueError(
            f"Temporal manifest missing required columns {sorted(missing)}: {manifest_path}"
        )

    manifest_df = manifest_df.sort_values("offset_start").reset_index(drop=True)
    starts = pd.to_numeric(manifest_df["offset_start"], errors="raise").to_numpy(dtype=np.int64)
    ends = pd.to_numeric(manifest_df["offset_end"], errors="raise").to_numpy(dtype=np.int64)

    if np.any(starts < 0) or np.any(ends < starts):
        raise ValueError(f"Temporal manifest contains invalid trial offsets: {manifest_path}")
    if starts.size > 1 and np.any(starts[1:] < ends[:-1]):
        raise ValueError(f"Temporal manifest has overlapping run segments: {manifest_path}")

    return manifest_df


def _load_condition_index_map(index_path: Path) -> dict[str, np.ndarray]:
    if not index_path.exists():
        raise FileNotFoundError(f"Condition-index NPZ not found: {index_path}")

    pack = np.load(index_path)
    mapping: dict[str, np.ndarray] = {}
    for key in pack.files:
        cols = np.asarray(pack[key], dtype=np.int64).ravel()
        if cols.size > 1 and np.any(cols[1:] < cols[:-1]):
            cols = np.sort(cols)
        mapping[str(key)] = cols

    if not mapping:
        raise ValueError(f"Condition-index NPZ has no arrays: {index_path}")
    return mapping


def _resolve_condition_columns(
    condition_index_map: dict[str, np.ndarray], label: str
) -> np.ndarray | None:
    if label in condition_index_map:
        return condition_index_map[label]
    label_lower = str(label).strip().lower()
    for key, values in condition_index_map.items():
        if str(key).strip().lower() == label_lower:
            return values
    return None


def _build_condition_segments(
    manifest_df: pd.DataFrame,
    condition_cols_global: np.ndarray,
    n_local_trials: int,
) -> list[dict]:
    cols = np.asarray(condition_cols_global, dtype=np.int64).ravel()
    if cols.size != int(n_local_trials):
        raise ValueError(
            f"Condition trial count mismatch between roi_ts ({n_local_trials}) and index map ({cols.size})."
        )
    if cols.size > 1 and np.any(cols[1:] < cols[:-1]):
        raise ValueError("Condition global column indices must be sorted in ascending order.")

    segments: list[dict] = []
    assigned = 0
    for row in manifest_df.itertuples(index=False):
        start = int(row.offset_start)
        end = int(row.offset_end)
        if end <= start:
            continue

        left = int(np.searchsorted(cols, start, side="left"))
        right = int(np.searchsorted(cols, end, side="left"))
        if right <= left:
            continue

        seg_cols = cols[left:right]
        if int(seg_cols[0]) < start or int(seg_cols[-1]) >= end:
            raise ValueError(
                f"Condition columns spill outside manifest run range [{start}, {end})."
            )

        local_idx = np.arange(left, right, dtype=np.int64)
        segments.append(
            {
                "sub_tag": str(row.sub_tag),
                "ses": int(row.ses),
                "run": int(row.run),
                "offset_start": start,
                "offset_end": end,
                "local_idx": local_idx,
            }
        )
        assigned += int(local_idx.size)

    if assigned != int(cols.size):
        raise ValueError(
            f"Condition columns were not fully covered by manifest runs ({assigned}/{cols.size})."
        )

    return segments


def _aggregate_temporal_metric_over_segments(
    metric_name: str,
    metric_fn,
    roi_ts: np.ndarray,
    kwargs: dict,
    segments: list[dict],
    min_trials: int,
) -> tuple[dict | None, dict]:
    lag = int(max(1, kwargs.get("max_lag", 1)))
    min_required = int(max(1, min_trials, lag + 5))

    matrices: list[np.ndarray] = []
    weights: list[float] = []
    template_result: dict | None = None
    skipped_short = 0

    n_nodes = int(roi_ts.shape[0])
    for segment in segments:
        local_idx = np.asarray(segment["local_idx"], dtype=np.int64)
        seg_len = int(local_idx.size)
        if seg_len < min_required:
            skipped_short += 1
            continue

        seg_ts = roi_ts[:, local_idx]
        seg_result = metric_fn(seg_ts, **kwargs)
        seg_matrix = np.asarray(seg_result.get("matrix"), dtype=np.float64)
        if seg_matrix.shape != (n_nodes, n_nodes):
            raise ValueError(
                f"{metric_name} segment matrix shape {seg_matrix.shape} != {(n_nodes, n_nodes)}"
            )
        matrices.append(seg_matrix)
        weights.append(float(max(seg_len - lag, 1)))
        if template_result is None:
            template_result = dict(seg_result)

    info = {
        "enabled": True,
        "metric": str(metric_name),
        "n_segments_total": int(len(segments)),
        "n_segments_used": int(len(matrices)),
        "n_segments_skipped_short": int(skipped_short),
        "min_trials_per_segment": int(min_required),
        "weighting": "n_trials_minus_max_lag",
    }

    if not matrices or template_result is None:
        return None, info

    stacked = np.stack(matrices, axis=0)
    weight_array = np.asarray(weights, dtype=np.float64)
    aggregated = np.average(stacked, axis=0, weights=weight_array)
    template_result["matrix"] = aggregated

    finite = aggregated[np.isfinite(aggregated)]
    if finite.size:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    else:
        vmin, vmax = 0.0, 1e-3
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(vmin + 1e-3)
    template_result["vmin"] = vmin
    template_result["vmax"] = vmax
    template_result["temporal_aggregation"] = {
        **info,
        "total_weight": float(np.sum(weight_array)),
    }

    return template_result, info


def run_metric_pipeline(
    network_dir: Path,
    out_subdir: str,
    metrics: str | list[str],
    labels: list[str] | None,
    args: argparse.Namespace,
) -> dict:
    network_dir = network_dir.expanduser().resolve()
    if not network_dir.exists():
        raise FileNotFoundError(f"Network directory not found: {network_dir}")

    summary = _load_summary(network_dir)
    selected_metrics = normalize_metric_list(metrics)
    selected_labels = labels if labels is not None else _resolve_labels(network_dir, requested="", summary=summary)
    if not selected_labels:
        raise ValueError("No condition labels found to process.")

    out_root = network_dir / out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    node_labels_from_summary = summary.get("roi_labels")
    node_coords_mm, node_labels_from_nodes = _load_node_geometry(network_dir)
    exclude_patterns = _build_exclude_patterns(getattr(args, "exclude_rois", []))
    respect_temporal_boundaries = bool(getattr(args, "respect_temporal_boundaries", False))
    temporal_metric_set = _parse_temporal_metric_set(getattr(args, "temporal_metrics", None))
    temporal_min_trials = int(max(1, int(getattr(args, "temporal_min_trials", 8))))

    temporal_manifest_df: pd.DataFrame | None = None
    condition_index_map: dict[str, np.ndarray] | None = None
    temporal_ready = False
    temporal_manifest_path = None
    temporal_index_path = None
    temporal_load_error = ""

    if respect_temporal_boundaries and temporal_metric_set:
        temporal_manifest_path = Path(getattr(args, "temporal_manifest_path", "")).expanduser().resolve()
        temporal_index_path = Path(
            getattr(args, "temporal_condition_indices_path", "")
        ).expanduser().resolve()
        try:
            temporal_manifest_df = _load_temporal_manifest(temporal_manifest_path)
            condition_index_map = _load_condition_index_map(temporal_index_path)
            temporal_ready = True
            print(
                "Temporal boundary mode enabled: "
                f"manifest={temporal_manifest_path}, condition_indices={temporal_index_path}",
                flush=True,
            )
        except Exception as exc:
            temporal_load_error = str(exc)
            print(
                f"Warning: temporal boundary mode requested but unavailable ({exc}). "
                "Falling back to legacy whole-series metric computation.",
                flush=True,
            )

    for label in selected_labels:
        ts_path = network_dir / label / f"roi_timeseries_{label}.npy"
        if not ts_path.exists():
            print(f"Skipping label={label}: missing {ts_path}", flush=True)
            continue

        roi_ts = np.load(ts_path)
        if roi_ts.ndim != 2:
            print(f"Skipping label={label}: ROI time series is not 2D ({roi_ts.shape})", flush=True)
            continue

        n_nodes_raw = int(roi_ts.shape[0])
        if (
            isinstance(node_labels_from_nodes, list)
            and node_coords_mm is not None
            and len(node_labels_from_nodes) == n_nodes_raw
            and int(node_coords_mm.shape[0]) == n_nodes_raw
        ):
            node_labels = [str(v) for v in node_labels_from_nodes]
        elif isinstance(node_labels_from_summary, list) and len(node_labels_from_summary) == n_nodes_raw:
            node_labels = [str(v) for v in node_labels_from_summary]
        else:
            node_labels = _auto_labels(n_nodes_raw)

        coords_for_label = None
        if node_coords_mm is not None and int(node_coords_mm.shape[0]) == n_nodes_raw:
            coords_for_label = node_coords_mm

        keep = np.asarray(
            [not any(pattern in str(node).lower() for pattern in exclude_patterns) for node in node_labels],
            dtype=bool,
        )
        if not np.all(keep):
            dropped = int(np.count_nonzero(~keep))
            if dropped >= len(node_labels):
                print(
                    f"Skipping label={label}: all ROI nodes excluded by patterns {exclude_patterns}.",
                    flush=True,
                )
                continue
            roi_ts = roi_ts[keep, :]
            node_labels = [node for node, ok in zip(node_labels, keep.tolist()) if ok]
            if coords_for_label is not None:
                coords_for_label = coords_for_label[keep]
            print(
                f"Filtered label={label}: removed {dropped} excluded ROI node(s).",
                flush=True,
            )

        n_nodes = int(roi_ts.shape[0])
        temporal_segments: list[dict] | None = None
        temporal_segment_error = ""
        if temporal_ready and temporal_manifest_df is not None and condition_index_map is not None:
            try:
                condition_cols = _resolve_condition_columns(condition_index_map, label)
                if condition_cols is None:
                    raise KeyError(
                        f"Condition label {label!r} not found in {temporal_index_path}"
                    )
                temporal_segments = _build_condition_segments(
                    temporal_manifest_df,
                    condition_cols_global=condition_cols,
                    n_local_trials=int(roi_ts.shape[1]),
                )
                print(
                    f"Temporal segments for label={label}: {len(temporal_segments)}",
                    flush=True,
                )
            except Exception as exc:
                temporal_segment_error = str(exc)
                temporal_segments = None
                print(
                    f"Warning: temporal segmentation unavailable for label={label} ({exc}). "
                    "Using legacy whole-series computation for this label.",
                    flush=True,
                )

        cond_out = out_root / label
        cond_out.mkdir(parents=True, exist_ok=True)

        for metric_name in selected_metrics:
            metric_fn = METRIC_REGISTRY[metric_name]
            kwargs = _metric_kwargs(metric_name, args)
            if (
                temporal_ready
                and metric_name in temporal_metric_set
                and temporal_segments is not None
            ):
                result, temporal_info = _aggregate_temporal_metric_over_segments(
                    metric_name=metric_name,
                    metric_fn=metric_fn,
                    roi_ts=roi_ts,
                    kwargs=kwargs,
                    segments=temporal_segments,
                    min_trials=temporal_min_trials,
                )
                if result is None:
                    result = metric_fn(roi_ts, **kwargs)
                    result["temporal_aggregation"] = {
                        **temporal_info,
                        "enabled": False,
                        "fallback": "legacy_whole_series",
                    }
                    print(
                        f"Temporal aggregation for metric={metric_name} label={label} "
                        "had no usable segments; used legacy whole-series computation.",
                        flush=True,
                    )
                else:
                    print(
                        f"Temporally aggregated metric={metric_name} label={label}: "
                        f"used {temporal_info['n_segments_used']}/{temporal_info['n_segments_total']} segments.",
                        flush=True,
                    )
            else:
                result = metric_fn(roi_ts, **kwargs)
                if (
                    respect_temporal_boundaries
                    and metric_name in temporal_metric_set
                    and not temporal_ready
                ):
                    result["temporal_aggregation"] = {
                        "enabled": False,
                        "fallback": "legacy_whole_series",
                        "reason": temporal_load_error or "temporal mode not available",
                    }
                elif (
                    temporal_ready
                    and metric_name in temporal_metric_set
                    and temporal_segments is None
                ):
                    result["temporal_aggregation"] = {
                        "enabled": False,
                        "fallback": "legacy_whole_series",
                        "reason": temporal_segment_error or "temporal segmentation unavailable for label",
                    }

            matrix = np.asarray(result.get("matrix"), dtype=np.float64)
            if matrix.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"{metric_name} for {label} returned shape {matrix.shape}, expected {(n_nodes, n_nodes)}"
                )

            metric_out = cond_out / metric_name
            metric_out.mkdir(parents=True, exist_ok=True)

            np.save(metric_out / f"{metric_name}.npy", matrix.astype(np.float32, copy=False))
            pd.DataFrame(matrix, index=node_labels, columns=node_labels).to_csv(
                metric_out / f"{metric_name}.csv"
            )

            vmin = float(result.get("vmin", np.nanmin(matrix)))
            vmax = float(result.get("vmax", np.nanmax(matrix)))
            if not np.isfinite(vmin):
                vmin = float(np.nanmin(np.nan_to_num(matrix, nan=0.0)))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = float(vmin + 1e-3)

            cmap = str(result.get("cmap", "jet"))
            directed = bool(result.get("directed", False))
            desc = str(result.get("description", ""))

            _plot_metric_heatmap(
                matrix=matrix,
                labels=node_labels,
                out_png=metric_out / f"{metric_name}.png",
                title=f"{metric_name} ({label})",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar_label=f"{metric_name} value",
            )

            connectome_meta = None
            if coords_for_label is not None and int(coords_for_label.shape[0]) == n_nodes:
                connectome_meta = _plot_metric_connectome(
                    matrix=matrix,
                    node_coords_mm=coords_for_label,
                    node_labels=node_labels,
                    out_png=metric_out / f"{metric_name}_connectome.png",
                    out_html=None
                    if bool(getattr(args, "no_connectome_html", False))
                    else metric_out / f"{metric_name}_connectome.html",
                    title=f"{metric_name} connectome ({label})",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    edge_threshold_percentile=float(
                        getattr(args, "connectome_edge_threshold_percentile", 85.0)
                    ),
                    node_size=float(getattr(args, "connectome_node_size", 135.0)),
                    edge_linewidth=float(getattr(args, "connectome_linewidth", 5.0)),
                    directed=directed,
                )

            meta = {
                "label": label,
                "metric": metric_name,
                "description": desc,
                "directed": directed,
                "n_nodes": n_nodes,
                "vmin": vmin,
                "vmax": vmax,
                "kwargs": kwargs,
            }
            extra_keys = {k: v for k, v in result.items() if k != "matrix" and k not in meta}
            meta["extra"] = extra_keys
            if connectome_meta is not None:
                meta["connectome"] = connectome_meta
            (metric_out / f"{metric_name}_meta.json").write_text(
                json.dumps(_json_safe(meta), indent=2),
                encoding="utf-8",
            )

            summary_rows.append(
                {
                    "label": label,
                    "metric": metric_name,
                    "directed": directed,
                    "n_nodes": n_nodes,
                    "value_min": float(np.nanmin(matrix)),
                    "value_max": float(np.nanmax(matrix)),
                    "value_mean": float(np.nanmean(matrix)),
                    "value_std": float(np.nanstd(matrix)),
                    "vmin_plot": vmin,
                    "vmax_plot": vmax,
                    "output_dir": str(metric_out),
                    "connectome_png": str(metric_out / f"{metric_name}_connectome.png")
                    if connectome_meta is not None
                    else "",
                    "connectome_html": str(metric_out / f"{metric_name}_connectome.html")
                    if (connectome_meta is not None and connectome_meta.get("connectome_html") is not None)
                    else "",
                }
            )
            print(f"Saved metric={metric_name} label={label} -> {metric_out}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "metric_run_summary.csv"
    if not summary_df.empty:
        summary_df.to_csv(summary_csv, index=False)

    run_manifest = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "network_dir": str(network_dir),
        "out_root": str(out_root),
        "selected_labels": selected_labels,
        "selected_metrics": selected_metrics,
        "excluded_roi_patterns": exclude_patterns,
        "temporal_boundary_mode": {
            "requested": bool(respect_temporal_boundaries),
            "active": bool(temporal_ready),
            "temporal_metrics": sorted(temporal_metric_set),
            "temporal_min_trials": int(temporal_min_trials),
            "manifest_path": str(temporal_manifest_path) if temporal_manifest_path is not None else "",
            "condition_indices_path": str(temporal_index_path) if temporal_index_path is not None else "",
            "load_error": str(temporal_load_error) if temporal_load_error else "",
        },
        "n_outputs": int(summary_df.shape[0]),
    }
    (out_root / "run_manifest.json").write_text(
        json.dumps(_json_safe(run_manifest), indent=2),
        encoding="utf-8",
    )

    return {
        "out_root": out_root,
        "summary_csv": summary_csv,
        "n_outputs": int(summary_df.shape[0]),
        "selected_labels": selected_labels,
        "selected_metrics": selected_metrics,
    }


def main() -> None:
    args = _parse_args()

    summary = _load_summary(args.network_dir.expanduser().resolve())
    labels = _resolve_labels(args.network_dir.expanduser().resolve(), requested=args.labels, summary=summary)

    run_info = run_metric_pipeline(
        network_dir=args.network_dir,
        out_subdir=args.out_subdir,
        metrics=args.metrics,
        labels=labels,
        args=args,
    )

    print(
        f"Metric pipeline done: outputs={run_info['n_outputs']} "
        f"labels={len(run_info['selected_labels'])} metrics={len(run_info['selected_metrics'])}",
        flush=True,
    )
    print(f"Summary: {run_info['summary_csv']}", flush=True)


if __name__ == "__main__":
    main()
