#!/usr/bin/env python3
"""Compute advanced ROI-network metrics from roi_edge_connectivity outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn import plotting

from roi_metrics import METRIC_REGISTRY, normalize_metric_list


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

    parser.add_argument("--wavelet-min-scale", type=int, default=2)
    parser.add_argument("--wavelet-max-scale", type=int, default=20)
    parser.add_argument("--wavelet-omega0", type=float, default=6.0)
    parser.add_argument("--wavelet-smooth-scale-sigma", type=float, default=1.0)
    parser.add_argument("--wavelet-smooth-time-sigma", type=float, default=2.0)
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
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_node_geometry(network_dir: Path) -> tuple[np.ndarray | None, list[str] | None]:
    node_path = network_dir / "roi_nodes.csv"
    if not node_path.exists():
        return None, None

    try:
        df = pd.read_csv(node_path)
    except Exception:
        return None, None

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
        colorbar=True,
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
            colorbar=True,
            node_size=max(3.0, float(node_size) / 16.0),
            title=title,
        )
        view.save_as_html(str(out_html))
        html_path = str(out_html)

    out_png.with_suffix(".labels.txt").write_text("\n".join(node_labels), encoding="utf-8")
    return {"matrix_for_connectome": mode, "edge_threshold": edge_threshold, "connectome_html": html_path}


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
    if metric_name in {"linear_granger", "nonlinear_granger"}:
        return {
            "max_lag": int(args.granger_max_lag),
            "ridge": float(args.granger_ridge),
        }
    if metric_name == "wavelet_transform_coherence":
        return {
            "min_scale": int(args.wavelet_min_scale),
            "max_scale": int(args.wavelet_max_scale),
            "omega0": float(args.wavelet_omega0),
            "smooth_scale_sigma": float(args.wavelet_smooth_scale_sigma),
            "smooth_time_sigma": float(args.wavelet_smooth_time_sigma),
        }
    return {}


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
        raise RuntimeError("No condition labels found to process.")

    out_root = network_dir / out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    node_labels_from_summary = summary.get("roi_labels")
    node_coords_mm, node_labels_from_nodes = _load_node_geometry(network_dir)

    for label in selected_labels:
        ts_path = network_dir / label / f"roi_timeseries_{label}.npy"
        if not ts_path.exists():
            print(f"Skipping label={label}: missing {ts_path}", flush=True)
            continue

        roi_ts = np.load(ts_path)
        if roi_ts.ndim != 2:
            print(f"Skipping label={label}: ROI time series is not 2D ({roi_ts.shape})", flush=True)
            continue

        n_nodes = int(roi_ts.shape[0])
        if (
            isinstance(node_labels_from_nodes, list)
            and node_coords_mm is not None
            and len(node_labels_from_nodes) == n_nodes
            and int(node_coords_mm.shape[0]) == n_nodes
        ):
            node_labels = [str(v) for v in node_labels_from_nodes]
        elif isinstance(node_labels_from_summary, list) and len(node_labels_from_summary) == n_nodes:
            node_labels = [str(v) for v in node_labels_from_summary]
        else:
            node_labels = _auto_labels(n_nodes)

        cond_out = out_root / label
        cond_out.mkdir(parents=True, exist_ok=True)

        for metric_name in selected_metrics:
            metric_fn = METRIC_REGISTRY[metric_name]
            kwargs = _metric_kwargs(metric_name, args)
            result = metric_fn(roi_ts, **kwargs)

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
            if node_coords_mm is not None and int(node_coords_mm.shape[0]) == n_nodes:
                connectome_meta = _plot_metric_connectome(
                    matrix=matrix,
                    node_coords_mm=node_coords_mm,
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
