#!/usr/bin/env python3
"""Extract top between-session edge changes and plot subject-wise delta heatmaps."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors, ticker
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_RE = re.compile(r"^(sub-[^_]+)_ses-(\d+)$")
GENERIC_NODE_LABEL_RE = re.compile(r"^([LR]) ROI_(\d+)$")
NODE_LABEL_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (" (relative)", ""),
    (" (Control & monitoring)", ""),
    ("Dorsolateral Prefrontal Cortex", "DLPFC"),
    ("vmPFC / dmPFC", "vmPFC/dmPFC"),
    ("Inferior Frontal Gyrus", "IFG"),
    ("Insular / Opercular Cortex", "Insular/Opercular"),
    ("Other Cerebral Cortex", "Other ctx"),
    ("Parietal Cortex", "Parietal"),
    ("Temporal Cortex", "Temporal"),
    ("Occipital Cortex", "Occipital"),
    ("Cingulate Cortex", "Cingulate"),
    ("Basal Ganglia", "Basal ganglia"),
    ("Unassigned Active Voxels", "Unassigned"),
)


@dataclass(frozen=True)
class SessionMetric:
    label: str
    subject: str
    session: int
    matrix: np.ndarray


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    advanced_root = script_dir.parent
    parser = argparse.ArgumentParser(
        description=(
            "Load paired subject/session connectivity matrices, rank edges by "
            "between-session change, and save a subject-by-edge delta heatmap."
        )
    )
    parser.add_argument(
        "--advanced-root",
        type=Path,
        default=advanced_root,
        help="Directory containing sub-*_ses-* metric folders.",
    )
    parser.add_argument(
        "--metric",
        default="wavelet_transform_coherence",
        help="Connectivity metric folder name to analyze.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <advanced-root>/<metric>/.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=19,
        help="Number of top-changing edges to report and plot.",
    )
    parser.add_argument(
        "--top-percentile",
        type=float,
        default=95.0,
        help=(
            "Percentile threshold applied to edge rank scores. Edges at or above this "
            "percentile are selected. Set to 100 to fall back to --top-k only."
        ),
    )
    parser.add_argument(
        "--rank-by",
        default="mean_abs_delta",
        choices=("mean_abs_delta", "abs_mean_delta"),
        help="How to rank edges across paired subjects.",
    )
    return parser.parse_args()


def _parse_label_session(label: str) -> tuple[str, int] | None:
    match = LABEL_RE.match(label)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _edge_rows_cols(n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1))
    return rows.astype(np.int64), cols.astype(np.int64)


def _resolve_node_labels(advanced_root: Path, labels: list[str]) -> list[str]:
    candidate_paths: list[Path] = []
    for candidate in (
        advanced_root / "roi_nodes.csv",
        advanced_root.parent / "roi_nodes.csv",
        REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "roi_nodes.csv",
        REPO_ROOT / "results" / "connectivity_new" / "roi_edge_network" / "roi_nodes.csv",
    ):
        if candidate not in candidate_paths:
            candidate_paths.append(candidate)

    for roi_nodes_path in candidate_paths:
        if not roi_nodes_path.exists():
            continue

        try:
            roi_df = pd.read_csv(roi_nodes_path)
        except Exception:
            continue

        required = {"base_roi_id", "hemisphere", "node_name"}
        if not required.issubset(set(roi_df.columns)):
            continue

        lookup: dict[tuple[str, int], str] = {}
        for row in roi_df.itertuples(index=False):
            try:
                key = (str(row.hemisphere).strip().upper(), int(row.base_roi_id))
            except (TypeError, ValueError):
                continue
            lookup[key] = str(row.node_name).strip()

        resolved: list[str] = []
        changed = False
        for label in labels:
            match = GENERIC_NODE_LABEL_RE.match(str(label).strip())
            if match is None:
                resolved.append(label)
                continue
            key = (match.group(1).upper(), int(match.group(2)))
            mapped = lookup.get(key, label)
            resolved.append(mapped)
            changed = changed or mapped != label

        if changed:
            return resolved

    return labels


def _load_metric_data(advanced_root: Path, metric: str) -> tuple[list[SessionMetric], list[str]]:
    session_metrics: list[SessionMetric] = []
    node_labels: list[str] | None = None
    matrix_shape: tuple[int, int] | None = None

    for label_dir in sorted(advanced_root.rglob("sub-*_ses-*")):
        parsed = _parse_label_session(label_dir.name)
        if parsed is None:
            continue
        subject, session = parsed
        metric_dir = label_dir / metric
        matrix_path = metric_dir / f"{metric}.npy"
        labels_path = metric_dir / f"{metric}_connectome.labels.txt"
        if not matrix_path.exists() or not labels_path.exists():
            continue

        matrix = np.asarray(np.load(matrix_path), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix in {matrix_path}, got {matrix.shape}")

        labels = _resolve_node_labels(
            advanced_root,
            labels_path.read_text(encoding="utf-8").strip().splitlines(),
        )
        if len(labels) != matrix.shape[0]:
            raise ValueError(
                f"Label count mismatch in {labels_path}: {len(labels)} labels for matrix {matrix.shape}"
            )

        if node_labels is None:
            node_labels = labels
            matrix_shape = matrix.shape
        else:
            if labels != node_labels:
                raise ValueError(f"Node-label mismatch detected in {labels_path}")
            if matrix.shape != matrix_shape:
                raise ValueError(
                    f"Matrix shape mismatch in {matrix_path}: {matrix.shape} vs {matrix_shape}"
                )

        session_metrics.append(
            SessionMetric(
                label=label_dir.name,
                subject=subject,
                session=session,
                matrix=matrix,
            )
        )

    if not session_metrics or node_labels is None:
        raise RuntimeError(f"No usable matrices found for metric '{metric}' in {advanced_root}")

    return session_metrics, node_labels


def _edge_name(node_labels: list[str], i: int, j: int) -> str:
    return f"{node_labels[i]} -- {node_labels[j]}"


def _display_edge_label(edge_label: str) -> str:
    parts = edge_label.split(" -- ")
    compact_parts: list[str] = []
    for part in parts:
        compact = part
        for old, new in NODE_LABEL_REPLACEMENTS:
            compact = compact.replace(old, new)
        compact = re.sub(r"\s+", " ", compact).strip()
        compact_parts.append(compact)
    return "\n".join(compact_parts)


def _rank_scores(delta_edges: np.ndarray, rank_by: str) -> np.ndarray:
    if rank_by == "mean_abs_delta":
        return np.mean(np.abs(delta_edges), axis=0)
    if rank_by == "abs_mean_delta":
        return np.abs(np.mean(delta_edges, axis=0))
    raise ValueError(f"Unsupported ranking: {rank_by}")


def _select_edge_order(
    rank_scores: np.ndarray,
    top_k: int,
    top_percentile: float,
) -> tuple[np.ndarray, dict[str, float | int | str | None]]:
    scores = np.asarray(rank_scores, dtype=np.float64)
    finite_mask = np.isfinite(scores)
    finite_indices = np.flatnonzero(finite_mask)
    if finite_indices.size == 0:
        raise RuntimeError("No finite edge rank scores were available for edge selection.")

    top_k = int(max(1, min(int(top_k), finite_indices.size)))
    sorted_finite_indices = finite_indices[np.argsort(scores[finite_indices])[::-1]]

    pct = float(np.clip(top_percentile, 0.0, 100.0))
    metadata: dict[str, float | int | str | None] = {
        "selection_mode": "top_k",
        "top_k_requested": int(top_k),
        "top_percentile": pct,
        "rank_score_threshold": None,
    }

    if pct < 100.0:
        threshold = float(np.percentile(scores[finite_mask], pct))
        selected = sorted_finite_indices[scores[sorted_finite_indices] >= threshold]
        if selected.size < top_k:
            selected = sorted_finite_indices[:top_k]
        metadata["selection_mode"] = "top_percentile"
        metadata["rank_score_threshold"] = threshold
    else:
        selected = sorted_finite_indices[:top_k]

    metadata["n_edges_selected"] = int(selected.size)
    return selected.astype(np.int64), metadata


def _plot_top_edge_heatmap(
    delta_matrix: np.ndarray,
    subject_labels: list[str],
    edge_labels: list[str],
    metric: str,
    out_png: Path,
    selection_label: str,
) -> None:
    def _plain_tick_label(value: float) -> str:
        scaled = float(value) * 1e7
        if abs(scaled) >= 100.0:
            text = f"{scaled:.0f}"
        elif abs(scaled) >= 10.0:
            text = f"{scaled:.1f}"
        elif abs(scaled) >= 1.0:
            text = f"{scaled:.2f}"
        else:
            text = f"{scaled:.3f}"
        return text.rstrip("0").rstrip(".")

    values = np.asarray(delta_matrix, dtype=np.float64)
    finite = values[np.isfinite(values)]
    abs_finite = np.abs(finite)
    if abs_finite.size == 0:
        vmax = 1e-6
        linthresh = 1e-8
    else:
        positive_abs = abs_finite[abs_finite > 0.0]
        vmax = float(np.percentile(abs_finite, 99.0))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = float(np.max(abs_finite))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1e-6

        if positive_abs.size == 0:
            linthresh = vmax * 1e-3
        else:
            # Use a robust lower scale so small, non-zero deltas remain visible without
            # letting a handful of extreme edges flatten the rest of the heatmap.
            linthresh = float(np.percentile(positive_abs, 25.0))
            linthresh = max(linthresh, vmax * 1e-6)
            linthresh = min(linthresh, vmax * 0.2)

    norm = colors.SymLogNorm(
        linthresh=linthresh,
        linscale=0.8,
        vmin=-vmax,
        vmax=vmax,
        base=10.0,
    )

    fig_width = max(14.0, 0.72 * len(edge_labels) + 3.0)
    fig_height = max(6.4, 0.45 * len(subject_labels) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#f2f2f2")
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(edge_labels), dtype=int))
    ax.set_yticks(np.arange(len(subject_labels), dtype=int))
    if len(edge_labels) <= 18:
        x_fontsize = 9
    elif len(edge_labels) <= 30:
        x_fontsize = 8
    elif len(edge_labels) <= 60:
        x_fontsize = 7
    elif len(edge_labels) <= 100:
        x_fontsize = 6
    else:
        x_fontsize = 5
    display_edge_labels = [_display_edge_label(label) for label in edge_labels]
    ax.set_xticklabels(
        display_edge_labels,
        rotation=52,
        ha="right",
        rotation_mode="anchor",
        fontsize=x_fontsize,
    )
    ax.set_yticklabels(subject_labels, fontsize=10)
    ax.tick_params(axis="x", pad=6)
    ax.tick_params(axis="y", pad=3)
    ax.set_ylabel("Subject", fontsize=12)
    ax.set_xticks(np.arange(-0.5, len(edge_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subject_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: _plain_tick_label(x))
    )
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Edge delta", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    advanced_root = args.advanced_root.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (advanced_root / args.metric)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    session_metrics, node_labels = _load_metric_data(advanced_root=advanced_root, metric=args.metric)
    rows, cols = _edge_rows_cols(len(node_labels))

    session1_by_subject = {
        item.subject: item for item in session_metrics if item.session == 1
    }
    session2_by_subject = {
        item.subject: item for item in session_metrics if item.session == 2
    }
    paired_subjects = sorted(set(session1_by_subject) & set(session2_by_subject))
    if not paired_subjects:
        raise RuntimeError(f"No paired session1/session2 subjects found for metric '{args.metric}'")

    session1_edges = []
    session2_edges = []
    for subject in paired_subjects:
        session1_edges.append(session1_by_subject[subject].matrix[rows, cols])
        session2_edges.append(session2_by_subject[subject].matrix[rows, cols])
    session1_edges = np.stack(session1_edges, axis=0)
    session2_edges = np.stack(session2_edges, axis=0)
    delta_edges = session2_edges - session1_edges

    rank_scores = _rank_scores(delta_edges=delta_edges, rank_by=str(args.rank_by))
    mean_abs_delta = np.mean(np.abs(delta_edges), axis=0)
    mean_signed_delta = np.mean(delta_edges, axis=0)
    std_delta = np.std(delta_edges, axis=0, ddof=1) if delta_edges.shape[0] > 1 else np.zeros(delta_edges.shape[1], dtype=np.float64)
    t_result = stats.ttest_1samp(delta_edges, popmean=0.0, axis=0, nan_policy="omit")
    t_stat = np.asarray(t_result.statistic, dtype=np.float64)
    p_value = np.asarray(t_result.pvalue, dtype=np.float64)

    top_k = int(max(1, min(int(args.top_k), delta_edges.shape[1])))
    order, selection_meta = _select_edge_order(
        rank_scores=rank_scores,
        top_k=top_k,
        top_percentile=float(args.top_percentile),
    )
    selected_edge_count = int(order.size)
    selection_label = (
        f"Top {selected_edge_count} edges (>= p{float(selection_meta['top_percentile']):.1f})"
        if selection_meta["selection_mode"] == "top_percentile"
        else f"Top {selected_edge_count} edges"
    )

    edge_summary_rows = []
    for rank, edge_idx in enumerate(order, start=1):
        i = int(rows[edge_idx])
        j = int(cols[edge_idx])
        edge_summary_rows.append(
            {
                "rank": rank,
                "edge_index": int(edge_idx),
                "node_i": i,
                "node_j": j,
                "edge": _edge_name(node_labels=node_labels, i=i, j=j),
                "rank_score": float(rank_scores[edge_idx]),
                "mean_abs_delta_session2_minus_session1": float(mean_abs_delta[edge_idx]),
                "mean_signed_delta_session2_minus_session1": float(mean_signed_delta[edge_idx]),
                "std_delta_session2_minus_session1": float(std_delta[edge_idx]),
                "t_stat_session2_minus_session1": float(t_stat[edge_idx]),
                "p_value_session2_minus_session1": float(p_value[edge_idx]),
            }
        )
    edge_summary_df = pd.DataFrame(edge_summary_rows)
    edge_summary_df.to_csv(out_dir / "top_edges_session2_minus_session1.csv", index=False)

    heatmap_matrix = delta_edges[:, order]
    heatmap_edge_labels = [_edge_name(node_labels=node_labels, i=int(rows[idx]), j=int(cols[idx])) for idx in order]
    heatmap_df = pd.DataFrame(heatmap_matrix, index=paired_subjects, columns=heatmap_edge_labels)
    heatmap_df.index.name = "subject"
    heatmap_df.to_csv(out_dir / "top_edges_subject_delta_heatmap.csv")

    _plot_top_edge_heatmap(
        delta_matrix=heatmap_matrix,
        subject_labels=paired_subjects,
        edge_labels=heatmap_edge_labels,
        metric=str(args.metric),
        out_png=out_dir / "top_edges_subject_delta_heatmap.png",
        selection_label=selection_label,
    )

    manifest = {
        "advanced_root": str(advanced_root),
        "metric": str(args.metric),
        "out_dir": str(out_dir),
        "top_k": top_k,
        "selection_mode": str(selection_meta["selection_mode"]),
        "top_percentile": float(selection_meta["top_percentile"]),
        "rank_score_threshold": (
            None
            if selection_meta["rank_score_threshold"] is None
            else float(selection_meta["rank_score_threshold"])
        ),
        "n_edges_selected": selected_edge_count,
        "rank_by": str(args.rank_by),
        "n_subjects_paired": int(len(paired_subjects)),
        "n_edges_total": int(delta_edges.shape[1]),
        "session_definition": {
            "session1": "OFF medication",
            "session2": "ON medication",
        },
    }
    (out_dir / "top_edges_session2_minus_session1_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"Saved edge summary to: {out_dir / 'top_edges_session2_minus_session1.csv'}", flush=True)
    print(
        f"Saved subject-by-edge delta matrix to: {out_dir / 'top_edges_subject_delta_heatmap.csv'}",
        flush=True,
    )
    print(f"Saved heatmap to: {out_dir / 'top_edges_subject_delta_heatmap.png'}", flush=True)


if __name__ == "__main__":
    main()
