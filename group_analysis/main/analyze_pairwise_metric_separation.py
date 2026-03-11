#!/usr/bin/env python3
"""Rank pairwise network-comparison metrics by medication-state separation."""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

LABEL_RE = re.compile(r"^(sub-[^_]+)_ses-(\d+)$")
SESSION_TO_STATE = {1: "off", 2: "on"}
DEFAULT_EXCLUDED_ROI_PATTERNS = ("brain stem", "brain-stem")
PAIR_LABEL_ORDER = ("off-off", "on-on", "off-on")
PAIR_CLASS_ORDER = ("between_condition", "within_condition")


@dataclass(frozen=True)
class SessionMatrix:
    label: str
    subject: str
    session: int
    state: str
    matrix: np.ndarray
    matrix_sym: np.ndarray
    edge_vector: np.ndarray
    edge_vector_sym: np.ndarray
    edge_distribution_signed: np.ndarray
    strength_vector: np.ndarray
    laplacian_spectrum_signed: np.ndarray
    mean_abs_offdiag_sym: float
    median_abs_offdiag_sym: float
    max_abs_offdiag_sym: float
    nonzero_fraction_offdiag_sym: float
    mean_node_strength_sym: float
    spectral_stability_ok: bool


@dataclass(frozen=True)
class ComparisonMetric:
    name: str
    kind: str
    higher_is_more_similar: bool
    fn: Callable[[SessionMatrix, SessionMatrix], float]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    advanced_root = script_dir
    parser = argparse.ArgumentParser(
        description=(
            "Compare saved subject-session connectivity matrices with multiple network "
            "similarity/distance measures and rank which ones best separate within-condition "
            "(OFF-OFF, ON-ON) from between-condition (ON-OFF) pairs."
        )
    )
    parser.add_argument(
        "--advanced-root",
        type=Path,
        default=advanced_root,
        help="Directory containing sub-*_ses-* metric folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=script_dir,
        help="Output directory for code-generated tables and figures.",
    )
    parser.add_argument(
        "--metrics",
        default="all",
        help="Comma-separated connectivity metrics to analyze or 'all'.",
    )
    parser.add_argument(
        "--comparison-metrics",
        default="all",
        help="Comma-separated pairwise comparison metrics to analyze or 'all'.",
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=None,
        help=(
            "Optional precomputed pairwise_metric_values.csv. When provided, reuse the saved "
            "pairwise scores instead of rebuilding them from per-session matrices."
        ),
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="Optional label-permutation count for a mean-difference permutation test.",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="How many top-ranked combinations to show in ranking plots.",
    )
    parser.add_argument(
        "--exclude-subjects",
        default="sub-pd017",
        help="Comma-separated subject IDs to exclude from all outputs.",
    )
    parser.add_argument(
        "--spectral-min-mean-abs-offdiag",
        type=float,
        default=1e-6,
        help=(
            "Flag spectral comparisons as unstable when a session matrix has mean |off-diagonal| "
            "below this value and also has no sufficiently large off-diagonal weights."
        ),
    )
    parser.add_argument(
        "--spectral-min-max-abs-offdiag",
        type=float,
        default=1e-4,
        help=(
            "Flag spectral comparisons as unstable when a session matrix max |off-diagonal| "
            "is below this value and the mean |off-diagonal| is also below the paired threshold."
        ),
    )
    parser.add_argument(
        "--skip-unstable-spectral-metrics",
        action="store_true",
        default=True,
        help="Skip Laplacian spectral distance when session matrices are numerically near-empty.",
    )
    parser.add_argument(
        "--allow-unstable-spectral-metrics",
        dest="skip_unstable_spectral_metrics",
        action="store_false",
        help="Do not skip Laplacian spectral distance even if spectral-stability diagnostics fail.",
    )
    return parser.parse_args()


def _parse_label_session(label: str) -> tuple[str, int] | None:
    match = LABEL_RE.match(label)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def _norm_roi_string(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", str(text).strip().lower())


def _build_exclude_patterns(raw_patterns: tuple[str, ...] | list[str]) -> list[str]:
    out: list[str] = []
    for pattern in raw_patterns:
        norm = _norm_roi_string(pattern)
        if norm and norm not in out:
            out.append(norm)
    return out


def _compute_keep_indices(
    node_labels: list[str],
    exclude_patterns: list[str],
) -> tuple[np.ndarray, list[str]]:
    if not exclude_patterns:
        return np.arange(len(node_labels), dtype=np.int64), []

    keep_idx: list[int] = []
    removed_labels: list[str] = []
    for idx, label in enumerate(node_labels):
        norm = _norm_roi_string(label)
        if any(pattern in norm for pattern in exclude_patterns):
            removed_labels.append(label)
            continue
        keep_idx.append(idx)
    return np.asarray(keep_idx, dtype=np.int64), removed_labels


def _edge_mask(shape: tuple[int, int], directed: bool) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = shape
    if n_rows != n_cols:
        raise ValueError(f"Expected square matrix, got {shape}")
    if directed:
        rows, cols = np.where(~np.eye(n_rows, dtype=bool))
    else:
        rows, cols = np.where(np.triu(np.ones((n_rows, n_rows), dtype=bool), k=1))
    return rows.astype(np.int64), cols.astype(np.int64)


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan")
    x = a[valid]
    y = b[valid]
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 1e-12 or y_std <= 1e-12:
        return float(np.allclose(x, y, equal_nan=True))
    return float(np.corrcoef(x, y)[0, 1])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 3:
        return float("nan")
    result = stats.spearmanr(a[valid], b[valid])
    return float(result.correlation)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan")
    x = a[valid]
    y = b[valid]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return float(np.allclose(x, y, equal_nan=True))
    return float(np.dot(x, y) / denom)


def _mean_abs_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(a[valid] - b[valid])))


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return float("nan")
    return float(np.linalg.norm(a[valid] - b[valid]))


def _frobenius_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.linalg.norm(diff, ord="fro"))


def _jensen_shannon_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.clip(np.asarray(a, dtype=np.float64).ravel(), 0.0, None)
    b = np.clip(np.asarray(b, dtype=np.float64).ravel(), 0.0, None)
    if a.size != b.size or a.size == 0:
        return float("nan")
    eps = 1e-12
    p = a + eps
    q = b + eps
    p /= float(np.sum(p))
    q /= float(np.sum(q))
    m = 0.5 * (p + q)
    js_div = 0.5 * float(stats.entropy(p, m) + stats.entropy(q, m))
    return float(np.sqrt(max(js_div, 0.0)))


def _signed_edge_distribution(edge_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(edge_weights, dtype=np.float64).ravel()
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.clip(weights, 0.0, None)
    neg = np.clip(-weights, 0.0, None)
    return np.concatenate([pos, neg], axis=0)


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
    eigvals = np.linalg.eigvalsh(laplacian)
    return np.sort(np.real(eigvals))


def _laplacian_spectral_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _rank_auc(within_scores: np.ndarray, between_scores: np.ndarray) -> float:
    x = np.asarray(within_scores, dtype=np.float64)
    y = np.asarray(between_scores, dtype=np.float64)
    ranks = stats.rankdata(np.concatenate([x, y]), method="average")
    n_x = x.size
    auc = (float(np.sum(ranks[:n_x])) - n_x * (n_x + 1) / 2.0) / float(n_x * y.size)
    return float(auc)


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    vx = float(np.var(x, ddof=1))
    vy = float(np.var(y, ddof=1))
    pooled_num = (x.size - 1) * vx + (y.size - 1) * vy
    pooled_den = x.size + y.size - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_sd = math.sqrt(max(pooled_num / pooled_den, 0.0))
    if pooled_sd <= 1e-12:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled_sd)


def _fdr_bh(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64).ravel()
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = float(ranked.size)
    adjusted = ranked * m / np.arange(1, ranked.size + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q[valid] = restored
    return q


def _display_subject_session_label(label: str) -> str:
    parsed = _parse_label_session(label)
    if parsed is None:
        return label
    subject, session = parsed
    return f"{subject}-ses{session}"


def _permutation_pvalue(
    within_scores: np.ndarray,
    between_scores: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> float:
    if permutations <= 0:
        return float("nan")
    x = np.asarray(within_scores, dtype=np.float64)
    y = np.asarray(between_scores, dtype=np.float64)
    observed = float(np.mean(x) - np.mean(y))
    combined = np.concatenate([x, y])
    n_x = x.size
    count = 0
    for _ in range(int(permutations)):
        perm = rng.permutation(combined.size)
        x_perm = combined[perm[:n_x]]
        y_perm = combined[perm[n_x:]]
        stat = float(np.mean(x_perm) - np.mean(y_perm))
        if abs(stat) >= abs(observed):
            count += 1
    return float((count + 1) / (permutations + 1))


def _discover_available_metrics(advanced_root: Path) -> list[str]:
    metrics: set[str] = set()
    for label_dir in sorted(advanced_root.glob("sub-*_ses-*")):
        if not label_dir.is_dir():
            continue
        for metric_dir in label_dir.iterdir():
            if not metric_dir.is_dir():
                continue
            matrix_path = metric_dir / f"{metric_dir.name}.npy"
            meta_path = metric_dir / f"{metric_dir.name}_meta.json"
            if matrix_path.exists() and meta_path.exists():
                metrics.add(metric_dir.name)
    return sorted(metrics)


def _normalize_selection(raw_value: str, available: list[str], field_name: str) -> list[str]:
    if not raw_value.strip() or raw_value.strip().lower() == "all":
        return available
    requested = [item.strip() for item in raw_value.split(",") if item.strip()]
    missing = [item for item in requested if item not in available]
    if missing:
        raise ValueError(f"Unknown {field_name}: {missing}. Available: {available}")
    out: list[str] = []
    seen: set[str] = set()
    for item in requested:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _normalize_subject_exclusions(raw_value: str) -> set[str]:
    return {item.strip() for item in str(raw_value).split(",") if item.strip()}


def _load_metric_sessions(
    advanced_root: Path,
    metric: str,
    excluded_subjects: set[str],
    spectral_min_mean_abs_offdiag: float,
    spectral_min_max_abs_offdiag: float,
) -> tuple[list[SessionMatrix], list[str], bool, list[str]]:
    label_dirs = [p for p in sorted(advanced_root.glob("sub-*_ses-*")) if p.is_dir()]
    if not label_dirs:
        raise RuntimeError(f"No subject/session directories found in {advanced_root}")

    sessions: list[SessionMatrix] = []
    node_labels: list[str] = []
    removed_labels: list[str] = []
    keep_idx: np.ndarray | None = None
    edge_rows: np.ndarray | None = None
    edge_cols: np.ndarray | None = None
    directed = False
    matrix_shape: tuple[int, int] | None = None
    exclude_patterns = _build_exclude_patterns(DEFAULT_EXCLUDED_ROI_PATTERNS)

    for label_dir in label_dirs:
        parsed = _parse_label_session(label_dir.name)
        if parsed is None:
            continue
        subject, session = parsed
        if subject in excluded_subjects:
            continue
        state = SESSION_TO_STATE.get(session)
        if state is None:
            continue

        metric_dir = label_dir / metric
        matrix_path = metric_dir / f"{metric}.npy"
        meta_path = metric_dir / f"{metric}_meta.json"
        labels_path = metric_dir / f"{metric}_connectome.labels.txt"
        if not matrix_path.exists() or not meta_path.exists():
            continue

        matrix = np.asarray(np.load(matrix_path), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            continue
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if matrix_shape is None:
            if labels_path.exists():
                node_labels = labels_path.read_text(encoding="utf-8").strip().splitlines()
            if len(node_labels) != matrix.shape[0]:
                node_labels = [f"Node_{idx + 1:03d}" for idx in range(matrix.shape[0])]
            keep_idx, removed_labels = _compute_keep_indices(node_labels, exclude_patterns)
            node_labels = [node_labels[int(idx)] for idx in keep_idx]
            directed = bool(meta.get("directed", False))
            matrix = matrix[np.ix_(keep_idx, keep_idx)]
            matrix_shape = (int(matrix.shape[0]), int(matrix.shape[1]))
            edge_rows, edge_cols = _edge_mask(matrix_shape, directed=directed)
        else:
            if keep_idx is None or edge_rows is None or edge_cols is None:
                raise RuntimeError("Metric loading state was not initialized correctly.")
            matrix = matrix[np.ix_(keep_idx, keep_idx)]
            if matrix.shape != matrix_shape:
                raise RuntimeError(
                    f"Metric {metric} matrix shape mismatch for {label_dir.name}: "
                    f"{matrix.shape} vs {matrix_shape}"
                )

        np.fill_diagonal(matrix, 0.0)
        matrix_sym = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix_sym, 0.0)
        edge_vector_sym = matrix_sym[edge_rows, edge_cols].astype(np.float64, copy=False)
        edge_distribution_signed = _signed_edge_distribution(edge_vector_sym)
        strength_vector = np.sum(matrix_sym, axis=1)
        abs_offdiag = np.abs(edge_vector_sym)
        mean_abs_offdiag = float(np.mean(abs_offdiag)) if abs_offdiag.size else 0.0
        median_abs_offdiag = float(np.median(abs_offdiag)) if abs_offdiag.size else 0.0
        max_abs_offdiag = float(np.max(abs_offdiag)) if abs_offdiag.size else 0.0
        nonzero_fraction = (
            float(np.count_nonzero(abs_offdiag > 0.0)) / float(abs_offdiag.size)
            if abs_offdiag.size
            else 0.0
        )
        mean_node_strength = float(np.mean(np.abs(strength_vector))) if strength_vector.size else 0.0
        spectral_stability_ok = not (
            (mean_abs_offdiag < float(spectral_min_mean_abs_offdiag))
            and (max_abs_offdiag < float(spectral_min_max_abs_offdiag))
        )
        laplacian_spectrum_signed = _signed_normalized_laplacian_spectrum(matrix_sym)
        sessions.append(
            SessionMatrix(
                label=label_dir.name,
                subject=subject,
                session=session,
                state=state,
                matrix=matrix,
                matrix_sym=matrix_sym,
                edge_vector=matrix[edge_rows, edge_cols].astype(np.float64, copy=False),
                edge_vector_sym=edge_vector_sym,
                edge_distribution_signed=edge_distribution_signed.astype(np.float64, copy=False),
                strength_vector=strength_vector.astype(np.float64, copy=False),
                laplacian_spectrum_signed=laplacian_spectrum_signed.astype(np.float64, copy=False),
                mean_abs_offdiag_sym=mean_abs_offdiag,
                median_abs_offdiag_sym=median_abs_offdiag,
                max_abs_offdiag_sym=max_abs_offdiag,
                nonzero_fraction_offdiag_sym=nonzero_fraction,
                mean_node_strength_sym=mean_node_strength,
                spectral_stability_ok=bool(spectral_stability_ok),
            )
        )

    if not sessions:
        raise RuntimeError(f"No usable session matrices found for {metric}")
    return sorted(sessions, key=lambda item: item.label), node_labels, directed, removed_labels


def _spectral_diagnostic_rows(metric_name: str, sessions: list[SessionMatrix]) -> pd.DataFrame:
    rows = []
    for session in sessions:
        rows.append(
            {
                "connectivity_metric": metric_name,
                "label": session.label,
                "subject": session.subject,
                "session": int(session.session),
                "state": session.state,
                "mean_abs_offdiag_sym": float(session.mean_abs_offdiag_sym),
                "median_abs_offdiag_sym": float(session.median_abs_offdiag_sym),
                "max_abs_offdiag_sym": float(session.max_abs_offdiag_sym),
                "nonzero_fraction_offdiag_sym": float(session.nonzero_fraction_offdiag_sym),
                "mean_node_strength_sym": float(session.mean_node_strength_sym),
                "spectral_stability_ok": bool(session.spectral_stability_ok),
            }
        )
    return pd.DataFrame(rows)


def _comparison_metric_specs() -> list[ComparisonMetric]:
    return [
        ComparisonMetric(
            name="edge_pearson_similarity",
            kind="edge_vector",
            higher_is_more_similar=True,
            fn=lambda a, b: _safe_pearson(a.edge_vector, b.edge_vector),
        ),
        ComparisonMetric(
            name="edge_spearman_similarity",
            kind="edge_vector",
            higher_is_more_similar=True,
            fn=lambda a, b: _safe_spearman(a.edge_vector, b.edge_vector),
        ),
        ComparisonMetric(
            name="edge_cosine_similarity",
            kind="edge_vector",
            higher_is_more_similar=True,
            fn=lambda a, b: _cosine_similarity(a.edge_vector, b.edge_vector),
        ),
        ComparisonMetric(
            name="edge_mean_abs_distance",
            kind="edge_vector",
            higher_is_more_similar=False,
            fn=lambda a, b: _mean_abs_distance(a.edge_vector, b.edge_vector),
        ),
        ComparisonMetric(
            name="edge_euclidean_distance",
            kind="edge_vector",
            higher_is_more_similar=False,
            fn=lambda a, b: _euclidean_distance(a.edge_vector, b.edge_vector),
        ),
        ComparisonMetric(
            name="frobenius_distance",
            kind="matrix",
            higher_is_more_similar=False,
            fn=lambda a, b: _frobenius_distance(a.matrix, b.matrix),
        ),
        ComparisonMetric(
            name="edge_js_distance_signed",
            kind="edge_distribution",
            higher_is_more_similar=False,
            fn=lambda a, b: _jensen_shannon_distance(
                a.edge_distribution_signed,
                b.edge_distribution_signed,
            ),
        ),
        ComparisonMetric(
            name="node_strength_cosine_similarity_signed",
            kind="graph_profile",
            higher_is_more_similar=True,
            fn=lambda a, b: _cosine_similarity(a.strength_vector, b.strength_vector),
        ),
        ComparisonMetric(
            name="node_strength_spearman_similarity_signed",
            kind="graph_profile",
            higher_is_more_similar=True,
            fn=lambda a, b: _safe_spearman(a.strength_vector, b.strength_vector),
        ),
        ComparisonMetric(
            name="laplacian_spectral_distance_signed",
            kind="graph_distance",
            higher_is_more_similar=False,
            fn=lambda a, b: _laplacian_spectral_distance(
                a.laplacian_spectrum_signed,
                b.laplacian_spectrum_signed,
            ),
        ),
    ]


def _build_pairwise_table(
    metric_name: str,
    sessions: list[SessionMatrix],
    comparison_metrics: list[ComparisonMetric],
) -> pd.DataFrame:
    if not comparison_metrics:
        return pd.DataFrame(
            columns=[
                "connectivity_metric",
                "label_a",
                "label_b",
                "subject_a",
                "subject_b",
                "session_a",
                "session_b",
                "state_a",
                "state_b",
                "pair_class",
                "pair_label",
                "same_subject",
                "comparison_metric",
                "comparison_kind",
                "higher_is_more_similar",
                "raw_score",
                "oriented_score",
            ]
        )
    rows: list[dict] = []
    for idx_a, sess_a in enumerate(sessions):
        for idx_b in range(idx_a + 1, len(sessions)):
            sess_b = sessions[idx_b]
            pair_class = "within_condition" if sess_a.state == sess_b.state else "between_condition"
            pair_label = (
                f"{sess_a.state}-{sess_b.state}"
                if sess_a.state <= sess_b.state
                else f"{sess_b.state}-{sess_a.state}"
            )
            same_subject = sess_a.subject == sess_b.subject
            base = {
                "connectivity_metric": metric_name,
                "label_a": sess_a.label,
                "label_b": sess_b.label,
                "subject_a": sess_a.subject,
                "subject_b": sess_b.subject,
                "session_a": int(sess_a.session),
                "session_b": int(sess_b.session),
                "state_a": sess_a.state,
                "state_b": sess_b.state,
                "pair_class": pair_class,
                "pair_label": pair_label,
                "same_subject": bool(same_subject),
            }
            for spec in comparison_metrics:
                raw_score = float(spec.fn(sess_a, sess_b))
                oriented_score = raw_score if spec.higher_is_more_similar else -raw_score
                row = dict(base)
                row.update(
                    {
                        "comparison_metric": spec.name,
                        "comparison_kind": spec.kind,
                        "higher_is_more_similar": bool(spec.higher_is_more_similar),
                        "raw_score": raw_score,
                        "oriented_score": oriented_score,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _save_square_matrix_csv(matrix: np.ndarray, labels: list[str], out_csv: Path) -> None:
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_csv)


def _plot_pairwise_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    out_png: Path,
    title: str,
    cbar_label: str,
) -> None:
    vals = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    finite = vals[np.isfinite(vals)]
    vmin = float(np.min(finite)) if finite.size else 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.imshow(vals, cmap="jet", vmin=vmin, vmax=vmax, aspect="auto")
    ticks = np.arange(len(labels), dtype=int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _save_pairwise_metric_matrix(
    pairwise_df: pd.DataFrame,
    connectivity_metric: str,
    comparison_metric: str,
    out_dir: Path,
) -> None:
    subset = pairwise_df.loc[
        (pairwise_df["connectivity_metric"] == connectivity_metric)
        & (pairwise_df["comparison_metric"] == comparison_metric)
    ].copy()
    if subset.empty:
        return

    labels_raw = sorted(
        set(subset["label_a"].astype(str).tolist()) | set(subset["label_b"].astype(str).tolist())
    )
    display_labels = [_display_subject_session_label(label) for label in labels_raw]
    label_to_idx = {label: idx for idx, label in enumerate(labels_raw)}
    matrix = np.zeros((len(labels_raw), len(labels_raw)), dtype=np.float64)

    for row in subset.itertuples(index=False):
        idx_a = label_to_idx[str(row.label_a)]
        idx_b = label_to_idx[str(row.label_b)]
        value = float(row.raw_score)
        matrix[idx_a, idx_b] = value
        matrix[idx_b, idx_a] = value

    metric_out = out_dir / connectivity_metric
    metric_out.mkdir(parents=True, exist_ok=True)
    stem = f"all_subject_session_pairwise_{comparison_metric}"
    _save_square_matrix_csv(matrix, display_labels, metric_out / f"{stem}.csv")
    _plot_pairwise_heatmap(
        matrix,
        display_labels,
        metric_out / f"{stem}.png",
        title=f"{connectivity_metric} | All subject-session {comparison_metric}",
        cbar_label=comparison_metric,
    )


def _summarize_pair_counts(pair_df: pd.DataFrame) -> dict[str, int]:
    return {
        "n_pairs_total": int(len(pair_df)),
        "n_pairs_within_condition": int((pair_df["pair_class"] == "within_condition").sum()),
        "n_pairs_between_condition": int((pair_df["pair_class"] == "between_condition").sum()),
        "n_pairs_same_subject": int(pair_df["same_subject"].sum()),
        "n_pairs_cross_subject": int((~pair_df["same_subject"]).sum()),
        "n_pairs_off_off": int((pair_df["pair_label"] == "off-off").sum()),
        "n_pairs_on_on": int((pair_df["pair_label"] == "on-on").sum()),
        "n_pairs_off_on": int((pair_df["pair_label"] == "off-on").sum()),
    }


def _summarize_comparison_group(
    group_df: pd.DataFrame,
    cohort_name: str,
) -> dict:
    within = group_df.loc[group_df["pair_class"] == "within_condition", "oriented_score"].to_numpy(dtype=np.float64)
    between = group_df.loc[group_df["pair_class"] == "between_condition", "oriented_score"].to_numpy(dtype=np.float64)
    raw_within = group_df.loc[group_df["pair_class"] == "within_condition", "raw_score"].to_numpy(dtype=np.float64)
    raw_between = group_df.loc[group_df["pair_class"] == "between_condition", "raw_score"].to_numpy(dtype=np.float64)
    off_off = group_df.loc[group_df["pair_label"] == "off-off", "raw_score"].to_numpy(dtype=np.float64)
    on_on = group_df.loc[group_df["pair_label"] == "on-on", "raw_score"].to_numpy(dtype=np.float64)
    off_on = group_df.loc[group_df["pair_label"] == "off-on", "raw_score"].to_numpy(dtype=np.float64)
    between_same_subject = group_df.loc[
        (group_df["pair_class"] == "between_condition") & (group_df["same_subject"]),
        "raw_score",
    ].to_numpy(dtype=np.float64)
    between_cross_subject = group_df.loc[
        (group_df["pair_class"] == "between_condition") & (~group_df["same_subject"]),
        "raw_score",
    ].to_numpy(dtype=np.float64)

    if within.size == 0 or between.size == 0:
        raise RuntimeError(f"Cohort '{cohort_name}' does not contain both comparison groups.")

    model_df = group_df.loc[
        :,
        ["oriented_score", "pair_class", "subject_a", "subject_b"],
    ].copy()
    model_df["pair_class"] = pd.Categorical(
        model_df["pair_class"],
        categories=list(PAIR_CLASS_ORDER),
        ordered=True,
    )
    class_coef_name = "C(pair_class, Treatment(reference='between_condition'))[T.within_condition]"
    fit, fit_info = _fit_crossed_subject_mixedlm(
        model_df=model_df,
        formula="oriented_score ~ C(pair_class, Treatment(reference='between_condition'))",
        value_col="oriented_score",
    )
    lme_stats = _mixedlm_contrast(fit, {class_coef_name: 1.0})
    auc = _rank_auc(within, between)
    mean_gap = float(np.mean(within) - np.mean(between))
    median_gap = float(np.median(within) - np.median(between))
    return {
        "cohort": cohort_name,
        "n_within_condition": int(within.size),
        "n_between_condition": int(between.size),
        "n_off_off": int(off_off.size),
        "n_on_on": int(on_on.size),
        "n_off_on": int(off_on.size),
        "n_between_same_subject": int(between_same_subject.size),
        "n_between_cross_subject": int(between_cross_subject.size),
        "within_mean_raw": float(np.mean(raw_within)),
        "within_median_raw": float(np.median(raw_within)),
        "between_mean_raw": float(np.mean(raw_between)),
        "between_median_raw": float(np.median(raw_between)),
        "off_off_mean_raw": float(np.mean(off_off)) if off_off.size else float("nan"),
        "on_on_mean_raw": float(np.mean(on_on)) if on_on.size else float("nan"),
        "off_on_mean_raw": float(np.mean(off_on)) if off_on.size else float("nan"),
        "between_same_subject_mean_raw": float(np.mean(between_same_subject))
        if between_same_subject.size
        else float("nan"),
        "between_cross_subject_mean_raw": float(np.mean(between_cross_subject))
        if between_cross_subject.size
        else float("nan"),
        "within_mean_oriented": float(np.mean(within)),
        "between_mean_oriented": float(np.mean(between)),
        "mean_gap_oriented": mean_gap,
        "median_gap_oriented": median_gap,
        "auc_within_gt_between": float(auc),
        "cliffs_delta_oriented": float(2.0 * auc - 1.0),
        "cohen_d_oriented": _cohen_d(within, between),
        "lme_coef_within_minus_between": float(lme_stats["estimate"]),
        "lme_se_within_minus_between": float(lme_stats["se"]),
        "lme_z_within_minus_between": float(lme_stats["z_value"]),
        "lme_p_two_sided": float(lme_stats["p_value_two_sided"]),
        "lme_n_obs": int(fit_info["n_obs"]),
        "lme_n_subjects": int(fit_info["n_subjects"]),
        "lme_fit_method": fit_info["fit_method"],
        "lme_converged": bool(fit_info["model_converged"]),
    }


def _build_summary_table(
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows: list[dict] = []
    group_cols = ["connectivity_metric", "comparison_metric"]
    for (connectivity_metric, comparison_metric), group_df in pairwise_df.groupby(group_cols, sort=True):
        higher_is_more_similar = bool(group_df["higher_is_more_similar"].iloc[0])
        comparison_kind = str(group_df["comparison_kind"].iloc[0])
        for cohort_name, cohort_df in (
            ("all_pairs", group_df),
            ("cross_subject_only", group_df.loc[~group_df["same_subject"]].copy()),
        ):
            row = _summarize_comparison_group(cohort_df, cohort_name=cohort_name)
            row.update(
                {
                    "connectivity_metric": connectivity_metric,
                    "comparison_metric": comparison_metric,
                    "comparison_kind": comparison_kind,
                    "higher_is_more_similar": higher_is_more_similar,
                }
            )
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    for cohort_name in sorted(summary_df["cohort"].unique()):
        mask = summary_df["cohort"] == cohort_name
        two_sided_vals = summary_df.loc[mask, "lme_p_two_sided"].to_numpy(dtype=np.float64)
        summary_df.loc[mask, "lme_q_two_sided_fdr_bh"] = _fdr_bh(two_sided_vals)

    summary_df = summary_df.sort_values(
        ["cohort", "auc_within_gt_between", "cohen_d_oriented"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return summary_df


def _save_heatmap(
    summary_df: pd.DataFrame,
    cohort_name: str,
    value_column: str,
    out_png: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    center: float | None = None,
) -> None:
    cohort_df = summary_df.loc[summary_df["cohort"] == cohort_name].copy()
    if cohort_df.empty:
        return
    pivot = cohort_df.pivot(
        index="connectivity_metric",
        columns="comparison_metric",
        values=value_column,
    )
    if pivot.empty:
        return
    values = pivot.to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(9.0, 0.8 * values.shape[1] + 3.5), 5.5))
    if center is None:
        finite = values[np.isfinite(values)]
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
    else:
        finite = np.abs(values[np.isfinite(values)])
        vmax = float(np.max(finite)) if finite.size else 1.0
        vmin = -vmax
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_title(title, fontsize=11)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label)
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _save_top_barplot(summary_df: pd.DataFrame, cohort_name: str, top_k: int, out_png: Path) -> None:
    cohort_df = summary_df.loc[summary_df["cohort"] == cohort_name].copy()
    if cohort_df.empty:
        return
    top_df = cohort_df.nlargest(int(top_k), "auc_within_gt_between").copy()
    labels = [
        f"{row.connectivity_metric}\n{row.comparison_metric}"
        for row in top_df.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(max(10.0, 0.85 * len(top_df)), 6.2))
    ax.bar(np.arange(len(top_df)), top_df["auc_within_gt_between"], color="#2c7fb8")
    ax.axhline(0.5, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(np.arange(len(top_df)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("AUC (within > between)")
    ax.set_title(f"Top {len(top_df)} separation scores: {cohort_name}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)


def _pvalue_to_stars(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n/a"
    if p_value < 1e-4:
        return "****"
    if p_value < 1e-3:
        return "***"
    if p_value < 1e-2:
        return "**"
    if p_value < 5e-2:
        return "*"
    return "ns"


def _fit_crossed_subject_mixedlm(
    model_df: pd.DataFrame,
    formula: str,
    value_col: str,
) -> tuple[object | None, dict[str, object]]:
    required_cols = {value_col, "subject_a", "subject_b"}
    missing = required_cols.difference(model_df.columns)
    if missing:
        raise ValueError(f"Missing required mixed-model columns: {sorted(missing)}")

    fit_df = model_df.copy()
    fit_df = fit_df.loc[np.isfinite(fit_df[value_col].to_numpy(dtype=np.float64))].reset_index(drop=True)
    if fit_df.empty:
        return None, {"n_obs": 0, "n_subjects": 0, "fit_method": None, "model_converged": False}

    fit_df["subject_a"] = fit_df["subject_a"].astype(str)
    fit_df["subject_b"] = fit_df["subject_b"].astype(str)
    fit_df["_group"] = "all_pairs"

    n_subjects = int(pd.unique(pd.concat([fit_df["subject_a"], fit_df["subject_b"]], axis=0)).size)
    base_info = {
        "n_obs": int(fit_df.shape[0]),
        "n_subjects": n_subjects,
        "fit_method": None,
        "model_converged": False,
    }
    if fit_df.shape[0] < 3 or n_subjects < 2:
        return None, base_info

    best_fit = None
    best_method = None
    best_llf = -np.inf
    methods = ("lbfgs", "powell", "bfgs", "cg", "nm")
    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    formula,
                    data=fit_df,
                    groups=fit_df["_group"],
                    re_formula="0",
                    vc_formula={
                        "subject_a": "0 + C(subject_a)",
                        "subject_b": "0 + C(subject_b)",
                    },
                ).fit(reml=False, method=method, disp=False)
        except Exception:
            continue

        llf = float(getattr(fit, "llf", float("-inf")))
        if bool(getattr(fit, "converged", False)):
            return fit, {
                **base_info,
                "fit_method": method,
                "model_converged": True,
            }
        if llf > best_llf:
            best_fit = fit
            best_method = method
            best_llf = llf

    if best_fit is None:
        return None, base_info
    return best_fit, {
        **base_info,
        "fit_method": best_method,
        "model_converged": bool(getattr(best_fit, "converged", False)),
    }


def _mixedlm_contrast(
    fit: object | None,
    weights: dict[str, float],
) -> dict[str, float]:
    row = {
        "estimate": float("nan"),
        "se": float("nan"),
        "z_value": float("nan"),
        "p_value_two_sided": float("nan"),
    }
    if fit is None:
        return row

    fe_params = getattr(fit, "fe_params", None)
    cov_params = getattr(fit, "cov_params", None)
    if fe_params is None or cov_params is None:
        return row

    fe_names = list(fe_params.index)
    if not fe_names:
        return row
    cov_fe = cov_params().loc[fe_names, fe_names].to_numpy(dtype=np.float64)
    beta = fe_params.to_numpy(dtype=np.float64)

    contrast = np.zeros(len(fe_names), dtype=np.float64)
    for idx, name in enumerate(fe_names):
        contrast[idx] = float(weights.get(name, 0.0))

    estimate = float(np.dot(contrast, beta))
    variance = float(np.dot(contrast, cov_fe @ contrast))
    variance = max(variance, 0.0)
    se = math.sqrt(variance)
    if se <= 1e-12:
        z_value = float("nan")
        p_value = float("nan")
    else:
        z_value = float(estimate / se)
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))

    row.update(
        {
            "estimate": estimate,
            "se": float(se),
            "z_value": z_value,
            "p_value_two_sided": p_value,
        }
    )
    return row


def _add_significance_bar(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1.0)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom", fontsize=9)


def _save_distribution_plot(
    pairwise_df: pd.DataFrame,
    connectivity_metric: str,
    comparison_metric: str,
    cohort_name: str,
    out_png: Path,
) -> pd.DataFrame:
    subset = pairwise_df.loc[
        (pairwise_df["connectivity_metric"] == connectivity_metric)
        & (pairwise_df["comparison_metric"] == comparison_metric)
    ].copy()
    if cohort_name == "cross_subject_only":
        subset = subset.loc[~subset["same_subject"]].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["pair_label"] = pd.Categorical(
        subset["pair_label"],
        categories=list(PAIR_LABEL_ORDER),
        ordered=True,
    )
    groups = [
        ("OFF-OFF", subset.loc[subset["pair_label"] == "off-off", "raw_score"].to_numpy(dtype=np.float64)),
        ("ON-ON", subset.loc[subset["pair_label"] == "on-on", "raw_score"].to_numpy(dtype=np.float64)),
        ("ON-OFF", subset.loc[subset["pair_label"] == "off-on", "raw_score"].to_numpy(dtype=np.float64)),
    ]
    group_lookup = {name: values for name, values in groups}
    plot_data = [values for _name, values in groups if values.size]
    plot_labels = [name for name, values in groups if values.size]
    if not plot_data:
        return pd.DataFrame()

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    box = ax.boxplot(plot_data, patch_artist=True, tick_labels=plot_labels)
    colors = ["#4c78a8", "#59a14f", "#e15759"]
    for patch, color in zip(box["boxes"], colors[: len(box["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for idx, values in enumerate(plot_data, start=1):
        x = rng.normal(loc=idx, scale=0.04, size=values.size)
        ax.scatter(x, values, s=14, alpha=0.55, color="black", linewidths=0.0)
    ax.set_title(f"{connectivity_metric} | {comparison_metric} | {cohort_name}")
    ax.set_ylabel("Raw pairwise score")

    y_max = max(float(np.max(values)) for values in plot_data if values.size)
    y_min = min(float(np.min(values)) for values in plot_data if values.size)
    y_span = max(y_max - y_min, 1e-6)
    h = 0.04 * y_span
    current_y = y_max + 0.08 * y_span
    label_to_x = {label: idx for idx, label in enumerate(plot_labels, start=1)}
    fit, fit_info = _fit_crossed_subject_mixedlm(
        model_df=subset.loc[:, ["raw_score", "pair_label", "subject_a", "subject_b"]].copy(),
        formula="raw_score ~ C(pair_label, Treatment(reference='off-off'))",
        value_col="raw_score",
    )
    pair_coef_names = {
        "on-on": "C(pair_label, Treatment(reference='off-off'))[T.on-on]",
        "off-on": "C(pair_label, Treatment(reference='off-off'))[T.off-on]",
    }
    comparisons = [
        ("OFF-OFF", "ON-ON"),
        ("OFF-OFF", "ON-OFF"),
        ("ON-ON", "ON-OFF"),
    ]
    stats_rows: list[dict] = []
    for left_name, right_name in comparisons:
        left = group_lookup.get(left_name, np.asarray([], dtype=np.float64))
        right = group_lookup.get(right_name, np.asarray([], dtype=np.float64))
        left_x = label_to_x.get(left_name)
        right_x = label_to_x.get(right_name)
        if left.size == 0 or right.size == 0 or left_x is None or right_x is None:
            continue
        if left_name == "OFF-OFF" and right_name == "ON-ON":
            contrast = _mixedlm_contrast(fit, {pair_coef_names["on-on"]: 1.0})
        elif left_name == "OFF-OFF" and right_name == "ON-OFF":
            contrast = _mixedlm_contrast(fit, {pair_coef_names["off-on"]: 1.0})
        elif left_name == "ON-ON" and right_name == "ON-OFF":
            contrast = _mixedlm_contrast(
                fit,
                {
                    pair_coef_names["off-on"]: 1.0,
                    pair_coef_names["on-on"]: -1.0,
                },
            )
        else:
            contrast = _mixedlm_contrast(fit, {})
        p_value = float(contrast["p_value_two_sided"])
        stars = _pvalue_to_stars(p_value)
        label = f"{stars} (p={p_value:.2g})"
        _add_significance_bar(ax, left_x, right_x, current_y, h, label)
        stats_rows.append(
            {
                "connectivity_metric": connectivity_metric,
                "comparison_metric": comparison_metric,
                "cohort": cohort_name,
                "group_a": left_name,
                "group_b": right_name,
                "n_group_a": int(left.size),
                "n_group_b": int(right.size),
                "mean_group_a": float(np.mean(left)),
                "mean_group_b": float(np.mean(right)),
                "test": "mixedlm_crossed_subject_intercepts",
                "estimate_group_b_minus_group_a": float(contrast["estimate"]),
                "lme_se": float(contrast["se"]),
                "lme_z": float(contrast["z_value"]),
                "p_value_two_sided": p_value,
                "significance_stars": stars,
                "lme_n_obs": int(fit_info["n_obs"]),
                "lme_n_subjects": int(fit_info["n_subjects"]),
                "lme_fit_method": fit_info["fit_method"],
                "lme_converged": bool(fit_info["model_converged"]),
            }
        )
        current_y += 0.12 * y_span

    ax.set_ylim(y_min - 0.05 * y_span, current_y + 0.08 * y_span)
    fig.tight_layout()
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def _edge_variance_rows(
    metric_name: str,
    sessions: list[SessionMatrix],
    node_labels: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_vectors: dict[str, list[np.ndarray]] = {"off": [], "on": []}
    for session in sessions:
        state_vectors[session.state].append(session.edge_vector)

    summary_row = {
        "connectivity_metric": metric_name,
        "n_off_sessions": len(state_vectors["off"]),
        "n_on_sessions": len(state_vectors["on"]),
    }

    detailed_rows: list[dict] = []
    if not state_vectors["off"] or not state_vectors["on"]:
        return pd.DataFrame([summary_row]), pd.DataFrame(detailed_rows)

    off_stack = np.vstack(state_vectors["off"])
    on_stack = np.vstack(state_vectors["on"])
    off_var = np.var(off_stack, axis=0, ddof=1)
    on_var = np.var(on_stack, axis=0, ddof=1)
    delta_var = on_var - off_var

    summary_row.update(
        {
            "mean_edge_variance_off": float(np.mean(off_var)),
            "median_edge_variance_off": float(np.median(off_var)),
            "mean_edge_variance_on": float(np.mean(on_var)),
            "median_edge_variance_on": float(np.median(on_var)),
            "mean_variance_delta_on_minus_off": float(np.mean(delta_var)),
            "median_variance_delta_on_minus_off": float(np.median(delta_var)),
            "variance_profile_spearman": _safe_spearman(off_var, on_var),
            "variance_profile_pearson": _safe_pearson(off_var, on_var),
        }
    )

    edge_counter = 0
    for row_idx in range(len(node_labels)):
        for col_idx in range(row_idx + 1, len(node_labels)):
            detailed_rows.append(
                {
                    "connectivity_metric": metric_name,
                    "edge_index": edge_counter,
                    "node_a": node_labels[row_idx],
                    "node_b": node_labels[col_idx],
                    "variance_off": float(off_var[edge_counter]),
                    "variance_on": float(on_var[edge_counter]),
                    "variance_delta_on_minus_off": float(delta_var[edge_counter]),
                    "abs_variance_delta": float(abs(delta_var[edge_counter])),
                }
            )
            edge_counter += 1
    return pd.DataFrame([summary_row]), pd.DataFrame(detailed_rows)


def main() -> None:
    args = parse_args()
    advanced_root = args.advanced_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    excluded_subjects = _normalize_subject_exclusions(args.exclude_subjects)

    comparison_specs_all = _comparison_metric_specs()
    comparison_names_all = [spec.name for spec in comparison_specs_all]
    selected_comparison_names = _normalize_selection(
        args.comparison_metrics,
        comparison_names_all,
        "comparison metrics",
    )
    comparison_specs = [spec for spec in comparison_specs_all if spec.name in selected_comparison_names]
    distribution_stats_tables: list[pd.DataFrame] = []
    pairwise_csv = args.pairwise_csv.resolve() if args.pairwise_csv is not None else None
    pairwise_input_mode = pairwise_csv is not None

    if pairwise_input_mode:
        if not pairwise_csv.exists():
            raise FileNotFoundError(f"Pairwise CSV not found: {pairwise_csv}")
        pairwise_df = pd.read_csv(pairwise_csv)
        required_cols = {
            "connectivity_metric",
            "subject_a",
            "subject_b",
            "pair_label",
            "pair_class",
            "comparison_metric",
            "comparison_kind",
            "higher_is_more_similar",
            "raw_score",
            "oriented_score",
            "same_subject",
        }
        missing_cols = required_cols.difference(pairwise_df.columns)
        if missing_cols:
            raise ValueError(f"Pairwise CSV is missing required columns: {sorted(missing_cols)}")
        if excluded_subjects:
            pairwise_df = pairwise_df.loc[
                (~pairwise_df["subject_a"].astype(str).isin(excluded_subjects))
                & (~pairwise_df["subject_b"].astype(str).isin(excluded_subjects))
            ].copy()
        available_metrics = sorted(pairwise_df["connectivity_metric"].astype(str).unique().tolist())
        selected_metrics = _normalize_selection(args.metrics, available_metrics, "metrics")
        pairwise_df = pairwise_df.loc[
            pairwise_df["connectivity_metric"].astype(str).isin(selected_metrics)
            & pairwise_df["comparison_metric"].astype(str).isin(selected_comparison_names)
        ].copy()
        pairwise_df["higher_is_more_similar"] = pairwise_df["higher_is_more_similar"].astype(bool)
        pairwise_df["same_subject"] = pairwise_df["same_subject"].astype(bool)
        skipped_metric_rows: list[dict] = []
    else:
        available_metrics = _discover_available_metrics(advanced_root)
        selected_metrics = _normalize_selection(args.metrics, available_metrics, "metrics")
        pairwise_tables: list[pd.DataFrame] = []
        metric_inventory_rows: list[dict] = []
        edge_variance_summary_tables: list[pd.DataFrame] = []
        edge_variance_detail_tables: list[pd.DataFrame] = []
        spectral_diagnostic_tables: list[pd.DataFrame] = []
        skipped_metric_rows: list[dict] = []

        for metric_name in selected_metrics:
            sessions, node_labels, directed, removed_labels = _load_metric_sessions(
                advanced_root,
                metric_name,
                excluded_subjects=excluded_subjects,
                spectral_min_mean_abs_offdiag=float(args.spectral_min_mean_abs_offdiag),
                spectral_min_max_abs_offdiag=float(args.spectral_min_max_abs_offdiag),
            )
            spectral_diag_df = _spectral_diagnostic_rows(metric_name, sessions)
            spectral_diagnostic_tables.append(spectral_diag_df)
            n_unstable_sessions = int((~spectral_diag_df["spectral_stability_ok"]).sum())
            spectral_metric_skipped = False
            metric_comparison_specs = list(comparison_specs)
            if (
                bool(args.skip_unstable_spectral_metrics)
                and n_unstable_sessions > 0
                and any(spec.name == "laplacian_spectral_distance_signed" for spec in metric_comparison_specs)
            ):
                metric_comparison_specs = [
                    spec for spec in metric_comparison_specs if spec.name != "laplacian_spectral_distance_signed"
                ]
                spectral_metric_skipped = True
                skipped_metric_rows.append(
                    {
                        "connectivity_metric": metric_name,
                        "comparison_metric": "laplacian_spectral_distance_signed",
                        "reason": (
                            f"Skipped because {n_unstable_sessions} session matrices had near-zero off-diagonal weights "
                            f"(thresholds: mean<{float(args.spectral_min_mean_abs_offdiag):.3g} and "
                            f"max<{float(args.spectral_min_max_abs_offdiag):.3g})."
                        ),
                    }
                )
                print(
                    f"Skipping laplacian_spectral_distance_signed for {metric_name}: "
                    f"{n_unstable_sessions}/{len(sessions)} sessions failed spectral-stability diagnostics.",
                    flush=True,
                )

            pairwise_metric_df = _build_pairwise_table(metric_name, sessions, metric_comparison_specs)
            pairwise_tables.append(pairwise_metric_df)

            count_source_name = metric_comparison_specs[0].name if metric_comparison_specs else comparison_specs[0].name
            counts = _summarize_pair_counts(
                pairwise_metric_df.loc[pairwise_metric_df["comparison_metric"] == count_source_name]
            )
            metric_inventory_rows.append(
                {
                    "connectivity_metric": metric_name,
                    "n_session_labels": len(sessions),
                    "n_subjects": len({session.subject for session in sessions}),
                    "n_off_sessions": int(sum(session.state == "off" for session in sessions)),
                    "n_on_sessions": int(sum(session.state == "on" for session in sessions)),
                    "n_nodes": len(node_labels),
                    "directed": bool(directed),
                    "removed_roi_labels": json.dumps(removed_labels),
                    "spectral_n_stable_sessions": int(len(sessions) - n_unstable_sessions),
                    "spectral_n_unstable_sessions": int(n_unstable_sessions),
                    "spectral_min_mean_abs_offdiag_sym": float(spectral_diag_df["mean_abs_offdiag_sym"].min()),
                    "spectral_median_mean_abs_offdiag_sym": float(spectral_diag_df["mean_abs_offdiag_sym"].median()),
                    "spectral_max_max_abs_offdiag_sym": float(spectral_diag_df["max_abs_offdiag_sym"].max()),
                    "laplacian_spectral_distance_skipped": bool(spectral_metric_skipped),
                    **counts,
                }
            )

            edge_summary_df, edge_detail_df = _edge_variance_rows(metric_name, sessions, node_labels)
            edge_variance_summary_tables.append(edge_summary_df)
            if not edge_detail_df.empty:
                top_edge_detail_df = edge_detail_df.nlargest(20, "abs_variance_delta").copy()
                edge_variance_detail_tables.append(top_edge_detail_df)

        pairwise_df = pd.concat(pairwise_tables, ignore_index=True)

    if pairwise_df.empty:
        raise RuntimeError("No pairwise rows available after filtering.")

    summary_df = _build_summary_table(pairwise_df)

    for metric_name in selected_metrics:
        _save_pairwise_metric_matrix(
            pairwise_df=pairwise_df,
            connectivity_metric=metric_name,
            comparison_metric="laplacian_spectral_distance_signed",
            out_dir=out_dir,
        )
        metric_out = out_dir / metric_name
        metric_out.mkdir(parents=True, exist_ok=True)
        dist_stats_df = _save_distribution_plot(
            pairwise_df=pairwise_df,
            connectivity_metric=metric_name,
            comparison_metric="laplacian_spectral_distance_signed",
            cohort_name="cross_subject_only",
            out_png=metric_out / "cross_subject_only_laplacian_spectral_distance_signed_distribution.png",
        )
        if not dist_stats_df.empty:
            distribution_stats_tables.append(dist_stats_df)

    summary_df.to_csv(out_dir / "pairwise_separation_summary.csv", index=False)
    pairwise_df.to_csv(out_dir / "pairwise_metric_values.csv", index=False)
    if not pairwise_input_mode:
        pd.DataFrame(metric_inventory_rows).to_csv(out_dir / "metric_inventory.csv", index=False)
        pd.concat(spectral_diagnostic_tables, ignore_index=True).to_csv(
            out_dir / "spectral_stability_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(skipped_metric_rows).to_csv(
            out_dir / "skipped_comparison_metrics.csv",
            index=False,
        )
        pd.concat(edge_variance_summary_tables, ignore_index=True).to_csv(
            out_dir / "edge_variance_summary.csv",
            index=False,
        )
        if edge_variance_detail_tables:
            pd.concat(edge_variance_detail_tables, ignore_index=True).to_csv(
                out_dir / "top_edge_variance_differences.csv",
                index=False,
            )
    elif skipped_metric_rows:
        pd.DataFrame(skipped_metric_rows).to_csv(
            out_dir / "skipped_comparison_metrics.csv",
            index=False,
        )
    if distribution_stats_tables:
        pd.concat(distribution_stats_tables, ignore_index=True).to_csv(
            out_dir / "laplacian_spectral_distance_signed_distribution_stats.csv",
            index=False,
        )

    for cohort_name in ("all_pairs", "cross_subject_only"):
        cohort_df = summary_df.loc[summary_df["cohort"] == cohort_name].copy()
        cohort_df.to_csv(out_dir / f"{cohort_name}_summary.csv", index=False)
        if not cohort_df.empty:
            best_by_connectivity = cohort_df.sort_values(
                ["connectivity_metric", "auc_within_gt_between", "cohen_d_oriented"],
                ascending=[True, False, False],
            ).groupby("connectivity_metric", as_index=False).head(1)
            best_by_connectivity.to_csv(out_dir / f"{cohort_name}_best_by_connectivity_metric.csv", index=False)

    _save_heatmap(
        summary_df,
        cohort_name="cross_subject_only",
        value_column="auc_within_gt_between",
        out_png=out_dir / "cross_subject_auc_heatmap.png",
        title="Separation Probability: within-condition score > between-condition score (cross-subject only)",
        cbar_label="Separation probability",
        cmap="viridis",
    )
    _save_heatmap(
        summary_df,
        cohort_name="all_pairs",
        value_column="auc_within_gt_between",
        out_png=out_dir / "all_pairs_auc_heatmap.png",
        title="Separation Probability: within-condition score > between-condition score (all pairs)",
        cbar_label="Separation probability",
        cmap="viridis",
    )
    _save_heatmap(
        summary_df,
        cohort_name="cross_subject_only",
        value_column="cohen_d_oriented",
        out_png=out_dir / "cross_subject_cohen_d_heatmap.png",
        title="Cohen d on oriented scores (cross-subject only)",
        cbar_label="Cohen d",
        cmap="coolwarm",
        center=0.0,
    )
    _save_top_barplot(
        summary_df,
        cohort_name="cross_subject_only",
        top_k=int(args.top_k),
        out_png=out_dir / "top_cross_subject_auc.png",
    )

    best_cross = summary_df.loc[summary_df["cohort"] == "cross_subject_only"].nlargest(1, "auc_within_gt_between")
    if not best_cross.empty:
        best_row = best_cross.iloc[0]
        _save_distribution_plot(
            pairwise_df,
            connectivity_metric=str(best_row["connectivity_metric"]),
            comparison_metric=str(best_row["comparison_metric"]),
            cohort_name="cross_subject_only",
            out_png=out_dir / "best_cross_subject_distribution.png",
        )

    manifest = {
        "advanced_root": str(advanced_root),
        "out_dir": str(out_dir),
        "pairwise_csv": str(pairwise_csv) if pairwise_csv is not None else None,
        "selected_connectivity_metrics": selected_metrics,
        "selected_comparison_metrics": selected_comparison_names,
        "excluded_subjects": sorted(excluded_subjects),
        "spectral_stability": {
            "skip_unstable_spectral_metrics": bool(args.skip_unstable_spectral_metrics),
            "min_mean_abs_offdiag": float(args.spectral_min_mean_abs_offdiag),
            "min_max_abs_offdiag": float(args.spectral_min_max_abs_offdiag),
            "skipped_metric_count": int(len(skipped_metric_rows)),
        },
        "permutations": int(args.permutations),
        "random_seed": int(args.random_seed),
        "n_pairwise_rows": int(len(pairwise_df)),
        "n_summary_rows": int(len(summary_df)),
        "best_cross_subject": (
            best_cross.iloc[0][
                [
                    "connectivity_metric",
                    "comparison_metric",
                    "auc_within_gt_between",
                    "cohen_d_oriented",
                    "lme_p_two_sided",
                    "lme_q_two_sided_fdr_bh",
                ]
            ].to_dict()
            if not best_cross.empty
            else None
        ),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
