#!/usr/bin/env python3
"""Regenerate the ROI-edge network analysis after dropping Unassigned nodes.

This is additive relative to the original analysis: it reads the exact saved
32-node mutual_information_ksg matrices that reproduce the original
cross_subject_only_laplacian_spectral_distance_signed_distribution.png, drops the
two Unassigned Active Voxels nodes, and writes filtered outputs under
results/connectivity/roi_edge_network/mutual_information_ksg/without_unassigned.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_pairwise_metric_separation import (
    LABEL_RE,
    SESSION_TO_STATE,
    _build_summary_table,
    _laplacian_spectral_distance,
    _plot_pairwise_heatmap,
    _save_distribution_plot,
    _signed_normalized_laplacian_spectrum,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
NETWORK_NODE_PATH = REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "roi_nodes.csv"
NETWORK_ADVANCED_ROOT = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "tmp"
    / "tmp-roi_edge_network"
    / "tmp"
)
NETWORK_MANIFEST_PATH = REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "analysis_manifest.json"
SOURCE_FIGURE_PATH = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "roi_edge_network"
    / "mutual_information_ksg"
    / "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
)
SOURCE_DISTANCE_MATRIX_PATH = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "roi_edge_network"
    / "mutual_information_ksg"
    / "all_subject_session_pairwise_laplacian_spectral_distance_signed.csv"
)
SOURCE_PAIRWISE_PATH = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "roi_edge_network"
    / "1tmp"
    / "pairwise_metric_values.csv"
)
OUT_DIR = REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "mutual_information_ksg" / "without_unassigned"
FILTERED_ADVANCED_ROOT = OUT_DIR / "advanced_metrics"
NETWORK_METRIC = "mutual_information_ksg"
COMPARISON_METRIC = "laplacian_spectral_distance_signed"
RANDOM_SEED = 0
N_CLASS_LABEL_SHUFFLES = 100_000


@dataclass(frozen=True)
class SessionSpectrum:
    label: str
    subject: str
    session: int
    state: str
    spectrum: np.ndarray


def _display_subject_session_label(label: str) -> str:
    match = LABEL_RE.match(label)
    if match is None:
        return label
    return f"{match.group(1)}-ses{int(match.group(2))}"


def _is_unassigned(label: str) -> bool:
    return "unassigned active voxels" in str(label).casefold()


def _load_excluded_subjects() -> set[str]:
    if not NETWORK_MANIFEST_PATH.exists():
        return {"sub-pd017"}
    payload = json.loads(NETWORK_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {str(item) for item in payload.get("excluded_subjects", ["sub-pd017"])}


def _pair_label(state_a: str, state_b: str) -> str:
    if state_a == state_b == "off":
        return "off-off"
    if state_a == state_b == "on":
        return "on-on"
    return "off-on"


def _matrix_path(label_dir: Path) -> Path:
    return label_dir / NETWORK_METRIC / f"{NETWORK_METRIC}.npy"


def _labels_path(label_dir: Path) -> Path:
    return label_dir / NETWORK_METRIC / f"{NETWORK_METRIC}_connectome.labels.txt"


def _load_source_labels() -> list[str]:
    for label_dir in sorted(NETWORK_ADVANCED_ROOT.glob("sub-*_ses-*")):
        path = _labels_path(label_dir)
        if path.exists():
            return path.read_text(encoding="utf-8").strip().splitlines()
    raise RuntimeError(f"No source labels found under {NETWORK_ADVANCED_ROOT}")


def _filter_matrix(matrix: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[np.ix_(keep_idx, keep_idx)]


def _write_filtered_advanced_metrics(keep_idx: np.ndarray, kept_labels: list[str]) -> int:
    n_written = 0
    for label_dir in sorted(NETWORK_ADVANCED_ROOT.glob("sub-*_ses-*")):
        match = LABEL_RE.match(label_dir.name)
        if match is None:
            continue
        matrix_path = _matrix_path(label_dir)
        if not matrix_path.exists():
            continue
        source_matrix = np.load(matrix_path)
        matrix = _filter_matrix(source_matrix, keep_idx)
        metric_out = FILTERED_ADVANCED_ROOT / label_dir.name / NETWORK_METRIC
        metric_out.mkdir(parents=True, exist_ok=True)
        np.save(metric_out / f"{NETWORK_METRIC}.npy", matrix.astype(np.float32))
        pd.DataFrame(matrix, index=kept_labels, columns=kept_labels).to_csv(
            metric_out / f"{NETWORK_METRIC}.csv"
        )
        (metric_out / f"{NETWORK_METRIC}_connectome.labels.txt").write_text(
            "\n".join(kept_labels) + "\n",
            encoding="utf-8",
        )
        meta_path = label_dir / NETWORK_METRIC / f"{NETWORK_METRIC}_meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.update(
            {
                "label": label_dir.name,
                "metric": NETWORK_METRIC,
                "n_nodes": int(matrix.shape[0]),
                "filtered_from": str(matrix_path),
                "removed_node_pattern": "Unassigned Active Voxels",
                "source_n_nodes": int(source_matrix.shape[0]),
            }
        )
        finite = matrix[np.isfinite(matrix)]
        if finite.size:
            meta["vmin"] = float(np.min(finite))
            meta["vmax"] = float(np.max(finite))
        (metric_out / f"{NETWORK_METRIC}_meta.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        n_written += 1
    return n_written


def _load_sessions(keep_idx: np.ndarray | None = None) -> list[SessionSpectrum]:
    excluded_subjects = _load_excluded_subjects()
    sessions: list[SessionSpectrum] = []
    for label_dir in sorted(NETWORK_ADVANCED_ROOT.glob("sub-*_ses-*")):
        match = LABEL_RE.match(label_dir.name)
        if match is None:
            continue
        subject = match.group(1)
        session = int(match.group(2))
        if subject in excluded_subjects or session not in SESSION_TO_STATE:
            continue
        matrix_path = _matrix_path(label_dir)
        if not matrix_path.exists():
            continue
        matrix = np.asarray(np.load(matrix_path), dtype=np.float64)
        if keep_idx is not None:
            matrix = _filter_matrix(matrix, keep_idx)
        matrix_for_spectrum = matrix.copy()
        np.fill_diagonal(matrix_for_spectrum, 0.0)
        matrix_sym = 0.5 * (matrix_for_spectrum + matrix_for_spectrum.T)
        np.fill_diagonal(matrix_sym, 0.0)
        sessions.append(
            SessionSpectrum(
                label=label_dir.name,
                subject=subject,
                session=session,
                state=SESSION_TO_STATE[session],
                spectrum=_signed_normalized_laplacian_spectrum(matrix_sym),
            )
        )
    sessions.sort(key=lambda item: (item.subject, item.session))
    return sessions


def _pairwise_rows(sessions: list[SessionSpectrum]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, session_a in enumerate(sessions):
        for session_b in sessions[i + 1 :]:
            distance = _laplacian_spectral_distance(session_a.spectrum, session_b.spectrum)
            same_subject = session_a.subject == session_b.subject
            pair_label = _pair_label(session_a.state, session_b.state)
            rows.append(
                {
                    "connectivity_metric": NETWORK_METRIC,
                    "label_a": session_a.label,
                    "label_b": session_b.label,
                    "subject_a": session_a.subject,
                    "subject_b": session_b.subject,
                    "session_a": session_a.session,
                    "session_b": session_b.session,
                    "state_a": session_a.state,
                    "state_b": session_b.state,
                    "pair_class": "within_condition"
                    if session_a.state == session_b.state
                    else "between_condition",
                    "pair_label": pair_label,
                    "same_subject": same_subject,
                    "comparison_metric": COMPARISON_METRIC,
                    "comparison_kind": "graph_distance",
                    "higher_is_more_similar": False,
                    "raw_score": distance,
                    "oriented_score": -distance,
                }
            )
    return pd.DataFrame(rows)


def _distance_matrix(pairwise_df: pd.DataFrame, sessions: list[SessionSpectrum]) -> np.ndarray:
    labels = [session.label for session in sessions]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for row in pairwise_df.itertuples(index=False):
        i = label_to_idx[str(row.label_a)]
        j = label_to_idx[str(row.label_b)]
        value = float(row.raw_score)
        matrix[i, j] = value
        matrix[j, i] = value
    return matrix


def _write_distance_matrix(pairwise_df: pd.DataFrame, sessions: list[SessionSpectrum]) -> Path:
    labels_raw = [session.label for session in sessions]
    labels = [_display_subject_session_label(label) for label in labels_raw]
    matrix = _distance_matrix(pairwise_df, sessions)
    out_csv = OUT_DIR / "all_subject_session_pairwise_laplacian_spectral_distance_signed.csv"
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(out_csv)
    _plot_pairwise_heatmap(
        matrix,
        labels,
        OUT_DIR / "all_subject_session_pairwise_laplacian_spectral_distance_signed.png",
        title=f"{NETWORK_METRIC} | All subject-session {COMPARISON_METRIC} (without unassigned)",
        cbar_label=COMPARISON_METRIC,
    )
    return out_csv


def _cross_subject_means(pairwise_df: pd.DataFrame) -> dict[str, float]:
    subset = pairwise_df.loc[~pairwise_df["same_subject"].astype(bool)]
    return {
        str(label): float(value)
        for label, value in subset.groupby("pair_label", observed=False)["raw_score"].mean().to_dict().items()
    }


def _source_reproduction_check() -> dict[str, object]:
    if not SOURCE_DISTANCE_MATRIX_PATH.exists():
        return {"source_distance_matrix": str(SOURCE_DISTANCE_MATRIX_PATH), "exists": False}

    sessions = _load_sessions(keep_idx=None)
    pairwise_df = _pairwise_rows(sessions)
    matrix = _distance_matrix(pairwise_df, sessions)
    source_df = pd.read_csv(SOURCE_DISTANCE_MATRIX_PATH, index_col=0)
    labels = [_display_subject_session_label(session.label) for session in sessions]
    result: dict[str, object] = {
        "source_figure": str(SOURCE_FIGURE_PATH),
        "source_distance_matrix": str(SOURCE_DISTANCE_MATRIX_PATH),
        "source_pairwise_values": str(SOURCE_PAIRWISE_PATH),
        "recomputed_n_sessions": int(len(sessions)),
        "recomputed_labels_match_source_matrix": bool(labels == source_df.index.astype(str).tolist()),
    }
    if matrix.shape == source_df.shape and labels == source_df.index.astype(str).tolist():
        diff = np.abs(matrix - source_df.to_numpy(dtype=np.float64))
        result.update(
            {
                "max_abs_distance_difference_vs_source": float(np.max(diff)),
                "mean_abs_distance_difference_vs_source": float(np.mean(diff)),
            }
        )
    return result


def _write_summary_tables(pairwise_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = _build_summary_table(pairwise_df)
    summary_df.to_csv(OUT_DIR / "pairwise_separation_summary.csv", index=False)
    for cohort_name in ("all_pairs", "cross_subject_only"):
        cohort_df = summary_df.loc[summary_df["cohort"] == cohort_name].copy()
        cohort_df.to_csv(OUT_DIR / f"{cohort_name}_summary.csv", index=False)
    return summary_df


def _run_top_edge_analysis() -> None:
    script = Path(__file__).resolve().parent / "extract_top_session_delta_edges.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--advanced-root",
            str(FILTERED_ADVANCED_ROOT),
            "--metric",
            NETWORK_METRIC,
            "--out-dir",
            str(OUT_DIR),
            "--top-k",
            "19",
            "--top-percentile",
            "95",
        ],
        check=True,
    )


def _run_label_permutation() -> None:
    script = Path(__file__).resolve().parent / "plot_permutation_null_figures.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--pairwise-csv",
            str(OUT_DIR / "pairwise_metric_values.csv"),
            "--out-dir",
            str(OUT_DIR / "label_permutation_test"),
            "--metric-label",
            "Mutual Information KSG (30 ROI, without unassigned)",
            "--connectivity-metric",
            NETWORK_METRIC,
            "--comparison-metric",
            COMPARISON_METRIC,
        ],
        check=True,
    )


def _class_contrasts(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    off_off = values[labels == "off-off"]
    on_on = values[labels == "on-on"]
    off_on = values[labels == "off-on"]
    mu_off_off = float(np.mean(off_off))
    mu_on_on = float(np.mean(on_on))
    mu_off_on = float(np.mean(off_on))
    anova = stats.f_oneway(off_off, on_on, off_on)
    return {
        "mu_off_off": mu_off_off,
        "mu_on_on": mu_on_on,
        "mu_off_on": mu_off_on,
        "delta_within": mu_on_on - mu_off_off,
        "delta_sep": mu_off_on - 0.5 * (mu_off_off + mu_on_on),
        "anova_f_stat": float(anova.statistic),
    }


def _shuffle_p_value(null: np.ndarray, observed: float, direction: str) -> float:
    if direction == "left":
        count = int(np.count_nonzero(null <= observed))
    elif direction == "right":
        count = int(np.count_nonzero(null >= observed))
    else:
        center = float(np.mean(null))
        count = int(np.count_nonzero(np.abs(null - center) >= abs(observed - center)))
    return float((count + 1) / (null.size + 1))


def _write_class_label_shuffle(pairwise_df: pd.DataFrame) -> None:
    out_dir = OUT_DIR / "class_label_shuffle_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    cross_df = pairwise_df.loc[~pairwise_df["same_subject"].astype(bool)].copy()
    values = cross_df["raw_score"].to_numpy(dtype=np.float64)
    labels = cross_df["pair_label"].astype(str).to_numpy()
    observed = _class_contrasts(values, labels)
    label_counts = {
        label: int(np.count_nonzero(labels == label))
        for label in ("off-off", "on-on", "off-on")
    }
    label_template = np.concatenate(
        [np.repeat(label, count) for label, count in label_counts.items()]
    )
    rng = np.random.default_rng(RANDOM_SEED)
    null_rows: list[dict[str, float]] = []
    for _ in range(N_CLASS_LABEL_SHUFFLES):
        shuffled_labels = rng.permutation(label_template)
        null_rows.append(_class_contrasts(values, shuffled_labels))
    null_df = pd.DataFrame(null_rows)

    directions = {
        "mu_off_off": "right",
        "mu_on_on": "left",
        "mu_off_on": "two_sided",
        "delta_within": "left",
        "delta_sep": "two_sided",
        "anova_f_stat": "right",
    }
    summary_rows = []
    for key, direction in directions.items():
        null = null_df[key].to_numpy(dtype=np.float64)
        obs = float(observed[key])
        null_sd = float(np.std(null, ddof=1))
        z = float((obs - np.mean(null)) / null_sd) if null_sd > 0 else float("nan")
        summary_rows.append(
            {
                "contrast": key,
                "observed": obs,
                "null_mean": float(np.mean(null)),
                "null_sd": null_sd,
                "z": z,
                "p_value": _shuffle_p_value(null, obs, direction),
                "p_direction": direction,
                "percentile_vs_null": float(np.mean(null <= obs) * 100.0),
                "n_permutations": int(N_CLASS_LABEL_SHUFFLES),
                "shuffle_mode": "pair_label_shuffle_cross_subject_only",
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "class_label_shuffle_summary.csv", index=False)
    (out_dir / "README.txt").write_text(
        "This folder contains a literal class-label shuffle null for the "
        "cross-subject-only mutual_information_ksg Laplacian distance figure after "
        "removing Unassigned Active Voxels. It shuffles OFF-OFF / ON-ON / OFF-ON "
        "labels across cross-subject pairwise scores while preserving class sizes.\n",
        encoding="utf-8",
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    for ax, key in zip(axes.ravel(), directions):
        null = null_df[key].to_numpy(dtype=np.float64)
        obs = float(observed[key])
        ax.hist(null, bins=45, color="#7b9acc", alpha=0.75, edgecolor="white")
        ax.axvline(obs, color="#b03a2e", linewidth=2.0, label="observed")
        ax.set_title(key, fontsize=10)
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "class_label_shuffle_combined.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FILTERED_ADVANCED_ROOT.mkdir(parents=True, exist_ok=True)

    node_df = pd.read_csv(NETWORK_NODE_PATH)
    label_col = "node_name" if "node_name" in node_df.columns else "roi_label"
    source_n_nodes = int(node_df.shape[0])
    source_n_voxels = int(node_df["n_selected_voxels"].sum())
    source_labels = _load_source_labels()
    node_labels = node_df[label_col].astype(str).tolist()
    if source_labels != node_labels:
        raise RuntimeError(
            "Source matrix labels do not match roi_nodes.csv; refusing to filter by position."
        )
    keep_mask = ~node_df[label_col].map(_is_unassigned)
    keep_idx = np.flatnonzero(keep_mask.to_numpy(dtype=bool))
    kept_nodes = node_df.loc[keep_mask].copy().reset_index(drop=True)
    kept_labels = kept_nodes[label_col].astype(str).tolist()
    kept_nodes.to_csv(OUT_DIR / "roi_nodes.csv", index=False)

    reproduction = _source_reproduction_check()
    n_filtered_matrices = _write_filtered_advanced_metrics(keep_idx, kept_labels)
    sessions = _load_sessions(keep_idx)
    pairwise_df = _pairwise_rows(sessions)
    pairwise_df.to_csv(OUT_DIR / "pairwise_metric_values.csv", index=False)

    _write_distance_matrix(pairwise_df, sessions)
    _write_summary_tables(pairwise_df)

    fig_path = OUT_DIR / "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
    stats_df = _save_distribution_plot(
        pairwise_df=pairwise_df,
        connectivity_metric=NETWORK_METRIC,
        comparison_metric=COMPARISON_METRIC,
        cohort_name="cross_subject_only",
        out_png=fig_path,
    )
    stats_df.to_csv(
        OUT_DIR / "cross_subject_only_laplacian_spectral_distance_signed_distribution_stats.csv",
        index=False,
    )
    stats_df.to_csv(OUT_DIR / "laplacian_spectral_distance_signed_distribution_stats.csv", index=False)
    _run_top_edge_analysis()
    _run_label_permutation()
    _write_class_label_shuffle(pairwise_df)

    manifest = {
        "source": "32-node ROI-edge network",
        "source_figure": str(SOURCE_FIGURE_PATH),
        "source_distance_matrix": str(SOURCE_DISTANCE_MATRIX_PATH),
        "source_pairwise_values": str(SOURCE_PAIRWISE_PATH),
        "source_node_csv": str(NETWORK_NODE_PATH),
        "source_advanced_root": str(NETWORK_ADVANCED_ROOT),
        "source_n_nodes": source_n_nodes,
        "source_n_selected_voxels": source_n_voxels,
        "removed_node_pattern": "Unassigned Active Voxels",
        "n_nodes_after_removal": int(kept_nodes.shape[0]),
        "n_edges_after_removal": int(kept_nodes.shape[0] * (kept_nodes.shape[0] - 1) / 2),
        "n_selected_voxels_after_removal": int(kept_nodes["n_selected_voxels"].sum()),
        "removed_selected_voxels": int(source_n_voxels - kept_nodes["n_selected_voxels"].sum()),
        "n_filtered_session_matrices_written": int(n_filtered_matrices),
        "filtered_advanced_root": str(FILTERED_ADVANCED_ROOT),
        "n_pairwise_rows": int(pairwise_df.shape[0]),
        "cross_subject_means": _cross_subject_means(pairwise_df),
        "source_reproduction_check": reproduction,
    }
    (OUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote 30-node filtered network outputs to {OUT_DIR}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
