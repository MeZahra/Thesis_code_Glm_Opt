#!/usr/bin/env python3
"""Create supplementary connectivity figures without Unassigned Active Voxels.

This script is intentionally additive: it reads existing analysis outputs and writes
new sidecar figures/tables without replacing the original results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

from GVS_similarity import (
    ALL_CONDITION_CODES,
    ROI_REFERENCE_DELTA_SPECS,
    _condition_display_name,
    _render_annotated_pivot_heatmap,
    _write_significant_roi_condition_panel_png,
    _write_significant_roi_condition_table_png,
)
from analyze_pairwise_metric_separation import (
    LABEL_RE,
    SESSION_TO_STATE,
    _laplacian_spectral_distance,
    _save_distribution_plot,
    _signed_normalized_laplacian_spectrum,
)
from roi_metrics import METRIC_REGISTRY


REPO_ROOT = Path(__file__).resolve().parents[2]
GVS_ROOT = REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "gvs_similarity_hemi"
GVS_DELTA_ROOT = GVS_ROOT / "roi_condition_reference_deltas"
GVS_TABLES_DIR = GVS_DELTA_ROOT / "tables"
GVS_PLOTS_DIR = GVS_DELTA_ROOT / "plots"
GVS_FILTERED_TABLES_DIR = GVS_TABLES_DIR / "without_unassigned"
GVS_FILTERED_PLOTS_DIR = GVS_PLOTS_DIR / "without_unassigned"

NETWORK_SELECTED_TS_ROOT = (
    REPO_ROOT / "results" / "connectivity" / "tmp" / "tmp-roi_edge_network_from_data" / "all"
)
NETWORK_NODE_PATH = (
    REPO_ROOT / "results" / "connectivity" / "tmp" / "tmp-roi_edge_network_from_data" / "roi_nodes.csv"
)
NETWORK_ROI_SUMMARY_PATH = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_roi_summary.json"
NETWORK_MANIFEST_PATH = REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "analysis_manifest.json"
NETWORK_ACCEPTED_PAIRWISE_PATH = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "roi_edge_network"
    / "advanced_metrics"
    / "random_graph_distance_null_laplacian"
    / "mutual_information_ksg"
    / "selected_pairwise_graph_distance.csv"
)
NETWORK_OUT_DIR = (
    REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "mutual_information_ksg"
)
NETWORK_FILTERED_OUT_DIR = NETWORK_OUT_DIR / "without_unassigned"
NETWORK_METRIC = "mutual_information_ksg"
COMPARISON_METRIC = "laplacian_spectral_distance_signed"

REQUESTED_GVS_PREFIXES = {
    "off_condition_minus_sham_off_roi_mean_delta",
    "on_condition_minus_sham_on_roi_mean_delta",
}


@dataclass(frozen=True)
class SessionSpectrum:
    label: str
    subject: str
    session: int
    state: str
    spectrum: np.ndarray


def _roi_name_lookup() -> dict[int, str]:
    if not NETWORK_ROI_SUMMARY_PATH.exists():
        return {}
    payload = json.loads(NETWORK_ROI_SUMMARY_PATH.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for row in payload.get("roi_rows", []):
        roi_id = row.get("roi_id")
        roi_name = row.get("roi_name")
        if isinstance(roi_id, int) and isinstance(roi_name, str):
            out[int(roi_id)] = roi_name
    return out


def _is_unassigned(label: str) -> bool:
    return "unassigned active voxels" in str(label).casefold()


def _is_unassigned_node(row: pd.Series, roi_names: dict[int, str]) -> bool:
    if _is_unassigned(str(row.get("node_name", row.get("roi_label", "")))):
        return True
    base_roi_id = row.get("base_roi_id")
    if pd.notna(base_roi_id):
        try:
            roi_name = roi_names.get(int(base_roi_id), "")
        except (TypeError, ValueError):
            roi_name = ""
        return _is_unassigned(roi_name)
    return False


def _ensure_dirs() -> None:
    GVS_FILTERED_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    GVS_FILTERED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    NETWORK_FILTERED_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _recompute_subject_fdr(stats_df: pd.DataFrame) -> pd.DataFrame:
    out = stats_df.copy()
    out["q_value_fdr"] = np.nan
    out["significant_fdr"] = False
    for _, group_df in out.groupby("subject", dropna=False, observed=False, sort=False):
        idx = group_df.index.to_numpy(dtype=np.int64)
        p_values = group_df["p_value_two_sided"].to_numpy(dtype=np.float64)
        finite = np.isfinite(p_values)
        if not np.any(finite):
            continue
        sig, q_values = fdrcorrection(p_values[finite], alpha=0.05)
        out.loc[idx[finite], "q_value_fdr"] = q_values
        out.loc[idx[finite], "significant_fdr"] = sig
    return out


def _write_filtered_gvs_outputs() -> list[Path]:
    roi_nodes = pd.read_csv(GVS_ROOT / "common" / "roi_nodes.csv")
    ordered_roi_labels = [
        str(label)
        for label in roi_nodes["roi_label"].tolist()
        if not _is_unassigned(str(label))
    ]
    ordered_condition_labels = [_condition_display_name(code) for code in ALL_CONDITION_CODES]
    written: list[Path] = []

    for spec in ROI_REFERENCE_DELTA_SPECS:
        file_prefix = str(spec["file_prefix"])
        if file_prefix not in REQUESTED_GVS_PREFIXES:
            continue

        stats_path = GVS_TABLES_DIR / f"{file_prefix}_ttest_stats_by_subject_long.csv"
        subject_path = GVS_TABLES_DIR / f"{file_prefix}_by_subject_long.csv"
        if not stats_path.exists() or not subject_path.exists():
            raise FileNotFoundError(f"Missing GVS source tables for {file_prefix}")

        stats_df = pd.read_csv(stats_path)
        stats_df = stats_df.loc[~stats_df["roi_label"].map(_is_unassigned)].copy()
        stats_df = _recompute_subject_fdr(stats_df)
        stats_out = GVS_FILTERED_TABLES_DIR / f"{file_prefix}_ttest_stats_by_subject_long.csv"
        stats_df.to_csv(stats_out, index=False)

        subject_df = pd.read_csv(subject_path)
        subject_df = subject_df.loc[~subject_df["roi_label"].map(_is_unassigned)].copy()
        subject_out = GVS_FILTERED_TABLES_DIR / f"{file_prefix}_by_subject_long.csv"
        subject_df.to_csv(subject_out, index=False)

        summary_df = (
            subject_df.groupby(
                ["target_condition_code", "target_condition_label", "roi_index", "roi_label"],
                dropna=False,
                observed=False,
            )["mean_delta_target_minus_reference"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "group_mean_delta_target_minus_reference",
                    "std": "group_std_delta_target_minus_reference",
                    "count": "n_subjects",
                }
            )
            .sort_values(["target_condition_code", "roi_index"])
            .reset_index(drop=True)
        )
        summary_out = GVS_FILTERED_TABLES_DIR / f"{file_prefix}_summary_long.csv"
        summary_df.to_csv(summary_out, index=False)

        ordered_summary = summary_df.copy()
        ordered_summary["target_condition_label"] = pd.Categorical(
            ordered_summary["target_condition_label"],
            categories=ordered_condition_labels,
            ordered=True,
        )
        ordered_summary["roi_label"] = pd.Categorical(
            ordered_summary["roi_label"],
            categories=ordered_roi_labels,
            ordered=True,
        )
        ordered_summary = ordered_summary.sort_values(
            ["target_condition_label", "roi_label"]
        ).reset_index(drop=True)

        delta_pivot = ordered_summary.pivot(
            index="target_condition_label",
            columns="roi_label",
            values="group_mean_delta_target_minus_reference",
        )
        delta_pivot = delta_pivot.reindex(
            index=ordered_condition_labels,
            columns=ordered_roi_labels,
        )
        delta_pivot.to_csv(GVS_FILTERED_TABLES_DIR / f"{file_prefix}_wide.csv")
        delta_values = subject_df["mean_delta_target_minus_reference"].to_numpy(dtype=np.float64)
        delta_values = delta_values[np.isfinite(delta_values)]
        delta_abs_max = float(np.max(np.abs(delta_values))) if delta_values.size else 1e-6
        delta_abs_max = max(delta_abs_max, 1e-6)

        heatmap_out = GVS_FILTERED_PLOTS_DIR / f"{file_prefix}_heatmap.png"
        _render_annotated_pivot_heatmap(
            delta_pivot,
            title=f"{spec['title']} (without Unassigned Active Voxels)",
            colorbar_label=str(spec["colorbar_label"]),
            out_path=heatmap_out,
            cmap_name="coolwarm",
            vmin=-delta_abs_max,
            vmax=delta_abs_max,
            scientific_below=1e-3,
        )
        written.append(heatmap_out)

        table_out = GVS_FILTERED_PLOTS_DIR / f"{file_prefix}_significant_rois_by_subject_table.png"
        _write_significant_roi_condition_table_png(
            stats_df,
            table_out,
            title=(
                f"Significant ROIs by subject and GVS: {spec['title']} "
                "(FDR < 0.05; without Unassigned Active Voxels)"
            ),
            condition_labels=ordered_condition_labels,
        )
        written.append(table_out)

        panel_out = GVS_FILTERED_PLOTS_DIR / f"{file_prefix}_significant_rois_by_subject_panel.png"
        _write_significant_roi_condition_panel_png(
            stats_df,
            panel_out,
            title=(
                f"Significant ROIs by subject and GVS: {spec['title']} "
                "(FDR < 0.05; without Unassigned Active Voxels)"
            ),
            condition_labels=ordered_condition_labels,
            roi_labels=ordered_roi_labels,
        )
        written.append(panel_out)

    return written


def _load_excluded_subjects() -> set[str]:
    if not NETWORK_MANIFEST_PATH.exists():
        return {"sub-pd017"}
    payload = json.loads(NETWORK_MANIFEST_PATH.read_text(encoding="utf-8"))
    excluded = payload.get("excluded_subjects", ["sub-pd017"])
    return {str(item) for item in excluded}


def _network_metric_kwargs() -> dict[str, float | int]:
    return {"k": 3, "jitter": 1e-10}


def _load_session_timeseries(keep_idx: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for label_dir in sorted(NETWORK_SELECTED_TS_ROOT.glob("sub-*_ses-*")):
        path = label_dir / f"roi_timeseries_{label_dir.name}.npy"
        if not path.exists():
            continue
        roi_ts = np.asarray(np.load(path), dtype=np.float32)
        out[label_dir.name] = roi_ts[keep_idx].astype(np.float32, copy=False)
    return out


def _compute_metric_matrices(ts_by_label: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    metric_fn = METRIC_REGISTRY[NETWORK_METRIC]
    out: dict[str, np.ndarray] = {}
    kwargs = _network_metric_kwargs()
    for label, roi_ts in ts_by_label.items():
        result = metric_fn(roi_ts, **kwargs)
        out[label] = np.asarray(result["matrix"], dtype=np.float64)
    return out


def _spectra_from_matrices(matrices_by_label: dict[str, np.ndarray]) -> list[SessionSpectrum]:
    excluded_subjects = _load_excluded_subjects()
    sessions: list[SessionSpectrum] = []
    for label in sorted(matrices_by_label.keys()):
        match = LABEL_RE.match(label)
        if match is None:
            continue
        subject = match.group(1)
        session = int(match.group(2))
        if subject in excluded_subjects or session not in SESSION_TO_STATE:
            continue
        matrix = np.asarray(matrices_by_label[label], dtype=np.float64)
        np.fill_diagonal(matrix, 0.0)
        matrix_sym = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix_sym, 0.0)
        sessions.append(
            SessionSpectrum(
                label=label,
                subject=subject,
                session=session,
                state=SESSION_TO_STATE[session],
                spectrum=_signed_normalized_laplacian_spectrum(matrix_sym),
            )
        )
    sessions.sort(key=lambda item: (item.subject, item.session))
    return sessions


def _load_filtered_session_spectra() -> tuple[list[SessionSpectrum], list[str], pd.DataFrame, dict[str, object]]:
    node_df = pd.read_csv(NETWORK_NODE_PATH)
    label_col = "node_name" if "node_name" in node_df.columns else "roi_label"
    roi_names = _roi_name_lookup()
    keep_mask = ~node_df.apply(lambda row: _is_unassigned_node(row, roi_names), axis=1)
    keep_idx = np.flatnonzero(keep_mask.to_numpy(dtype=bool))
    kept_nodes = node_df.loc[keep_mask].copy().reset_index(drop=True)
    kept_labels = kept_nodes[label_col].astype(str).tolist()
    kept_nodes.to_csv(NETWORK_FILTERED_OUT_DIR / "roi_nodes.csv", index=False)

    ts_by_label = _load_session_timeseries(keep_idx)
    matrices_by_label = _compute_metric_matrices(ts_by_label)
    sessions = _spectra_from_matrices(matrices_by_label)

    all_idx = np.arange(node_df.shape[0], dtype=np.int64)
    all_ts_by_label = _load_session_timeseries(all_idx)
    all_matrices_by_label = _compute_metric_matrices(all_ts_by_label)
    all_sessions = _spectra_from_matrices(all_matrices_by_label)
    all_pairwise_df = _pairwise_rows_from_spectra(all_sessions)
    reproduction: dict[str, object] = {
        "accepted_pairwise_source": str(NETWORK_ACCEPTED_PAIRWISE_PATH),
        "all_node_recomputed_cross_subject_means": _cross_subject_means(all_pairwise_df),
    }
    if NETWORK_ACCEPTED_PAIRWISE_PATH.exists():
        accepted_df = pd.read_csv(NETWORK_ACCEPTED_PAIRWISE_PATH)
        accepted_df = accepted_df.rename(columns={"distance": "raw_score"})
        reproduction["accepted_cross_subject_means"] = _cross_subject_means(accepted_df)
        merged = accepted_df.merge(
            all_pairwise_df,
            on=["label_a", "label_b"],
            suffixes=("_accepted", "_recomputed"),
        )
        if not merged.empty:
            reproduction["all_node_max_abs_distance_difference_vs_accepted"] = float(
                np.max(np.abs(merged["raw_score_accepted"] - merged["raw_score_recomputed"]))
            )
            reproduction["all_node_n_compared_pairs"] = int(merged.shape[0])
    return sessions, kept_labels, kept_nodes, reproduction


def _pair_label(state_a: str, state_b: str) -> str:
    if state_a == state_b == "off":
        return "off-off"
    if state_a == state_b == "on":
        return "on-on"
    return "off-on"


def _pairwise_rows_from_spectra(sessions: list[SessionSpectrum]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, session_a in enumerate(sessions):
        for j in range(i + 1, len(sessions)):
            session_b = sessions[j]
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


def _cross_subject_means(pairwise_df: pd.DataFrame) -> dict[str, float]:
    if pairwise_df.empty:
        return {}
    same_col = "same_subject"
    value_col = "raw_score" if "raw_score" in pairwise_df.columns else "distance"
    subset = pairwise_df.loc[~pairwise_df[same_col].astype(bool)].copy()
    return {
        str(k): float(v)
        for k, v in subset.groupby("pair_label", observed=False)[value_col].mean().to_dict().items()
    }


def _distance_matrix_from_pairwise(pairwise_df: pd.DataFrame, sessions: list[SessionSpectrum]) -> np.ndarray:
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


def _write_filtered_network_outputs() -> list[Path]:
    sessions, kept_labels, kept_nodes, reproduction = _load_filtered_session_spectra()
    if not sessions:
        raise RuntimeError("No session spectra were loaded for the filtered network analysis.")

    label_names = [session.label.replace("_ses-", "-ses") for session in sessions]
    pairwise_df = _pairwise_rows_from_spectra(sessions)
    distance_matrix = _distance_matrix_from_pairwise(pairwise_df, sessions)
    pairwise_out = NETWORK_FILTERED_OUT_DIR / "pairwise_metric_values.csv"
    pairwise_df.to_csv(pairwise_out, index=False)

    matrix_out = (
        NETWORK_FILTERED_OUT_DIR
        / "all_subject_session_pairwise_laplacian_spectral_distance_signed.csv"
    )
    pd.DataFrame(distance_matrix, index=label_names, columns=label_names).to_csv(matrix_out)

    out_png = (
        NETWORK_FILTERED_OUT_DIR
        / "cross_subject_only_laplacian_spectral_distance_signed_distribution.png"
    )
    stats_df = _save_distribution_plot(
        pairwise_df=pairwise_df,
        connectivity_metric=NETWORK_METRIC,
        comparison_metric=COMPARISON_METRIC,
        cohort_name="cross_subject_only",
        out_png=out_png,
    )
    stats_out = (
        NETWORK_FILTERED_OUT_DIR
        / "cross_subject_only_laplacian_spectral_distance_signed_distribution_stats.csv"
    )
    stats_df.to_csv(stats_out, index=False)

    metadata = {
        "source_selected_ts_root": str(NETWORK_SELECTED_TS_ROOT),
        "source_node_csv": str(NETWORK_NODE_PATH),
        "connectivity_metric": NETWORK_METRIC,
        "comparison_metric": COMPARISON_METRIC,
        "removed_node_pattern": "Unassigned Active Voxels",
        "n_nodes": int(len(kept_labels)),
        "n_edges": int(len(kept_labels) * (len(kept_labels) - 1) / 2),
        "n_selected_voxels_after_removal": int(kept_nodes["n_selected_voxels"].sum())
        if "n_selected_voxels" in kept_nodes.columns
        else None,
        "labels": kept_labels,
        "n_sessions": int(len(sessions)),
        "n_pairwise_rows": int(pairwise_df.shape[0]),
        "filtered_cross_subject_means": _cross_subject_means(pairwise_df),
        "all_node_reproduction_check": reproduction,
    }
    (NETWORK_FILTERED_OUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return [out_png, out_png.with_suffix(".pdf"), matrix_out, pairwise_out, stats_out]


def main() -> None:
    _ensure_dirs()
    written = []
    written.extend(_write_filtered_gvs_outputs())
    written.extend(_write_filtered_network_outputs())
    print("Wrote supplementary outputs without Unassigned Active Voxels:")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
