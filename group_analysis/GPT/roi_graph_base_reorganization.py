from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import adjusted_rand_score

from common_io import (
    DEFAULT_DROP_LABEL_PATTERNS,
    REPO_ROOT,
    ROI_EDGE_RESULTS_ROOT,
    aggregate_matrix_by_base_roi,
    bh_fdr,
    drop_labels_and_matrix,
    ensure_dir,
    infer_anatomical_system,
    list_paired_subjects_for_metric,
    load_metric_matrix,
    paired_delta_stats,
    sanitize_matrix,
    to_serializable,
    write_json,
)
from roi_graph_reorganization import (
    _compute_group_pair_rows,
    _plot_hub_reorganization,
    _plot_module_delta_heatmap,
    classify_hub,
    consensus_partition,
    participation_coefficient,
    weighted_modularity,
    within_module_degree_z,
)


RUN_LABEL_RE = re.compile(r"^(sub-[^_]+)_ses-(\d+)_run-(\d+)$")
DEFAULT_RUN_LEVEL_METRICS_ROOT = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "GPT"
    / "run_level_partial_correlation_metrics"
    / "roi_edge_network"
    / "advanced_metrics"
)
NODE_METRICS = [
    "node_strength_abs",
    "node_strength_positive",
    "participation_coeff",
    "within_module_z",
]
_RUN_LABEL_MAP: dict[str, str] | None = None


def _run_label_map() -> dict[str, str]:
    global _RUN_LABEL_MAP
    if _RUN_LABEL_MAP is not None:
        return _RUN_LABEL_MAP

    nodes_path = ROI_EDGE_RESULTS_ROOT / "roi_nodes.csv"
    nodes_df = pd.read_csv(nodes_path)
    mapping = {}
    for row in nodes_df.itertuples(index=False):
        generic_label = f"{row.hemisphere} ROI_{int(row.base_roi_id)}"
        mapping[generic_label] = str(row.node_name)
    _RUN_LABEL_MAP = mapping
    return mapping


def _normalize_run_labels(labels: list[str]) -> list[str]:
    mapping = _run_label_map()
    return [mapping.get(label, label) for label in labels]


def _aggregate_base_graph(
    matrix: np.ndarray,
    labels: list[str],
    include_base_rois: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    adjacency = sanitize_matrix(matrix)
    aggregated, base_labels, _ = aggregate_matrix_by_base_roi(
        adjacency,
        labels,
        include_base_rois=include_base_rois,
    )
    aggregated = sanitize_matrix(aggregated, zero_diagonal=True)
    return aggregated, base_labels


def _prepare_session_graph(subject: str, session: int, metric: str, drop_patterns) -> dict[str, object]:
    raw_matrix, raw_labels = load_metric_matrix(subject, session, metric)
    trimmed_matrix, trimmed_labels, _ = drop_labels_and_matrix(raw_labels, raw_matrix, drop_patterns)
    adjacency, labels = _aggregate_base_graph(trimmed_matrix, trimmed_labels)
    positive = np.clip(adjacency, 0.0, None)
    absolute = np.abs(adjacency)
    return {
        "labels": labels,
        "adjacency": adjacency,
        "positive": positive,
        "absolute": absolute,
    }


def _load_run_level_graph(
    metrics_root: Path,
    label: str,
    metric: str,
    drop_patterns,
    include_base_rois: list[str],
) -> dict[str, object]:
    metric_dir = metrics_root / label / metric
    matrix_path = metric_dir / f"{metric}.npy"
    labels_path = metric_dir / f"{metric}_connectome.labels.txt"
    if not matrix_path.exists() or not labels_path.exists():
        raise FileNotFoundError(f"Missing run-level metric files for {label}: {metric_dir}")
    raw_matrix = np.load(matrix_path)
    raw_labels = _normalize_run_labels(labels_path.read_text(encoding="utf-8").splitlines())
    trimmed_matrix, trimmed_labels, _ = drop_labels_and_matrix(raw_labels, raw_matrix, drop_patterns)
    adjacency, labels = _aggregate_base_graph(
        trimmed_matrix,
        trimmed_labels,
        include_base_rois=include_base_rois,
    )
    if labels != include_base_rois:
        raise ValueError(f"Run-level labels do not match session-level base ROI order for {label}")
    positive = np.clip(adjacency, 0.0, None)
    absolute = np.abs(adjacency)
    return {
        "labels": labels,
        "adjacency": adjacency,
        "positive": positive,
        "absolute": absolute,
    }


def _list_complete_run_labels(
    metrics_root: Path,
    metric: str,
    required_subjects: list[str],
) -> list[str]:
    if not metrics_root.exists():
        return []

    store: dict[str, dict[int, set[int]]] = {}
    for label_dir in sorted(metrics_root.glob("sub-pd*_ses-*_run-*")):
        match = RUN_LABEL_RE.match(label_dir.name)
        if not match:
            continue
        subject = str(match.group(1))
        session = int(match.group(2))
        run = int(match.group(3))
        metric_path = label_dir / metric / f"{metric}.npy"
        labels_path = label_dir / metric / f"{metric}_connectome.labels.txt"
        if not metric_path.exists() or not labels_path.exists():
            continue
        store.setdefault(subject, {}).setdefault(session, set()).add(run)

    labels = []
    for subject in required_subjects:
        runs_by_session = store.get(subject, {})
        if runs_by_session.get(1) != {1, 2} or runs_by_session.get(2) != {1, 2}:
            continue
        for session in (1, 2):
            for run in (1, 2):
                labels.append(f"{subject}_ses-{session}_run-{run}")
    return labels


def _append_node_rows(
    rows: list[dict],
    graph: dict[str, object],
    *,
    subject: str,
    session: int,
    run: int | None,
    consensus_labels_arr: np.ndarray,
) -> None:
    labels = list(graph["labels"])
    positive = np.asarray(graph["positive"])
    absolute = np.asarray(graph["absolute"])

    node_strength_abs = absolute.sum(axis=1)
    node_strength_pos = positive.sum(axis=1)
    participation = participation_coefficient(positive, consensus_labels_arr)
    within_z = within_module_degree_z(positive, consensus_labels_arr)

    for idx, roi in enumerate(labels):
        row = {
            "subject": subject,
            "session": int(session),
            "state": "OFF" if int(session) == 1 else "ON",
            "roi": roi,
            "base_roi": roi,
            "anatomical_system": infer_anatomical_system(roi),
            "node_strength_abs": float(node_strength_abs[idx]),
            "node_strength_positive": float(node_strength_pos[idx]),
            "participation_coeff": float(participation[idx]),
            "within_module_z": float(within_z[idx]),
            "consensus_community": int(consensus_labels_arr[idx] + 1),
            "hub_class": classify_hub(float(participation[idx]), float(within_z[idx])),
        }
        if run is not None:
            row["run"] = int(run)
        rows.append(row)


def _build_subject_level_deltas(node_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_rows = []
    node_tests = []
    for roi, roi_df in node_df.groupby("roi", sort=True):
        off = roi_df[roi_df["session"] == 1].sort_values("subject")
        on = roi_df[roi_df["session"] == 2].sort_values("subject")
        merged = off.merge(on, on="subject", suffixes=("_off", "_on"))
        if merged.empty:
            continue
        for metric_name in NODE_METRICS:
            delta = merged[f"{metric_name}_on"] - merged[f"{metric_name}_off"]
            stats_dict = paired_delta_stats(delta)
            node_tests.append(
                {
                    "roi": roi,
                    "base_roi": off["base_roi"].iloc[0],
                    "anatomical_system": off["anatomical_system"].iloc[0],
                    "metric": metric_name,
                    "mean_off": float(merged[f"{metric_name}_off"].mean()),
                    "mean_on": float(merged[f"{metric_name}_on"].mean()),
                    "p_primary": float(stats_dict["p_signflip"]),
                    "primary_test": "subject_mean_signflip",
                    **stats_dict,
                }
            )
            for subject, value in zip(merged["subject"], delta):
                delta_rows.append(
                    {
                        "subject": subject,
                        "roi": roi,
                        "base_roi": off["base_roi"].iloc[0],
                        "anatomical_system": off["anatomical_system"].iloc[0],
                        f"delta_{metric_name}": float(value),
                    }
                )

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = (
            delta_df.groupby(["subject", "roi", "base_roi", "anatomical_system"], as_index=False)
            .first()
            .sort_values(["subject", "roi"])
        )
    return pd.DataFrame(node_tests), delta_df


def _fit_repeated_state_model(metric_df: pd.DataFrame) -> dict[str, object]:
    model_df = metric_df.copy()
    model_df["state_num"] = (model_df["session"] == 2).astype(float)
    model_df["run_label"] = model_df["run"].astype(str)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(
                "value ~ state_num + C(run_label)",
                data=model_df,
                groups=model_df["subject"],
            ).fit(reml=False, method="lbfgs", disp=False)
        p_value = float(fit.pvalues.get("state_num", np.nan))
        estimate = float(fit.params.get("state_num", np.nan))
        if np.isfinite(p_value):
            return {
                "p_primary": p_value,
                "primary_estimate": estimate,
                "primary_test": "mixedlm_subject_intercept_run_fixed",
            }
    except Exception:
        pass

    fit = smf.ols(
        "value ~ state_num + C(run_label) + C(subject)",
        data=model_df,
    ).fit(cov_type="cluster", cov_kwds={"groups": model_df["subject"]})
    return {
        "p_primary": float(fit.pvalues.get("state_num", np.nan)),
        "primary_estimate": float(fit.params.get("state_num", np.nan)),
        "primary_test": "ols_subject_fixed_effects_clustered_by_subject",
    }


def _apply_run_level_primary_tests(
    node_test_df: pd.DataFrame,
    run_node_df: pd.DataFrame,
) -> pd.DataFrame:
    if run_node_df.empty:
        return node_test_df

    updated = node_test_df.copy()
    for idx, row in updated.iterrows():
        metric_df = run_node_df.loc[
            run_node_df["roi"].eq(row["roi"]),
            ["subject", "session", "run", row["metric"]],
        ].rename(columns={row["metric"]: "value"})
        if metric_df.empty or metric_df["value"].isna().all():
            continue
        repeated_stats = _fit_repeated_state_model(metric_df.dropna())
        if np.isfinite(float(repeated_stats["p_primary"])):
            updated.loc[idx, "p_primary"] = float(repeated_stats["p_primary"])
            updated.loc[idx, "primary_estimate"] = float(repeated_stats["primary_estimate"])
            updated.loc[idx, "primary_test"] = str(repeated_stats["primary_test"])
            updated.loc[idx, "n_runs"] = int(metric_df["run"].nunique())
            updated.loc[idx, "n_observations_primary"] = int(len(metric_df))
    return updated


def _build_run_level_node_metrics(
    metrics_root: Path,
    metric: str,
    drop_patterns,
    base_labels: list[str],
    consensus_labels_arr: np.ndarray,
    required_subjects: list[str],
) -> pd.DataFrame:
    run_rows: list[dict] = []
    for label in _list_complete_run_labels(metrics_root, metric, required_subjects):
        match = RUN_LABEL_RE.match(label)
        if match is None:
            continue
        subject = str(match.group(1))
        session = int(match.group(2))
        run = int(match.group(3))
        graph = _load_run_level_graph(
            metrics_root=metrics_root,
            label=label,
            metric=metric,
            drop_patterns=drop_patterns,
            include_base_rois=base_labels,
        )
        _append_node_rows(
            run_rows,
            graph,
            subject=subject,
            session=session,
            run=run,
            consensus_labels_arr=consensus_labels_arr,
        )
    return pd.DataFrame(run_rows)


def run_base_roi_graph_reorganization(
    out_dir: Path,
    metric: str = "partial_correlation",
    drop_patterns=DEFAULT_DROP_LABEL_PATTERNS,
    run_level_metrics_root: Path | None = DEFAULT_RUN_LEVEL_METRICS_ROOT,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    subjects = list_paired_subjects_for_metric(metric)

    grand_positive = []
    node_rows = []
    community_rows = []
    module_rows = []
    subject_partition_meta = []
    labels_after_drop: list[str] | None = None

    prepared_graphs: dict[tuple[str, int], dict[str, object]] = {}
    for subject in subjects:
        for session in (1, 2):
            graph = _prepare_session_graph(subject, session, metric, drop_patterns)
            labels = list(graph["labels"])
            positive = np.asarray(graph["positive"])
            grand_positive.append(positive)
            labels_after_drop = labels
            prepared_graphs[(subject, session)] = graph

    mean_positive = np.mean(np.stack(grand_positive, axis=0), axis=0)
    consensus_labels_arr, consensus_q, _ = consensus_partition(mean_positive)
    consensus_assignments = {
        label: int(community + 1)
        for label, community in zip(labels_after_drop or [], consensus_labels_arr)
    }

    for subject in subjects:
        for session in (1, 2):
            graph = prepared_graphs[(subject, session)]
            labels = list(graph["labels"])
            positive = np.asarray(graph["positive"])
            absolute = np.asarray(graph["absolute"])

            subject_partition, subject_q, _ = consensus_partition(positive)
            ari_to_consensus = float(adjusted_rand_score(consensus_labels_arr, subject_partition))
            subject_partition_meta.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "state": "OFF" if int(session) == 1 else "ON",
                    "subject_modularity_q": float(subject_q),
                    "consensus_modularity_q": float(weighted_modularity(positive, consensus_labels_arr)),
                    "ari_to_consensus": ari_to_consensus,
                }
            )

            _append_node_rows(
                node_rows,
                graph,
                subject=subject,
                session=session,
                run=None,
                consensus_labels_arr=consensus_labels_arr,
            )
            for idx, roi in enumerate(labels):
                community_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "state": "OFF" if int(session) == 1 else "ON",
                        "roi": roi,
                        "consensus_community": int(consensus_labels_arr[idx] + 1),
                        "subject_specific_community": int(subject_partition[idx] + 1),
                        "ari_to_consensus": ari_to_consensus,
                    }
                )

            anatomical_group_map = {
                roi: infer_anatomical_system(roi)
                for roi in labels
            }
            module_rows.extend(
                _compute_group_pair_rows(
                    absolute,
                    labels,
                    anatomical_group_map,
                    subject,
                    session,
                    "anatomical_system",
                )
            )
            community_group_map = {
                label: f"C{int(consensus_labels_arr[idx]) + 1}"
                for idx, label in enumerate(labels)
            }
            module_rows.extend(
                _compute_group_pair_rows(
                    absolute,
                    labels,
                    community_group_map,
                    subject,
                    session,
                    "consensus_community",
                )
            )

    node_df = pd.DataFrame(node_rows)
    community_df = pd.DataFrame(community_rows)
    module_df = pd.DataFrame(module_rows)
    partition_meta_df = pd.DataFrame(subject_partition_meta)

    node_test_df, delta_df = _build_subject_level_deltas(node_df)

    run_node_df = pd.DataFrame()
    if run_level_metrics_root is not None:
        metrics_root = Path(run_level_metrics_root)
        if metrics_root.exists():
            run_node_df = _build_run_level_node_metrics(
                metrics_root=metrics_root,
                metric=metric,
                drop_patterns=drop_patterns,
                base_labels=list(labels_after_drop or []),
                consensus_labels_arr=consensus_labels_arr,
                required_subjects=subjects,
            )
            if not run_node_df.empty:
                node_test_df = _apply_run_level_primary_tests(node_test_df, run_node_df)

    if "p_primary" not in node_test_df.columns:
        node_test_df["p_primary"] = node_test_df["p_signflip"]
    if "primary_test" not in node_test_df.columns:
        node_test_df["primary_test"] = "subject_mean_signflip"
    node_test_df["q_fdr_within_metric"] = np.nan
    for metric_name, metric_df in node_test_df.groupby("metric"):
        q_values = bh_fdr(metric_df["p_primary"])
        node_test_df.loc[metric_df.index, "q_fdr_within_metric"] = q_values
    node_test_df = node_test_df.sort_values(
        ["metric", "q_fdr_within_metric", "p_primary", "mean_delta"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    hub_summary = (
        node_df.groupby(["roi", "base_roi", "anatomical_system", "state"], as_index=False)
        .agg(
            mean_node_strength_abs=("node_strength_abs", "mean"),
            mean_participation_coeff=("participation_coeff", "mean"),
            mean_within_module_z=("within_module_z", "mean"),
            connector_hub_fraction=("hub_class", lambda s: float((s == "connector_hub").mean())),
            provincial_hub_fraction=("hub_class", lambda s: float((s == "provincial_hub").mean())),
        )
        .pivot(
            index=["roi", "base_roi", "anatomical_system"],
            columns="state",
            values=[
                "mean_node_strength_abs",
                "mean_participation_coeff",
                "mean_within_module_z",
                "connector_hub_fraction",
                "provincial_hub_fraction",
            ],
        )
    )
    hub_summary.columns = [
        f"{metric_name.lower()}_{state.lower()}" for metric_name, state in hub_summary.columns
    ]
    hub_summary = hub_summary.reset_index()
    if {"mean_node_strength_abs_off", "mean_node_strength_abs_on"} <= set(hub_summary.columns):
        hub_summary["mean_node_strength_abs_delta"] = (
            hub_summary["mean_node_strength_abs_on"] - hub_summary["mean_node_strength_abs_off"]
        )
        hub_summary["mean_participation_coeff_delta"] = (
            hub_summary["mean_participation_coeff_on"] - hub_summary["mean_participation_coeff_off"]
        )
        hub_summary["mean_within_module_z_delta"] = (
            hub_summary["mean_within_module_z_on"] - hub_summary["mean_within_module_z_off"]
        )

    node_metrics_path = out_dir / "node_metrics_by_subject_session.csv"
    run_node_metrics_path = out_dir / "node_metrics_by_subject_session_run.csv"
    delta_path = out_dir / "node_metric_deltas_on_minus_off.csv"
    test_path = out_dir / "node_metric_tests_fdr.csv"
    community_path = out_dir / "community_assignments_by_subject_session.csv"
    module_path = out_dir / "module_integration_summary.csv"
    hub_path = out_dir / "hub_summary.csv"
    summary_path = out_dir / "community_consensus_summary.json"
    hub_fig_path = out_dir / "hub_reorganization.png"
    module_fig_path = out_dir / "module_delta_heatmap.png"

    node_df.to_csv(node_metrics_path, index=False)
    if not run_node_df.empty:
        run_node_df.to_csv(run_node_metrics_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    node_test_df.to_csv(test_path, index=False)
    community_df.to_csv(community_path, index=False)
    module_df.to_csv(module_path, index=False)
    hub_summary.to_csv(hub_path, index=False)
    _plot_hub_reorganization(node_test_df, hub_fig_path)
    _plot_module_delta_heatmap(module_df, module_fig_path)

    primary_test_counts = (
        node_test_df["primary_test"].value_counts(dropna=False).to_dict()
        if not node_test_df.empty
        else {}
    )
    summary = {
        "metric": metric,
        "analysis_level": "base_roi",
        "aggregate_hemispheres": True,
        "n_subjects": len(subjects),
        "n_base_rois": len(labels_after_drop or []),
        "dropped_label_patterns": list(drop_patterns),
        "consensus_n_communities": int(len(np.unique(consensus_labels_arr))),
        "consensus_modularity_q": float(consensus_q),
        "mean_subject_specific_modularity_q": float(partition_meta_df["subject_modularity_q"].mean()),
        "mean_ari_to_consensus": float(partition_meta_df["ari_to_consensus"].mean()),
        "run_level_metrics_root": str(run_level_metrics_root) if run_level_metrics_root is not None else None,
        "primary_test_counts": to_serializable(primary_test_counts),
        "consensus_communities": {
            label: int(community)
            for label, community in consensus_assignments.items()
        },
        "top_node_effects": to_serializable(
            node_test_df.assign(abs_mean_delta=node_test_df["mean_delta"].abs())
            .sort_values(["q_fdr_within_metric", "abs_mean_delta"], ascending=[True, False])
            .head(12)[["roi", "metric", "mean_delta", "p_primary", "q_fdr_within_metric", "primary_test"]]
            .to_dict(orient="records")
        ),
    }
    write_json(summary_path, summary)

    return {
        "subjects": subjects,
        "metric": metric,
        "analysis_level": "base_roi",
        "node_metrics_path": node_metrics_path,
        "run_node_metrics_path": run_node_metrics_path if not run_node_df.empty else None,
        "node_delta_path": delta_path,
        "node_tests_path": test_path,
        "community_assignments_path": community_path,
        "module_summary_path": module_path,
        "hub_summary_path": hub_path,
        "summary_path": summary_path,
        "hub_figure_path": hub_fig_path,
        "module_figure_path": module_fig_path,
    }
