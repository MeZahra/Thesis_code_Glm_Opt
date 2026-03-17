from __future__ import annotations

from collections import Counter
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings(
    "ignore",
    message="Graph is not fully connected, spectral embedding may not work as expected.",
)

from common_io import (
    ANATOMICAL_SYSTEM_ORDER,
    DEFAULT_DROP_LABEL_PATTERNS,
    PRIMARY_METRIC,
    aggregate_matrix_by_base_roi,
    base_roi_name,
    bh_fdr,
    drop_labels_and_matrix,
    ensure_dir,
    infer_anatomical_system,
    list_paired_subjects_for_metric,
    load_metric_matrix,
    paired_delta_stats,
    safe_slug,
    sanitize_matrix,
    to_serializable,
    write_json,
)


def weighted_modularity(weights: np.ndarray, communities: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    m = weights.sum() / 2.0
    if m <= 0.0:
        return 0.0
    degree = weights.sum(axis=1)
    modularity = 0.0
    for idx in range(weights.shape[0]):
        for jdx in range(weights.shape[1]):
            if communities[idx] != communities[jdx]:
                continue
            modularity += weights[idx, jdx] - degree[idx] * degree[jdx] / (2.0 * m)
    return float(modularity / (2.0 * m))


def consensus_partition(
    weights: np.ndarray,
    n_repeats: int = 20,
    k_min: int = 2,
    k_max: int = 6,
) -> tuple[np.ndarray, float, list[dict]]:
    n_nodes = weights.shape[0]
    if n_nodes < 3 or np.allclose(weights, 0.0):
        communities = np.ones(n_nodes, dtype=int)
        return communities, 0.0, []

    candidates: list[dict] = []
    affinity = weights.copy()
    affinity += np.eye(n_nodes) * max(float(np.nanmean(weights)) * 1e-6, 1e-6)
    max_clusters = max(k_min, min(k_max, n_nodes - 1))
    for k in range(k_min, max_clusters + 1):
        for seed in range(n_repeats):
            try:
                model = SpectralClustering(
                    n_clusters=k,
                    affinity="precomputed",
                    assign_labels="kmeans",
                    random_state=seed,
                    n_init=10,
                )
                labels = model.fit_predict(affinity)
            except Exception:
                continue
            score = weighted_modularity(weights, labels)
            candidates.append({"labels": labels, "modularity_q": score, "n_clusters": k})

    if not candidates:
        communities = np.ones(n_nodes, dtype=int)
        return communities, 0.0, []

    candidates = sorted(candidates, key=lambda item: item["modularity_q"], reverse=True)
    top_n = max(5, min(25, len(candidates)))
    top_candidates = candidates[:top_n]
    cluster_counts = [item["n_clusters"] for item in top_candidates]
    consensus_k = Counter(cluster_counts).most_common(1)[0][0]
    coassign = np.zeros((n_nodes, n_nodes), dtype=float)
    for item in top_candidates:
        labels = item["labels"]
        coassign += (labels[:, None] == labels[None, :]).astype(float)
    coassign /= float(len(top_candidates))
    coassign += np.eye(n_nodes) * 1e-6
    model = SpectralClustering(
        n_clusters=consensus_k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
        n_init=20,
    )
    communities = model.fit_predict(coassign)
    return communities, weighted_modularity(weights, communities), top_candidates


def participation_coefficient(weights: np.ndarray, communities: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    communities = np.asarray(communities)
    strength = weights.sum(axis=1)
    output = np.zeros(weights.shape[0], dtype=float)
    unique_communities = np.unique(communities)
    for idx in range(weights.shape[0]):
        if strength[idx] <= 0.0:
            continue
        same_strength_sq = 0.0
        for community in unique_communities:
            members = communities == community
            community_strength = weights[idx, members].sum()
            same_strength_sq += (community_strength / strength[idx]) ** 2
        output[idx] = 1.0 - same_strength_sq
    return output


def within_module_degree_z(weights: np.ndarray, communities: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    communities = np.asarray(communities)
    output = np.zeros(weights.shape[0], dtype=float)
    for community in np.unique(communities):
        members = np.where(communities == community)[0]
        if members.size == 0:
            continue
        within_strength = weights[np.ix_(members, members)].sum(axis=1)
        mean_strength = float(np.mean(within_strength))
        std_strength = float(np.std(within_strength, ddof=0))
        if std_strength == 0.0:
            output[members] = 0.0
        else:
            output[members] = (within_strength - mean_strength) / std_strength
    return output


def classify_hub(participation: float, within_z: float) -> str:
    if within_z >= 1.0 and participation >= 0.30:
        return "connector_hub"
    if within_z >= 1.0:
        return "provincial_hub"
    return "non_hub"


def _compute_group_pair_rows(
    matrix: np.ndarray,
    labels: list[str],
    groups: dict[str, str],
    subject: str,
    session: int,
    scheme_name: str,
) -> list[dict]:
    rows = []
    unique_groups = [group for group in ANATOMICAL_SYSTEM_ORDER if group in groups.values()]
    if scheme_name == "consensus_community":
        unique_groups = sorted(set(groups.values()), key=lambda item: int(item[1:]))

    index_by_group: dict[str, list[int]] = {group: [] for group in unique_groups}
    for idx, label in enumerate(labels):
        group = groups.get(label)
        if group in index_by_group:
            index_by_group[group].append(idx)

    for left_idx, group_a in enumerate(unique_groups):
        for group_b in unique_groups[left_idx:]:
            idx_a = index_by_group.get(group_a, [])
            idx_b = index_by_group.get(group_b, [])
            if not idx_a or not idx_b:
                continue
            if group_a == group_b:
                block = matrix[np.ix_(idx_a, idx_a)]
                values = block[np.triu_indices_from(block, k=1)]
            else:
                values = matrix[np.ix_(idx_a, idx_b)].ravel()
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "state": "OFF" if int(session) == 1 else "ON",
                    "module_scheme": scheme_name,
                    "module_a": group_a,
                    "module_b": group_b,
                    "mean_abs_strength": float(np.mean(values)),
                    "edge_count": int(values.size),
                }
            )
    return rows


def _plot_hub_reorganization(node_tests: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, metric in zip(axes, ["node_strength_abs", "participation_coeff"]):
        metric_df = node_tests[node_tests["metric"] == metric].copy()
        metric_df["abs_mean_delta"] = metric_df["mean_delta"].abs()
        metric_df = metric_df.sort_values("abs_mean_delta", ascending=False).head(12)
        ax.barh(metric_df["roi"], metric_df["mean_delta"], color="#4c78a8")
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Mean ON - OFF")
        ax.invert_yaxis()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_module_delta_heatmap(module_df: pd.DataFrame, out_path: Path) -> None:
    anatomical = module_df[module_df["module_scheme"] == "anatomical_system"].copy()
    if anatomical.empty:
        return
    off = (
        anatomical[anatomical["session"] == 1]
        .groupby(["module_a", "module_b"])["mean_abs_strength"]
        .mean()
    )
    on = (
        anatomical[anatomical["session"] == 2]
        .groupby(["module_a", "module_b"])["mean_abs_strength"]
        .mean()
    )
    delta = (on - off).reset_index(name="delta")
    labels = [label for label in ANATOMICAL_SYSTEM_ORDER if label in set(delta["module_a"]) | set(delta["module_b"])]
    matrix = np.full((len(labels), len(labels)), np.nan, dtype=float)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    for _, row in delta.iterrows():
        left = label_to_idx[row["module_a"]]
        right = label_to_idx[row["module_b"]]
        matrix[left, right] = row["delta"]
        matrix[right, left] = row["delta"]

    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    vmax = float(np.nanmax(np.abs(matrix))) or 1.0
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean Anatomical-System Delta (ON - OFF)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_roi_graph_reorganization(
    out_dir: Path,
    metric: str = PRIMARY_METRIC,
    drop_patterns=DEFAULT_DROP_LABEL_PATTERNS,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    subjects = list_paired_subjects_for_metric(metric)

    grand_positive = []
    node_rows = []
    community_rows = []
    module_rows = []
    subject_partition_meta = []
    labels_after_drop = None

    prepared_graphs: dict[tuple[str, int], dict[str, np.ndarray | list[str]]] = {}
    for subject in subjects:
        for session in (1, 2):
            raw_matrix, raw_labels = load_metric_matrix(subject, session, metric)
            trimmed_matrix, trimmed_labels, _ = drop_labels_and_matrix(
                raw_labels, raw_matrix, drop_patterns
            )
            adjacency = sanitize_matrix(trimmed_matrix)
            positive = np.clip(adjacency, 0.0, None)
            absolute = np.abs(adjacency)
            labels_after_drop = trimmed_labels
            grand_positive.append(positive)
            prepared_graphs[(subject, session)] = {
                "labels": trimmed_labels,
                "adjacency": adjacency,
                "positive": positive,
                "absolute": absolute,
            }

    mean_positive = np.mean(np.stack(grand_positive, axis=0), axis=0)
    consensus_labels_arr, consensus_q, consensus_candidates = consensus_partition(mean_positive)
    consensus_assignments = {
        label: int(community + 1)
        for label, community in zip(labels_after_drop, consensus_labels_arr)
    }

    for subject in subjects:
        for session in (1, 2):
            graph = prepared_graphs[(subject, session)]
            labels = graph["labels"]
            positive = graph["positive"]
            absolute = graph["absolute"]
            adjacency = graph["adjacency"]

            subject_partition, subject_q, _ = consensus_partition(positive)
            ari_to_consensus = float(
                adjusted_rand_score(consensus_labels_arr, subject_partition)
            )
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

            node_strength_abs = absolute.sum(axis=1)
            node_strength_pos = positive.sum(axis=1)
            participation = participation_coefficient(positive, consensus_labels_arr)
            within_z = within_module_degree_z(positive, consensus_labels_arr)

            for idx, roi in enumerate(labels):
                node_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "state": "OFF" if int(session) == 1 else "ON",
                        "roi": roi,
                        "base_roi": base_roi_name(roi),
                        "anatomical_system": infer_anatomical_system(base_roi_name(roi)),
                        "node_strength_abs": float(node_strength_abs[idx]),
                        "node_strength_positive": float(node_strength_pos[idx]),
                        "participation_coeff": float(participation[idx]),
                        "within_module_z": float(within_z[idx]),
                        "consensus_community": int(consensus_labels_arr[idx] + 1),
                        "subject_specific_community": int(subject_partition[idx] + 1),
                        "ari_to_consensus": ari_to_consensus,
                        "subject_modularity_q": float(subject_q),
                        "consensus_modularity_q": float(weighted_modularity(positive, consensus_labels_arr)),
                        "hub_class": classify_hub(float(participation[idx]), float(within_z[idx])),
                    }
                )
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

            anatomical_groups = {
                base: infer_anatomical_system(base) for base in sorted({base_roi_name(label) for label in labels})
            }
            absolute_base, base_labels, _ = aggregate_matrix_by_base_roi(absolute, labels)
            anatomical_group_map = {
                label: anatomical_groups[label] for label in base_labels
            }
            module_rows.extend(
                _compute_group_pair_rows(
                    absolute_base,
                    base_labels,
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

    delta_rows = []
    node_tests = []
    for roi, roi_df in node_df.groupby("roi", sort=True):
        off = roi_df[roi_df["session"] == 1].sort_values("subject")
        on = roi_df[roi_df["session"] == 2].sort_values("subject")
        merged = off.merge(
            on,
            on="subject",
            suffixes=("_off", "_on"),
        )
        if merged.empty:
            continue
        for metric_name in [
            "node_strength_abs",
            "node_strength_positive",
            "participation_coeff",
            "within_module_z",
        ]:
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

    node_test_df = pd.DataFrame(node_tests)
    node_test_df["q_fdr_within_metric"] = np.nan
    for metric_name, metric_df in node_test_df.groupby("metric"):
        q_values = bh_fdr(metric_df["p_signflip"])
        node_test_df.loc[metric_df.index, "q_fdr_within_metric"] = q_values
    node_test_df = node_test_df.sort_values(
        ["metric", "q_fdr_within_metric", "p_signflip", "mean_delta"],
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
    delta_path = out_dir / "node_metric_deltas_on_minus_off.csv"
    test_path = out_dir / "node_metric_tests_fdr.csv"
    community_path = out_dir / "community_assignments_by_subject_session.csv"
    module_path = out_dir / "module_integration_summary.csv"
    hub_path = out_dir / "hub_summary.csv"
    summary_path = out_dir / "community_consensus_summary.json"
    hub_fig_path = out_dir / "hub_reorganization.png"
    module_fig_path = out_dir / "module_delta_heatmap.png"

    node_df.to_csv(node_metrics_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    node_test_df.to_csv(test_path, index=False)
    community_df.to_csv(community_path, index=False)
    module_df.to_csv(module_path, index=False)
    hub_summary.to_csv(hub_path, index=False)
    _plot_hub_reorganization(node_test_df, hub_fig_path)
    _plot_module_delta_heatmap(module_df, module_fig_path)

    summary = {
        "metric": metric,
        "n_subjects": len(subjects),
        "n_nodes_after_exclusion": len(labels_after_drop or []),
        "dropped_label_patterns": list(drop_patterns),
        "consensus_n_communities": int(len(np.unique(consensus_labels_arr))),
        "consensus_modularity_q": float(consensus_q),
        "mean_subject_specific_modularity_q": float(partition_meta_df["subject_modularity_q"].mean()),
        "mean_ari_to_consensus": float(partition_meta_df["ari_to_consensus"].mean()),
        "consensus_communities": {
            label: int(community)
            for label, community in consensus_assignments.items()
        },
        "top_node_effects": to_serializable(
            node_test_df.assign(abs_mean_delta=node_test_df["mean_delta"].abs())
            .sort_values(["q_fdr_within_metric", "abs_mean_delta"], ascending=[True, False])
            .head(12)[["roi", "metric", "mean_delta", "p_signflip", "q_fdr_within_metric"]]
            .to_dict(orient="records")
        ),
    }
    write_json(summary_path, summary)

    return {
        "subjects": subjects,
        "metric": metric,
        "node_metrics_path": node_metrics_path,
        "node_delta_path": delta_path,
        "node_tests_path": test_path,
        "community_assignments_path": community_path,
        "module_summary_path": module_path,
        "hub_summary_path": hub_path,
        "summary_path": summary_path,
        "hub_figure_path": hub_fig_path,
        "module_figure_path": module_fig_path,
    }
