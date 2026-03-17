from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

from common_io import (
    ANATOMICAL_SYSTEM_ORDER,
    CIRCUIT_BASE_ROIS,
    TASK_BEHAVIOR_COLUMN_SPECS,
    aggregate_matrix_by_base_roi,
    base_roi_name,
    ensure_dir,
    list_paired_dcm_subjects,
    load_behavior_deltas,
    load_dcm_labels,
    load_subject_dcm_matrix,
    load_task_behavior_deltas,
    safe_slug,
    to_serializable,
    write_json,
)

LEGACY_MODULE_PAIRS = [
    ("cognitive_control", "cognitive_control"),
    ("cognitive_control", "subcortical_relay"),
    ("motor_sensorimotor", "subcortical_relay"),
]


def _zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    centered = df - df.mean(axis=0)
    scale = df.std(axis=0, ddof=0).replace(0.0, 1.0)
    return centered / scale


def _align_vector_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if np.dot(reference, candidate) < 0:
        return -candidate
    return candidate


def _analysis_output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "neural_wide_path": out_dir / "subject_neural_deltas.csv",
        "behavior_path": out_dir / "subject_behavior_deltas.csv",
        "summary_path": out_dir / "behavioral_pls_summary.json",
        "loading_path": out_dir / "behavioral_pls_loadings.csv",
        "score_path": out_dir / "behavioral_pls_scores.csv",
        "permutation_path": out_dir / "behavioral_pls_permutation.csv",
        "loo_path": out_dir / "behavioral_pls_loo_stability.csv",
        "figure_path": out_dir / "brain_behavior_coupling.png",
        "behavior_weight_path": out_dir / "behavioral_pls_behavior_weights.csv",
    }


def _correlation_statistic(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: str,
) -> float:
    if method == "pearson":
        return float(stats.pearsonr(x, y).statistic)
    if method == "spearman":
        return float(stats.spearmanr(x, y).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")


def _plot_behavior_coupling(
    scores_df: pd.DataFrame,
    loadings_df: pd.DataFrame,
    out_path: Path,
    scatter_col: str,
    scatter_label: str,
    scatter_title: str,
    scatter_corr_method: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].scatter(scores_df["x_score"], scores_df[scatter_col], color="#4c78a8", s=55)
    corr = _correlation_statistic(
        scores_df["x_score"], scores_df[scatter_col], method=scatter_corr_method
    )
    corr_label = "r" if scatter_corr_method == "pearson" else "rho"
    axes[0].set_title(f"{scatter_title} ({corr_label}={corr:.2f})")
    axes[0].set_xlabel("PLS brain score")
    axes[0].set_ylabel(scatter_label)
    for _, row in scores_df.iterrows():
        axes[0].text(row["x_score"], row[scatter_col], row["subject"], fontsize=7)

    top_loadings = (
        loadings_df[loadings_df["set"] == "X"]
        .assign(abs_weight=lambda df: df["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .head(12)
    )
    axes[1].barh(top_loadings["name"], top_loadings["weight"], color="#f58518")
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].invert_yaxis()
    axes[1].set_title("Top Neural Weights")
    axes[1].set_xlabel("PLS weight")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _build_dcm_features(
    dcm_subject_df: pd.DataFrame,
    base_rois: list[str],
) -> pd.DataFrame:
    edge_df = dcm_subject_df[dcm_subject_df["parameter_type"] == "coupling"].copy()
    rows = []
    for subject, group in edge_df.groupby("subject", sort=True):
        row = {"subject": subject}
        pivot = group.pivot(index="target_roi", columns="source_roi", values="delta_on_minus_off")
        for roi in base_rois:
            feature_name = f"dcm_outgoing_delta_{safe_slug(roi)}"
            if roi not in pivot.index or roi not in pivot.columns:
                row[feature_name] = float("nan")
                continue
            outgoing = pivot.loc[pivot.index != roi, roi].mean()
            row[feature_name] = float(outgoing)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def _build_dcm_features_from_raw(base_rois: list[str] | None = None) -> pd.DataFrame:
    labels = load_dcm_labels()
    resolved_base_rois = (
        list(base_rois)
        if base_rois is not None
        else sorted({base_roi_name(label) for label in labels})
    )
    rows = []
    for subject in list_paired_dcm_subjects():
        delta_full = load_subject_dcm_matrix(subject, "dcm_session2_minus_session1.npy")
        aggregated, aggregated_labels, _ = aggregate_matrix_by_base_roi(
            delta_full,
            labels,
            include_base_rois=resolved_base_rois,
        )
        label_to_idx = {label: idx for idx, label in enumerate(aggregated_labels)}
        row = {"subject": subject}
        for roi in resolved_base_rois:
            feature_name = f"dcm_outgoing_delta_{safe_slug(roi)}"
            idx = label_to_idx.get(roi)
            if idx is None:
                row[feature_name] = float("nan")
                continue
            keep_mask = np.ones(len(aggregated_labels), dtype=bool)
            keep_mask[idx] = False
            outgoing = aggregated[keep_mask, idx]
            row[feature_name] = float(np.nanmean(outgoing)) if outgoing.size else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def _build_graph_features(
    node_delta_df: pd.DataFrame,
    base_rois: list[str] | None = None,
) -> pd.DataFrame:
    resolved_base_rois = (
        list(base_rois)
        if base_rois is not None
        else sorted(node_delta_df["base_roi"].dropna().unique().tolist())
    )
    rows = []
    for subject, group in node_delta_df.groupby("subject", sort=True):
        row = {"subject": subject}
        grouped = (
            group.groupby("base_roi", as_index=False)
            .agg(
                delta_node_strength_abs=("delta_node_strength_abs", "mean"),
                delta_participation_coeff=("delta_participation_coeff", "mean"),
            )
            .set_index("base_roi")
        )
        for base_roi in resolved_base_rois:
            strength_name = f"graph_strength_delta_{safe_slug(base_roi)}"
            participation_name = f"graph_participation_delta_{safe_slug(base_roi)}"
            if base_roi not in grouped.index:
                row[strength_name] = float("nan")
                row[participation_name] = float("nan")
                continue
            row[strength_name] = float(grouped.loc[base_roi, "delta_node_strength_abs"])
            row[participation_name] = float(
                grouped.loc[base_roi, "delta_participation_coeff"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def _resolve_module_pairs(
    module_df: pd.DataFrame,
    selected_pairs: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if selected_pairs is not None:
        return list(selected_pairs)
    anatomical = module_df[module_df["module_scheme"] == "anatomical_system"].copy()
    unique_pairs = [
        tuple(item)
        for item in anatomical[["module_a", "module_b"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ]
    order = {name: idx for idx, name in enumerate(ANATOMICAL_SYSTEM_ORDER)}
    return sorted(
        unique_pairs,
        key=lambda pair: (
            order.get(pair[0], len(order)),
            order.get(pair[1], len(order)),
            pair[0],
            pair[1],
        ),
    )


def _build_module_features(
    module_df: pd.DataFrame,
    selected_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    anatomical = module_df[module_df["module_scheme"] == "anatomical_system"].copy()
    resolved_pairs = _resolve_module_pairs(anatomical, selected_pairs=selected_pairs)
    rows = []
    for subject, group in anatomical.groupby("subject", sort=True):
        row = {"subject": subject}
        for module_a, module_b in resolved_pairs:
            feature_name = f"module_delta_{safe_slug(module_a)}__{safe_slug(module_b)}"
            off = group[
                (group["session"] == 1)
                & (group["module_a"] == module_a)
                & (group["module_b"] == module_b)
            ]
            on = group[
                (group["session"] == 2)
                & (group["module_a"] == module_a)
                & (group["module_b"] == module_b)
            ]
            if off.empty or on.empty:
                row[feature_name] = float("nan")
                continue
            row[feature_name] = float(
                on["mean_abs_strength"].iloc[0] - off["mean_abs_strength"].iloc[0]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def _run_pls_analysis(
    out_dir: Path,
    merged: pd.DataFrame,
    feature_cols: list[str],
    behavior_cols: list[str],
    n_permutations: int,
    scatter_col: str,
    scatter_label: str,
    scatter_title: str,
    scatter_corr_method: str,
    behavior_label_map: dict[str, str] | None = None,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    outputs = _analysis_output_paths(out_dir)

    merged = merged.dropna(subset=feature_cols + behavior_cols).sort_values("subject").reset_index(
        drop=True
    )
    if merged.empty:
        raise ValueError(f"No subjects remained for PLS analysis in {out_dir}.")

    neural_df = merged[["subject"] + feature_cols].copy()
    behavior_export = merged[[col for col in merged.columns if col not in feature_cols]].copy()
    neural_df.to_csv(outputs["neural_wide_path"], index=False)
    behavior_export.to_csv(outputs["behavior_path"], index=False)

    X = _zscore_frame(merged[feature_cols])
    Y = _zscore_frame(merged[behavior_cols])

    model = PLSRegression(n_components=1)
    model.fit(X, Y)
    x_scores = model.x_scores_.ravel()
    y_scores = model.y_scores_.ravel()
    observed_score_corr = float(stats.pearsonr(x_scores, y_scores).statistic)

    rng = np.random.default_rng(0)
    permutation_rows = []
    for permutation_id in range(n_permutations):
        permuted_Y = Y.iloc[rng.permutation(len(Y))].reset_index(drop=True)
        perm_model = PLSRegression(n_components=1)
        perm_model.fit(X, permuted_Y)
        perm_corr = float(
            stats.pearsonr(perm_model.x_scores_.ravel(), perm_model.y_scores_.ravel()).statistic
        )
        permutation_rows.append(
            {"permutation_id": permutation_id, "score_correlation": perm_corr}
        )
    permutation_df = pd.DataFrame(permutation_rows)
    permutation_p = float(
        (np.sum(np.abs(permutation_df["score_correlation"]) >= abs(observed_score_corr)) + 1)
        / (len(permutation_df) + 1)
    )

    loadings_rows = []
    for name, weight in zip(feature_cols, model.x_weights_.ravel()):
        loadings_rows.append({"set": "X", "name": name, "weight": float(weight)})
    for name, weight in zip(behavior_cols, model.y_weights_.ravel()):
        loadings_rows.append({"set": "Y", "name": name, "weight": float(weight)})
    loadings_df = pd.DataFrame(loadings_rows)

    score_df = merged[["subject"] + behavior_cols].copy()
    score_df["x_score"] = x_scores
    score_df["y_score"] = y_scores

    full_weights = model.x_weights_.ravel()
    loo_rows = []
    for left_out in range(len(merged)):
        train_mask = np.ones(len(merged), dtype=bool)
        train_mask[left_out] = False
        loo_model = PLSRegression(n_components=1)
        loo_model.fit(X.iloc[train_mask], Y.iloc[train_mask])
        loo_weights = _align_vector_sign(full_weights, loo_model.x_weights_.ravel())
        train_corr = float(
            stats.pearsonr(loo_model.x_scores_.ravel(), loo_model.y_scores_.ravel()).statistic
        )
        predicted = loo_model.predict(X.iloc[[left_out]])[0]
        row = {
            "left_out_subject": merged.loc[left_out, "subject"],
            "weights_corr_to_full": float(np.corrcoef(full_weights, loo_weights)[0, 1]),
            "train_score_correlation": train_corr,
        }
        for idx, name in enumerate(behavior_cols):
            row[f"predicted_{name}"] = float(predicted[idx])
            row[f"actual_{name}"] = float(merged.loc[left_out, name])
        loo_rows.append(row)
    loo_df = pd.DataFrame(loo_rows)

    behavior_weight_df = (
        loadings_df[loadings_df["set"] == "Y"]
        .assign(abs_weight=lambda df: df["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )

    loadings_df.to_csv(outputs["loading_path"], index=False)
    score_df.to_csv(outputs["score_path"], index=False)
    permutation_df.to_csv(outputs["permutation_path"], index=False)
    loo_df.to_csv(outputs["loo_path"], index=False)
    behavior_weight_df.to_csv(outputs["behavior_weight_path"], index=False)
    _plot_behavior_coupling(
        score_df,
        loadings_df,
        outputs["figure_path"],
        scatter_col=scatter_col,
        scatter_label=scatter_label,
        scatter_title=scatter_title,
        scatter_corr_method=scatter_corr_method,
    )

    neural_weight_df = (
        loadings_df[loadings_df["set"] == "X"]
        .assign(abs_weight=lambda df: df["weight"].abs())
        .sort_values("abs_weight", ascending=False)
        .reset_index(drop=True)
    )
    summary = {
        "n_subjects": int(merged.shape[0]),
        "x_shape": [int(merged.shape[0]), int(len(feature_cols))],
        "y_shape": [int(merged.shape[0]), int(len(behavior_cols))],
        "n_neural_features": len(feature_cols),
        "n_behavior_features": len(behavior_cols),
        "neural_features": feature_cols,
        "behavior_features": behavior_cols,
        "observed_score_correlation": observed_score_corr,
        "permutation_p_two_sided": permutation_p,
        "n_permutations": n_permutations,
        "brain_score_vs_scatter_metric": float(
            _correlation_statistic(score_df["x_score"], score_df[scatter_col], scatter_corr_method)
        ),
        "scatter_metric": scatter_col,
        "mean_loo_weights_corr_to_full": float(loo_df["weights_corr_to_full"].mean()),
        "min_loo_weights_corr_to_full": float(loo_df["weights_corr_to_full"].min()),
        "top_neural_weights": to_serializable(
            neural_weight_df.head(10)[["name", "weight"]].to_dict(orient="records")
        ),
        "top_behavior_weights": to_serializable(
            behavior_weight_df.head(10)[["name", "weight"]].to_dict(orient="records")
        ),
    }
    if behavior_label_map:
        summary["behavior_feature_labels"] = {
            name: behavior_label_map.get(name, name) for name in behavior_cols
        }
    write_json(outputs["summary_path"], summary)

    return {
        "subjects": merged["subject"].tolist(),
        "subject_behavior_deltas_path": outputs["behavior_path"],
        "subject_neural_deltas_path": outputs["neural_wide_path"],
        "summary_path": outputs["summary_path"],
        "loadings_path": outputs["loading_path"],
        "scores_path": outputs["score_path"],
        "permutation_path": outputs["permutation_path"],
        "loo_path": outputs["loo_path"],
        "figure_path": outputs["figure_path"],
        "behavior_weight_path": outputs["behavior_weight_path"],
    }


def _all_outputs_exist(out_dir: Path) -> bool:
    legacy_expected = [
        out_dir / "subject_neural_deltas.csv",
        out_dir / "subject_behavior_deltas.csv",
        out_dir / "behavioral_pls_summary.json",
        out_dir / "behavioral_pls_loadings.csv",
        out_dir / "behavioral_pls_scores.csv",
        out_dir / "behavioral_pls_permutation.csv",
        out_dir / "behavioral_pls_loo_stability.csv",
        out_dir / "brain_behavior_coupling.png",
    ]
    return all(path.exists() for path in legacy_expected)


def run_behavior_network_coupling(
    out_dir: Path,
    dcm_subject_parameters_path: Path,
    graph_node_delta_path: Path,
    graph_module_summary_path: Path,
    n_permutations: int = 2000,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))

    behavior_df = load_behavior_deltas().sort_values("subject").reset_index(drop=True)
    task_behavior_df = load_task_behavior_deltas().sort_values("subject").reset_index(drop=True)
    dcm_subject_df = pd.read_csv(dcm_subject_parameters_path)
    node_delta_df = pd.read_csv(graph_node_delta_path)
    module_df = pd.read_csv(graph_module_summary_path)

    legacy_behavior_cols = [
        "behavior_vigor_delta",
        "behavior_lag1_corr_delta",
        "behavior_consistency_improvement_delta",
    ]
    legacy_behavior_labels = {
        "behavior_vigor_delta": "Behavior vigor delta",
        "behavior_lag1_corr_delta": "Behavior lag-1 correlation delta",
        "behavior_consistency_improvement_delta": "Behavior consistency improvement delta",
    }

    legacy_dcm_features = _build_dcm_features(dcm_subject_df, base_rois=CIRCUIT_BASE_ROIS)
    legacy_graph_features = _build_graph_features(
        node_delta_df, base_rois=CIRCUIT_BASE_ROIS
    )
    legacy_module_features = _build_module_features(
        module_df, selected_pairs=LEGACY_MODULE_PAIRS
    )
    legacy_neural_df = (
        legacy_dcm_features.merge(legacy_graph_features, on="subject", how="inner")
        .merge(legacy_module_features, on="subject", how="inner")
        .sort_values("subject")
        .reset_index(drop=True)
    )
    legacy_feature_cols = [col for col in legacy_neural_df.columns if col != "subject"]
    legacy_merged = behavior_df.merge(legacy_neural_df, on="subject", how="inner")

    if _all_outputs_exist(out_dir):
        legacy_results = {
            "subjects": legacy_merged.dropna(
                subset=legacy_feature_cols + legacy_behavior_cols
            )["subject"].sort_values().tolist(),
            "subject_behavior_deltas_path": out_dir / "subject_behavior_deltas.csv",
            "subject_neural_deltas_path": out_dir / "subject_neural_deltas.csv",
            "summary_path": out_dir / "behavioral_pls_summary.json",
            "loadings_path": out_dir / "behavioral_pls_loadings.csv",
            "scores_path": out_dir / "behavioral_pls_scores.csv",
            "permutation_path": out_dir / "behavioral_pls_permutation.csv",
            "loo_path": out_dir / "behavioral_pls_loo_stability.csv",
            "figure_path": out_dir / "brain_behavior_coupling.png",
            "behavior_weight_path": out_dir / "behavioral_pls_behavior_weights.csv",
        }
    else:
        legacy_results = _run_pls_analysis(
            out_dir,
            legacy_merged,
            feature_cols=legacy_feature_cols,
            behavior_cols=legacy_behavior_cols,
            n_permutations=n_permutations,
            scatter_col="behavior_vigor_delta",
            scatter_label="Behavior vigor delta",
            scatter_title="Brain Score vs Vigor Delta",
            scatter_corr_method="spearman",
            behavior_label_map=legacy_behavior_labels,
        )

    expanded_dcm_features = _build_dcm_features_from_raw()
    expanded_graph_features = _build_graph_features(node_delta_df)
    expanded_module_features = _build_module_features(module_df)
    expanded_neural_df = (
        expanded_dcm_features.merge(expanded_graph_features, on="subject", how="inner")
        .merge(expanded_module_features, on="subject", how="inner")
        .sort_values("subject")
        .reset_index(drop=True)
    )
    expanded_feature_cols = [col for col in expanded_neural_df.columns if col != "subject"]

    task_behavior_cols = [f"{spec['key']}_delta" for spec in TASK_BEHAVIOR_COLUMN_SPECS]
    task_behavior_labels = {
        f"{spec['key']}_delta": f"{spec['label']} session mean delta (ON - OFF)"
        for spec in TASK_BEHAVIOR_COLUMN_SPECS
    }

    task_only_dir = ensure_dir(out_dir / "expanded_all_rois_task_only")
    task_only_merged = task_behavior_df.merge(expanded_neural_df, on="subject", how="inner")
    task_only_results = _run_pls_analysis(
        task_only_dir,
        task_only_merged,
        feature_cols=expanded_feature_cols,
        behavior_cols=task_behavior_cols,
        n_permutations=n_permutations,
        scatter_col="y_score",
        scatter_label="PLS behavior score",
        scatter_title="Brain Score vs Task Behavior Score",
        scatter_corr_method="pearson",
        behavior_label_map=task_behavior_labels,
    )

    combined_behavior_df = (
        behavior_df.merge(task_behavior_df, on="subject", how="inner")
        .sort_values("subject")
        .reset_index(drop=True)
    )
    combined_behavior_cols = legacy_behavior_cols + task_behavior_cols
    combined_behavior_labels = {**legacy_behavior_labels, **task_behavior_labels}
    combined_dir = ensure_dir(out_dir / "expanded_all_rois_combined_behavior")
    combined_merged = combined_behavior_df.merge(expanded_neural_df, on="subject", how="inner")
    combined_results = _run_pls_analysis(
        combined_dir,
        combined_merged,
        feature_cols=expanded_feature_cols,
        behavior_cols=combined_behavior_cols,
        n_permutations=n_permutations,
        scatter_col="y_score",
        scatter_label="PLS behavior score",
        scatter_title="Brain Score vs Combined Behavior Score",
        scatter_corr_method="pearson",
        behavior_label_map=combined_behavior_labels,
    )

    return {
        **legacy_results,
        "legacy_feature_count": len(legacy_feature_cols),
        "expanded_feature_count": len(expanded_feature_cols),
        "expanded_task_only": task_only_results,
        "expanded_combined_behavior": combined_results,
    }
