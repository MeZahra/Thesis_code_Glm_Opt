"""ROI graph analysis: run-averaged partial correlation + run-level mixed LM.

Hemispheric variant of roi_graph_runaveraged_analysis:
- keeps left/right ROI labels separate
- uses the same run-level mixed model
- uses the same anatomical-system partition, but derived from hemisphere-stripped base ROI names
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from common_io import (
    ANATOMICAL_SYSTEM_ORDER,
    DEFAULT_DROP_LABEL_PATTERNS,
    base_roi_name,
    bh_fdr,
    drop_labels_and_matrix,
    ensure_dir,
    infer_anatomical_system,
    list_paired_subjects_for_metric,
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
    weighted_modularity,
)
from roi_graph_base_reorganization import _normalize_run_labels
from roi_graph_runaveraged_analysis import (
    METRIC,
    PD_CIRCUIT_ROIS,
    RUN_LEVEL_ROOT,
    _compute_node_metrics,
    _fit_run_level_model,
)


ANATOMICAL_SYSTEM_TO_INT = {
    name: idx for idx, name in enumerate(ANATOMICAL_SYSTEM_ORDER)
}


def _build_anatomical_partition(roi_labels: list[str]) -> np.ndarray:
    return np.array(
        [
            ANATOMICAL_SYSTEM_TO_INT[infer_anatomical_system(base_roi_name(label))]
            for label in roi_labels
        ],
        dtype=int,
    )


def _reorder_to_reference(
    matrix: np.ndarray,
    labels: list[str],
    reference_labels: list[str],
) -> tuple[np.ndarray, list[str]]:
    if labels == reference_labels:
        return matrix, labels
    if set(labels) != set(reference_labels):
        raise ValueError("Run labels do not match the reference hemispheric ROI set.")
    index_by_label = {label: idx for idx, label in enumerate(labels)}
    order = [index_by_label[label] for label in reference_labels]
    reordered = np.asarray(matrix)[np.ix_(order, order)]
    return reordered, list(reference_labels)


def _load_run_matrix(
    subject: str,
    session: int,
    run: int,
    metrics_root: Path,
    drop_patterns,
    reference_labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]] | None:
    label = f"{subject}_ses-{session}_run-{run}"
    metric_dir = metrics_root / label / METRIC
    matrix_path = metric_dir / f"{METRIC}.npy"
    labels_path = metric_dir / f"{METRIC}_connectome.labels.txt"
    if not matrix_path.exists() or not labels_path.exists():
        return None

    raw_matrix = np.load(matrix_path)
    raw_labels = _normalize_run_labels(
        labels_path.read_text(encoding="utf-8").splitlines()
    )
    trimmed_matrix, trimmed_labels, _ = drop_labels_and_matrix(
        raw_labels, raw_matrix, drop_patterns
    )
    adjacency = sanitize_matrix(trimmed_matrix, zero_diagonal=True)
    labels = list(trimmed_labels)
    if reference_labels is not None:
        adjacency, labels = _reorder_to_reference(adjacency, labels, reference_labels)
    return adjacency, labels


def run_roi_graph_runaveraged_hemispheric(
    out_dir: Path,
    metrics_root: Path = RUN_LEVEL_ROOT,
    drop_patterns=DEFAULT_DROP_LABEL_PATTERNS,
    roi_subset_base: list[str] | None = PD_CIRCUIT_ROIS,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))

    subjects = list_paired_subjects_for_metric(METRIC)

    # run_data[(subject, session, run)] = (matrix, roi_labels)
    run_data: dict[tuple[str, int, int], tuple[np.ndarray, list[str]]] = {}
    roi_labels_ref: list[str] | None = None

    for subject in subjects:
        complete = True
        subject_runs: dict[tuple[str, int, int], tuple[np.ndarray, list[str]]] = {}
        for session in (1, 2):
            for run in (1, 2):
                result = _load_run_matrix(
                    subject,
                    session,
                    run,
                    metrics_root,
                    drop_patterns,
                    reference_labels=roi_labels_ref,
                )
                if result is None:
                    complete = False
                    break
                matrix, roi_labels = result
                if roi_labels_ref is None:
                    roi_labels_ref = roi_labels
                elif roi_labels != roi_labels_ref:
                    complete = False
                    break
                subject_runs[(subject, session, run)] = (matrix, roi_labels)
            if not complete:
                break
        if complete:
            run_data.update(subject_runs)

    valid_subjects = sorted({subject for subject, _, _ in run_data})
    if not valid_subjects or roi_labels_ref is None:
        raise RuntimeError("No subjects with complete hemispheric run-level data found.")

    anatomical_partition = _build_anatomical_partition(roi_labels_ref)

    session_node_rows: list[dict] = []
    run_node_rows: list[dict] = []
    module_rows: list[dict] = []
    grand_positive: list[np.ndarray] = []

    for subject in valid_subjects:
        for session in (1, 2):
            matrices = [run_data[(subject, session, run)][0] for run in (1, 2)]
            avg_matrix = np.mean(np.stack(matrices, axis=0), axis=0)
            avg_positive = np.clip(avg_matrix, 0.0, None)
            grand_positive.append(avg_positive)

            session_metrics = _compute_node_metrics(
                avg_matrix, roi_labels_ref, anatomical_partition
            )
            for idx, roi in enumerate(roi_labels_ref):
                base_roi = base_roi_name(roi)
                session_node_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "state": "OFF" if int(session) == 1 else "ON",
                        "roi": roi,
                        "base_roi": base_roi,
                        "anatomical_system": infer_anatomical_system(base_roi),
                        "node_strength_abs": float(session_metrics["node_strength_abs"][idx]),
                        "node_strength_positive": float(session_metrics["node_strength_positive"][idx]),
                        "participation_coeff": float(session_metrics["participation_coeff"][idx]),
                        "within_module_z": float(session_metrics["within_module_z"][idx]),
                        "anatomical_community": int(anatomical_partition[idx] + 1),
                        "hub_class": classify_hub(
                            float(session_metrics["participation_coeff"][idx]),
                            float(session_metrics["within_module_z"][idx]),
                        ),
                    }
                )

            for run in (1, 2):
                run_matrix = run_data[(subject, session, run)][0]
                run_metrics = _compute_node_metrics(
                    run_matrix, roi_labels_ref, anatomical_partition
                )
                for idx, roi in enumerate(roi_labels_ref):
                    base_roi = base_roi_name(roi)
                    run_node_rows.append(
                        {
                            "subject": subject,
                            "session": int(session),
                            "run": int(run),
                            "state": "OFF" if int(session) == 1 else "ON",
                            "roi": roi,
                            "base_roi": base_roi,
                            "anatomical_system": infer_anatomical_system(base_roi),
                            "node_strength_abs": float(run_metrics["node_strength_abs"][idx]),
                            "node_strength_positive": float(run_metrics["node_strength_positive"][idx]),
                            "participation_coeff": float(run_metrics["participation_coeff"][idx]),
                            "within_module_z": float(run_metrics["within_module_z"][idx]),
                        }
                    )

            absolute = np.abs(avg_matrix)
            anatomical_group_map = {
                roi: infer_anatomical_system(base_roi_name(roi))
                for roi in roi_labels_ref
            }
            module_rows.extend(
                _compute_group_pair_rows(
                    absolute,
                    roi_labels_ref,
                    anatomical_group_map,
                    subject,
                    session,
                    "anatomical_system",
                )
            )

    mean_positive = np.mean(np.stack(grand_positive, axis=0), axis=0)
    anatomical_q = float(weighted_modularity(mean_positive, anatomical_partition))

    node_df = pd.DataFrame(session_node_rows)
    run_node_df = pd.DataFrame(run_node_rows)
    module_df = pd.DataFrame(module_rows)

    delta_rows: list[dict] = []
    node_tests: list[dict] = []

    for roi in roi_labels_ref:
        roi_session_df = node_df[node_df["roi"] == roi]
        off = roi_session_df[roi_session_df["session"] == 1].sort_values("subject")
        on = roi_session_df[roi_session_df["session"] == 2].sort_values("subject")
        merged = off.merge(on, on="subject", suffixes=("_off", "_on"))
        if merged.empty:
            continue

        roi_run_rows = run_node_df[run_node_df["roi"] == roi].to_dict("records")

        for metric_name in [
            "node_strength_abs",
            "node_strength_positive",
            "participation_coeff",
            "within_module_z",
        ]:
            delta = merged[f"{metric_name}_on"] - merged[f"{metric_name}_off"]
            signflip_stats = paired_delta_stats(delta)

            lm_result = _fit_run_level_model(roi_run_rows, metric_name)
            if lm_result:
                p_primary = float(lm_result["p_primary"])
                primary_test = str(lm_result["primary_test"])
                primary_estimate = float(lm_result.get("primary_estimate", np.nan))
            else:
                p_primary = float(signflip_stats["p_signflip"])
                primary_test = "signflip_fallback"
                primary_estimate = float(signflip_stats["mean_delta"])

            base_roi = base_roi_name(roi)
            node_tests.append(
                {
                    "roi": roi,
                    "base_roi": base_roi,
                    "anatomical_system": infer_anatomical_system(base_roi),
                    "metric": metric_name,
                    "mean_off": float(merged[f"{metric_name}_off"].mean()),
                    "mean_on": float(merged[f"{metric_name}_on"].mean()),
                    "p_primary": p_primary,
                    "primary_test": primary_test,
                    "primary_estimate": primary_estimate,
                    **signflip_stats,
                }
            )
            for subject, value in zip(merged["subject"], delta):
                delta_rows.append(
                    {
                        "subject": subject,
                        "roi": roi,
                        "base_roi": base_roi,
                        "anatomical_system": infer_anatomical_system(base_roi),
                        f"delta_{metric_name}": float(value),
                    }
                )

    delta_df = pd.DataFrame(delta_rows)
    if not delta_df.empty:
        delta_df = (
            delta_df.groupby(
                ["subject", "roi", "base_roi", "anatomical_system"], as_index=False
            )
            .first()
            .sort_values(["subject", "roi"])
        )

    node_test_df = pd.DataFrame(node_tests)
    node_test_df["in_fdr_subset"] = True
    if roi_subset_base is not None:
        node_test_df["in_fdr_subset"] = node_test_df["base_roi"].isin(roi_subset_base)
    node_test_df["q_fdr_within_metric"] = np.nan
    for metric_name, metric_df in node_test_df.groupby("metric"):
        subset_mask = metric_df["in_fdr_subset"]
        if subset_mask.any():
            q_subset = bh_fdr(metric_df.loc[subset_mask, "p_primary"])
            node_test_df.loc[metric_df.index[subset_mask], "q_fdr_within_metric"] = q_subset

    node_test_df["p_subject"] = node_test_df["p_wilcoxon"]
    missing_subject_p = ~np.isfinite(node_test_df["p_subject"])
    node_test_df.loc[missing_subject_p, "p_subject"] = node_test_df.loc[
        missing_subject_p, "p_signflip"
    ]
    node_test_df["subject_test"] = "wilcoxon"
    node_test_df.loc[missing_subject_p, "subject_test"] = "signflip_fallback"
    node_test_df["q_subject_fdr_within_metric"] = np.nan
    for metric_name, metric_df in node_test_df.groupby("metric"):
        q_subject = bh_fdr(metric_df["p_subject"])
        node_test_df.loc[metric_df.index, "q_subject_fdr_within_metric"] = q_subject

    node_test_df = node_test_df.sort_values(
        [
            "metric",
            "q_subject_fdr_within_metric",
            "q_fdr_within_metric",
            "p_primary",
            "mean_delta",
        ],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)

    hub_summary = (
        node_df.groupby(["roi", "base_roi", "anatomical_system", "state"], as_index=False)
        .agg(
            mean_node_strength_abs=("node_strength_abs", "mean"),
            mean_participation_coeff=("participation_coeff", "mean"),
            mean_within_module_z=("within_module_z", "mean"),
            connector_hub_fraction=(
                "hub_class", lambda s: float((s == "connector_hub").mean())
            ),
            provincial_hub_fraction=(
                "hub_class", lambda s: float((s == "provincial_hub").mean())
            ),
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
    hub_summary.columns = [f"{m.lower()}_{s.lower()}" for m, s in hub_summary.columns]
    hub_summary = hub_summary.reset_index()
    if {"mean_node_strength_abs_off", "mean_node_strength_abs_on"} <= set(hub_summary.columns):
        hub_summary["mean_node_strength_abs_delta"] = (
            hub_summary["mean_node_strength_abs_on"]
            - hub_summary["mean_node_strength_abs_off"]
        )
        hub_summary["mean_participation_coeff_delta"] = (
            hub_summary["mean_participation_coeff_on"]
            - hub_summary["mean_participation_coeff_off"]
        )
        hub_summary["mean_within_module_z_delta"] = (
            hub_summary["mean_within_module_z_on"]
            - hub_summary["mean_within_module_z_off"]
        )

    node_metrics_path = out_dir / "node_metrics_by_subject_session.csv"
    run_node_metrics_path = out_dir / "node_metrics_by_subject_session_run.csv"
    delta_path = out_dir / "node_metric_deltas_on_minus_off.csv"
    test_path = out_dir / "node_metric_tests_fdr.csv"
    module_path = out_dir / "module_integration_summary.csv"
    hub_path = out_dir / "hub_summary.csv"
    summary_path = out_dir / "community_consensus_summary.json"
    hub_fig_path = out_dir / "hub_reorganization.png"
    module_fig_path = out_dir / "module_delta_heatmap.png"

    node_df.to_csv(node_metrics_path, index=False)
    run_node_df.to_csv(run_node_metrics_path, index=False)
    delta_df.to_csv(delta_path, index=False)
    node_test_df.to_csv(test_path, index=False)
    module_df.to_csv(module_path, index=False)
    hub_summary.to_csv(hub_path, index=False)
    _plot_hub_reorganization(node_test_df, hub_fig_path)
    _plot_module_delta_heatmap(module_df, module_fig_path)

    significant = node_test_df[node_test_df["q_fdr_within_metric"] < 0.05]
    if roi_subset_base is None:
        n_fdr_tests = len(roi_labels_ref)
    else:
        n_fdr_tests = int(
            sum(base_roi_name(label) in set(roi_subset_base) for label in roi_labels_ref)
        )

    summary = {
        "metric": METRIC,
        "analysis_level": "hemispheric_run_averaged_mixed_lm",
        "aggregate_hemispheres": False,
        "community_scheme": "anatomical_system_predefined",
        "community_order": ANATOMICAL_SYSTEM_ORDER,
        "n_subjects": len(valid_subjects),
        "n_hemispheric_rois": len(roi_labels_ref),
        "fdr_base_roi_subset": roi_subset_base,
        "n_tests_per_metric_family": n_fdr_tests,
        "dropped_label_patterns": list(drop_patterns),
        "anatomical_modularity_q": anatomical_q,
        "n_fdr_significant": int(len(significant)),
        "fdr_significant_effects": to_serializable(
            significant[
                ["roi", "metric", "mean_delta", "p_primary", "q_fdr_within_metric", "cohen_dz"]
            ].to_dict(orient="records")
        ),
        "top_node_effects": to_serializable(
            node_test_df.assign(abs_delta=node_test_df["mean_delta"].abs())
            .sort_values(
                ["q_fdr_within_metric", "abs_delta"],
                ascending=[True, False],
            )
            .head(12)[
                [
                    "roi",
                    "metric",
                    "mean_delta",
                    "cohen_dz",
                    "p_signflip",
                    "p_primary",
                    "q_fdr_within_metric",
                    "primary_test",
                ]
            ]
            .to_dict(orient="records")
        ),
    }
    write_json(summary_path, summary)

    return {
        "subjects": valid_subjects,
        "metric": METRIC,
        "analysis_level": "hemispheric_run_averaged_mixed_lm",
        "node_metrics_path": node_metrics_path,
        "run_node_metrics_path": run_node_metrics_path,
        "node_delta_path": delta_path,
        "node_tests_path": test_path,
        "module_summary_path": module_path,
        "hub_summary_path": hub_path,
        "summary_path": summary_path,
        "hub_figure_path": hub_fig_path,
        "module_figure_path": module_fig_path,
    }


if __name__ == "__main__":
    out_dir = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "connectivity"
        / "GPT"
        / "tmp"
        / "roi_graph_analysis_runaveraged_anatomical_hemispheric"
    )
    results = run_roi_graph_runaveraged_hemispheric(out_dir)
    summary = json.loads(results["summary_path"].read_text(encoding="utf-8"))
    print(f"n_subjects: {summary['n_subjects']}")
    print(f"n_hemispheric_rois: {summary['n_hemispheric_rois']}")
    print(f"anatomical_modularity_q: {summary['anatomical_modularity_q']:.4f}")
    print(f"n_fdr_significant: {summary['n_fdr_significant']}")
    if summary["fdr_significant_effects"]:
        print("\nFDR-significant effects (q < 0.05):")
        for eff in summary["fdr_significant_effects"]:
            print(
                f"  {eff['roi']:45s} {eff['metric']:30s}  "
                f"d={eff['cohen_dz']:+.3f}  q={eff['q_fdr_within_metric']:.4f}"
            )
    else:
        print("\nNo FDR-significant effects.")
