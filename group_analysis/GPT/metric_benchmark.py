from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common_io import (
    BENCHMARK_METRICS,
    CIRCUIT_BASE_ROIS,
    DEFAULT_DROP_LABEL_PATTERNS,
    ROI_EDGE_RESULTS_ROOT,
    aggregate_matrix_by_base_roi,
    base_roi_name,
    bh_fdr,
    drop_labels_and_matrix,
    ensure_dir,
    infer_anatomical_system,
    list_paired_subjects_for_metric,
    load_metric_matrix,
    paired_delta_stats,
    read_json,
    sanitize_matrix,
    to_serializable,
    upper_triangle_mean,
    write_json,
)


def _reduced_metric_summaries(matrix: np.ndarray, labels: list[str]) -> dict[str, float]:
    absolute = np.abs(sanitize_matrix(matrix))
    base_matrix, base_labels, _ = aggregate_matrix_by_base_roi(absolute, labels)
    system_map = {label: infer_anatomical_system(label) for label in base_labels}

    def block_mean(group_a: str, group_b: str) -> float:
        idx_a = [idx for idx, label in enumerate(base_labels) if system_map[label] == group_a]
        idx_b = [idx for idx, label in enumerate(base_labels) if system_map[label] == group_b]
        if not idx_a or not idx_b:
            return float("nan")
        if group_a == group_b:
            block = base_matrix[np.ix_(idx_a, idx_a)]
            return upper_triangle_mean(block)
        block = base_matrix[np.ix_(idx_a, idx_b)]
        return float(np.nanmean(block))

    circuit_idx = [idx for idx, label in enumerate(base_labels) if label in CIRCUIT_BASE_ROIS]
    circuit_block = base_matrix[np.ix_(circuit_idx, circuit_idx)]
    return {
        "global_abs_strength": upper_triangle_mean(base_matrix),
        "circuit_abs_strength": upper_triangle_mean(circuit_block),
        "within_cognitive_control": block_mean("cognitive_control", "cognitive_control"),
        "within_motor_sensorimotor": block_mean("motor_sensorimotor", "motor_sensorimotor"),
        "between_cognitive_control_motor_sensorimotor": block_mean(
            "cognitive_control", "motor_sensorimotor"
        ),
        "between_cognitive_control_subcortical_relay": block_mean(
            "cognitive_control", "subcortical_relay"
        ),
        "between_motor_sensorimotor_subcortical_relay": block_mean(
            "motor_sensorimotor", "subcortical_relay"
        ),
    }


def _plot_benchmark(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].bar(summary_df["metric"], summary_df["best_effect_abs_cohen_dz"], color="#4c78a8")
    axes[0].set_title("Best ON-OFF Effect Size")
    axes[0].set_ylabel("|Cohen dz|")
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary_df["metric"], summary_df["best_behavior_abs_rho"], color="#54a24b")
    axes[1].set_title("Best Brain-Behavior Link")
    axes[1].set_ylabel("|Spearman rho|")
    axes[1].tick_params(axis="x", rotation=25)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_metric_benchmark(
    out_dir: Path,
    subject_behavior_deltas_path: Path,
    metrics: tuple[str, ...] = BENCHMARK_METRICS,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    behavior_df = pd.read_csv(subject_behavior_deltas_path)
    null_summary_path = (
        ROI_EDGE_RESULTS_ROOT
        / "advanced_metrics"
        / "random_graph_distance_null"
        / "observed_vs_random_null_summary.csv"
    )
    null_summary_df = pd.read_csv(null_summary_path)

    subject_rows = []
    effect_rows = []
    behavior_rows = []
    summary_rows = []

    for metric in metrics:
        subjects = list_paired_subjects_for_metric(metric)
        for subject in subjects:
            for session in (1, 2):
                matrix, labels = load_metric_matrix(subject, session, metric)
                trimmed_matrix, trimmed_labels, _ = drop_labels_and_matrix(
                    labels, matrix, DEFAULT_DROP_LABEL_PATTERNS
                )
                summaries = _reduced_metric_summaries(trimmed_matrix, trimmed_labels)
                subject_rows.append(
                    {
                        "metric": metric,
                        "subject": subject,
                        "session": int(session),
                        "state": "OFF" if int(session) == 1 else "ON",
                        **summaries,
                    }
                )

    subject_df = pd.DataFrame(subject_rows)

    for metric, metric_df in subject_df.groupby("metric", sort=True):
        effect_df_rows = []
        for summary_name in [col for col in metric_df.columns if col not in {"metric", "subject", "session", "state"}]:
            off = metric_df[metric_df["session"] == 1][["subject", summary_name]].rename(
                columns={summary_name: "off"}
            )
            on = metric_df[metric_df["session"] == 2][["subject", summary_name]].rename(
                columns={summary_name: "on"}
            )
            merged = off.merge(on, on="subject", how="inner")
            delta = merged["on"] - merged["off"]
            stats_dict = paired_delta_stats(delta)
            effect_df_rows.append(
                {
                    "metric": metric,
                    "summary_name": summary_name,
                    "mean_off": float(merged["off"].mean()),
                    "mean_on": float(merged["on"].mean()),
                    **stats_dict,
                }
            )
            behavior_merge = merged.merge(behavior_df[["subject", "behavior_vigor_delta"]], on="subject", how="inner")
            rho, p_val = stats.spearmanr(
                behavior_merge["on"] - behavior_merge["off"],
                behavior_merge["behavior_vigor_delta"],
            )
            behavior_rows.append(
                {
                    "metric": metric,
                    "summary_name": summary_name,
                    "behavior_metric": "behavior_vigor_delta",
                    "spearman_rho": float(rho),
                    "p_value": float(p_val),
                    "n_subjects": int(behavior_merge.shape[0]),
                }
            )

        effect_metric_df = pd.DataFrame(effect_df_rows)
        effect_metric_df["q_fdr_within_metric"] = bh_fdr(effect_metric_df["p_signflip"])
        effect_rows.extend(effect_metric_df.to_dict(orient="records"))

        behavior_metric_df = pd.DataFrame(
            [row for row in behavior_rows if row["metric"] == metric]
        )
        best_effect = effect_metric_df.assign(
            abs_cohen_dz=effect_metric_df["cohen_dz"].abs()
        ).sort_values(["q_fdr_within_metric", "abs_cohen_dz"], ascending=[True, False]).iloc[0]
        best_behavior = behavior_metric_df.assign(
            abs_rho=behavior_metric_df["spearman_rho"].abs()
        ).sort_values("abs_rho", ascending=False).iloc[0]

        null_row = null_summary_df[null_summary_df["metric"] == metric]
        summary_rows.append(
            {
                "metric": metric,
                "best_effect_summary": best_effect["summary_name"],
                "best_effect_mean_delta": float(best_effect["mean_delta"]),
                "best_effect_abs_cohen_dz": float(abs(best_effect["cohen_dz"])),
                "best_effect_q_fdr": float(best_effect["q_fdr_within_metric"]),
                "best_behavior_summary": best_behavior["summary_name"],
                "best_behavior_abs_rho": float(abs(best_behavior["spearman_rho"])),
                "best_behavior_rho": float(best_behavior["spearman_rho"]),
                "best_behavior_p_value": float(best_behavior["p_value"]),
                "observed_null_delta_sep": float(null_row["delta_sep"].iloc[0]) if not null_row.empty else float("nan"),
                "observed_null_delta_sep_p_empirical_right": float(
                    null_row["delta_sep_p_empirical_right"].iloc[0]
                )
                if not null_row.empty
                else float("nan"),
            }
        )

    effect_df = pd.DataFrame(effect_rows)
    behavior_link_df = pd.DataFrame(behavior_rows)
    summary_df = pd.DataFrame(summary_rows)

    summary_path = out_dir / "metric_benchmark_summary.csv"
    effect_path = out_dir / "metric_benchmark_effectsizes.csv"
    behavior_path = out_dir / "metric_benchmark_behavior_links.csv"
    figure_path = out_dir / "metric_benchmark_overview.png"

    summary_df.to_csv(summary_path, index=False)
    effect_df.to_csv(effect_path, index=False)
    behavior_link_df.to_csv(behavior_path, index=False)
    _plot_benchmark(summary_df, figure_path)

    write_json(
        out_dir / "metric_benchmark_summary.json",
        {
            "metrics": list(metrics),
            "summary_rows": to_serializable(summary_df.to_dict(orient="records")),
        },
    )

    return {
        "summary_path": summary_path,
        "effectsizes_path": effect_path,
        "behavior_links_path": behavior_path,
        "figure_path": figure_path,
    }
