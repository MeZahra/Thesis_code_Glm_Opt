from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from common_io import bh_fdr, paired_delta_stats


METRICS = ["node_strength_abs", "participation_coeff", "within_module_z"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def summarize_metric_family(node_tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        metric_df = node_tests[node_tests["metric"] == metric].copy()
        if metric_df.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "n_tests": int(len(metric_df)),
                "raw_p_lt_0.05": int((metric_df["p_signflip"] < 0.05).sum()),
                "raw_p_lt_0.10": int((metric_df["p_signflip"] < 0.10).sum()),
                "q_lt_0.05": int((metric_df["q_fdr_within_metric"] < 0.05).sum()),
                "min_p": float(metric_df["p_signflip"].min()),
                "min_q": float(metric_df["q_fdr_within_metric"].min()),
                "median_abs_dz": float(metric_df["cohen_dz"].abs().median()),
            }
        )
    return pd.DataFrame(rows)


def approximate_n_for_bh_first_hit(cohen_dz: float, n_tests: int, alpha: float = 0.05) -> float:
    if not np.isfinite(cohen_dz) or cohen_dz == 0.0:
        return float("nan")
    p_threshold = alpha / float(n_tests)
    effect = abs(float(cohen_dz))
    for n_subjects in range(14, 201):
        t_stat = effect * math.sqrt(n_subjects)
        p_value = 2.0 * stats.t.sf(abs(t_stat), df=n_subjects - 1)
        if p_value < p_threshold:
            return float(n_subjects)
    return float("nan")


def top_hits_with_sample_size(node_tests: pd.DataFrame, metric: str, top_n: int = 5) -> pd.DataFrame:
    metric_df = node_tests[node_tests["metric"] == metric].copy()
    metric_df = metric_df.sort_values(["p_signflip", "q_fdr_within_metric", "mean_delta"])
    metric_df = metric_df.head(top_n).copy()
    n_tests = max(int((node_tests["metric"] == metric).sum()), 1)
    metric_df["approx_n_for_first_bh_hit"] = metric_df["cohen_dz"].apply(
        lambda value: approximate_n_for_bh_first_hit(value, n_tests)
    )
    return metric_df[
        ["roi", "mean_delta", "cohen_dz", "p_signflip", "q_fdr_within_metric", "approx_n_for_first_bh_hit"]
    ]


def collapsed_family_stats(node_metrics: pd.DataFrame, group_col: str) -> pd.DataFrame:
    agg = (
        node_metrics.groupby(["subject", "session", "state", group_col], as_index=False)[
            ["node_strength_abs", "node_strength_positive", "participation_coeff", "within_module_z"]
        ]
        .mean()
    )

    rows = []
    for group_name, group_df in agg.groupby(group_col):
        off = group_df[group_df["session"] == 1].sort_values("subject")
        on = group_df[group_df["session"] == 2].sort_values("subject")
        merged = off.merge(on, on="subject", suffixes=("_off", "_on"))
        if merged.empty:
            continue
        for metric in METRICS:
            delta = merged[f"{metric}_on"] - merged[f"{metric}_off"]
            stats_dict = paired_delta_stats(delta)
            rows.append({group_col: group_name, "metric": metric, **stats_dict})

    collapsed = pd.DataFrame(rows)
    collapsed["q_fdr_within_metric"] = np.nan
    for metric_name, metric_df in collapsed.groupby("metric"):
        q_values = bh_fdr(metric_df["p_signflip"])
        collapsed.loc[metric_df.index, "q_fdr_within_metric"] = q_values
    return collapsed


def trial_balance_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    return (
        split_df.groupby("session", as_index=False)["n_trials"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )


def partial_corr_summary_if_available(results_root: Path) -> pd.DataFrame | None:
    path = results_root / "roi_graph_analysis_partial_correlation_check" / "node_metric_tests_fdr.csv"
    if not path.exists():
        return None
    node_tests = pd.read_csv(path)
    summary = summarize_metric_family(node_tests)
    summary.insert(0, "analysis", "partial_correlation_node_level")
    return summary


def render_block(title: str, df: pd.DataFrame) -> list[str]:
    lines = [f"## {title}", ""]
    if df.empty:
        lines.append("No rows available.")
        lines.append("")
        return lines
    lines.append("```text")
    lines.append(df.to_string(index=False))
    lines.append("```")
    lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the current ROI graph FDR bottlenecks.")
    parser.add_argument(
        "--results-root",
        type=Path,
        default=repo_root() / "results" / "connectivity" / "GPT",
        help="Path to the GPT connectivity results root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "results" / "connectivity" / "GPT" / "roi_graph_analysis" / "fdr_diagnostics_report.md",
        help="Markdown report path.",
    )
    args = parser.parse_args()

    results_root = args.results_root
    roi_graph_root = results_root / "roi_graph_analysis"

    node_tests = pd.read_csv(roi_graph_root / "node_metric_tests_fdr.csv")
    node_metrics = pd.read_csv(roi_graph_root / "node_metrics_by_subject_session.csv")
    split_df = pd.read_csv(repo_root() / "results" / "connectivity" / "data" / "subject_session_split_summary.csv")
    benchmark_df = pd.read_csv(results_root / "metric_benchmark" / "metric_benchmark_summary.csv")

    current_summary = summarize_metric_family(node_tests)
    current_summary.insert(0, "analysis", "ksg_node_level")

    base_roi_summary = summarize_metric_family(collapsed_family_stats(node_metrics, "base_roi"))
    base_roi_summary.insert(0, "analysis", "collapsed_base_roi")

    anatomical_summary = summarize_metric_family(collapsed_family_stats(node_metrics, "anatomical_system"))
    anatomical_summary.insert(0, "analysis", "collapsed_anatomical_system")

    top_strength = top_hits_with_sample_size(node_tests, "node_strength_abs", top_n=6)
    top_strength["approx_n_for_first_bh_hit"] = top_strength["approx_n_for_first_bh_hit"].astype("Int64")

    trial_summary = trial_balance_summary(split_df)

    metric_benchmark = benchmark_df[
        ["metric", "best_effect_summary", "best_effect_abs_cohen_dz", "best_effect_q_fdr", "best_behavior_abs_rho"]
    ].sort_values(["best_effect_q_fdr", "best_effect_abs_cohen_dz"], ascending=[True, False])

    partial_corr_summary = partial_corr_summary_if_available(results_root)

    report_lines: list[str] = [
        "# ROI Graph FDR Diagnostics",
        "",
        "This report explains why the current ROI graph analysis is not reaching low q-values and which levers are most defensible.",
        "",
    ]
    report_lines.extend(render_block("Current KSG Node-Level Family", current_summary))
    report_lines.extend(render_block("Top KSG Node-Strength Hits With Approximate Sample Size Targets", top_strength))
    report_lines.extend(render_block("Collapsed Base-ROI Family", base_roi_summary))
    report_lines.extend(render_block("Collapsed Anatomical-System Family", anatomical_summary))
    report_lines.extend(render_block("Session Trial Balance", trial_summary))
    report_lines.extend(render_block("Connectivity Metric Benchmark", metric_benchmark))
    if partial_corr_summary is not None:
        report_lines.extend(render_block("Partial-Correlation Node-Level Check", partial_corr_summary))

    report_lines.extend(
        [
            "## Interpretation",
            "",
            "- The BH/FDR implementation is standard; high q-values are not a software bug.",
            "- Session trial counts are balanced, so the current bottleneck is not a gross ON/OFF trial imbalance.",
            "- The current node-level family is large enough that the best raw p-values are still too weak after multiple-comparison correction.",
            "- Collapsing left/right ROIs to base ROIs halves the family size, but still does not produce q < 0.05 with the current sample.",
            "- Collapsing to anatomical systems reduces the family to six tests, but the strongest q-value is still above 0.05, which points to limited power rather than only excessive multiplicity.",
            "- The benchmark summary suggests that partial correlation is a more promising metric than KSG for coarse summaries, but it still does not rescue the current node-level family.",
            "",
            "## Recommended Next Steps",
            "",
            "- Treat node-level hub reorganization as exploratory unless you add sample size or impose a tighter a priori hypothesis set.",
            "- If hemispheric specificity is not central, predefine base-ROI or anatomical-system families before testing.",
            "- If the scientific target is a network-level medication effect, move the primary inference to the coarser family where the metric benchmark is strongest.",
            "- Prefer one predeclared metric family instead of searching across many metrics and many graph summaries after looking at the results.",
            "- If you need node-level claims, the present effect sizes suggest materially more paired subjects will likely be required.",
            "",
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
