from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from behavior_network_coupling import run_behavior_network_coupling
from common_io import (
    BENCHMARK_METRICS,
    GPT_RESULTS_ROOT,
    PRIMARY_METRIC,
    REPO_ROOT,
    TMP_DCM_ROOT,
    ensure_dir,
    load_behavior_session_summary,
    load_subject_session_split_summary,
    paired_delta_stats,
    recursive_file_inventory,
    to_serializable,
    write_json,
)
from dcm_medication_analysis import run_dcm_medication_analysis
from metric_benchmark import run_metric_benchmark
from roi_graph_reorganization import run_roi_graph_reorganization
from supplementary_ksg_summary import run_supplementary_ksg_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-oriented medication analyses.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=GPT_RESULTS_ROOT,
        help="Root folder for GPT paper-analysis outputs.",
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default=PRIMARY_METRIC,
        help="Primary ROI-connectivity metric for graph analyses.",
    )
    parser.add_argument(
        "--pls-permutations",
        type=int,
        default=2000,
        help="Number of permutations for the behavioral PLS score-correlation test.",
    )
    return parser.parse_args()


def build_paper_tables(
    out_dir: Path,
    included_subjects: list[str],
    dcm_results: dict,
    graph_results: dict,
    behavior_results: dict,
    benchmark_results: dict,
    supplementary_results: dict,
) -> dict:
    out_dir = ensure_dir(out_dir)

    split_df = load_subject_session_split_summary()
    behavior_session_df = load_behavior_session_summary()[
        ["subject", "session", "n_trials_behavior", "behavior_vigor", "projection_metric"]
    ]
    subject_trials_df = (
        split_df[split_df["subject"].isin(included_subjects)]
        .merge(behavior_session_df, on=["subject", "session"], how="left")
        .sort_values(["subject", "session"])
        .reset_index(drop=True)
    )
    subject_trials_path = out_dir / "table_subjects_and_trials.csv"
    subject_trials_df.to_csv(subject_trials_path, index=False)

    dcm_df = pd.read_csv(dcm_results["group_comparison_fdr_path"]).head(10)
    dcm_table = dcm_df.assign(
        analysis_family="effective_connectivity",
        effect_name=dcm_df["edge"],
        estimate=dcm_df["mean_delta"],
        p_value=dcm_df["p_signflip"],
        q_value=dcm_df["q_fdr"],
        effect_size=dcm_df["cohen_dz"],
    )[
        ["analysis_family", "effect_name", "estimate", "p_value", "q_value", "effect_size"]
    ]

    graph_df = pd.read_csv(graph_results["node_tests_path"]).head(12)
    graph_table = graph_df.assign(
        analysis_family="roi_graph_reorganization",
        effect_name=graph_df["roi"] + " | " + graph_df["metric"],
        estimate=graph_df["mean_delta"],
        p_value=graph_df["p_signflip"],
        q_value=graph_df["q_fdr_within_metric"],
        effect_size=graph_df["cohen_dz"],
    )[
        ["analysis_family", "effect_name", "estimate", "p_value", "q_value", "effect_size"]
    ]

    behavior_summary = pd.read_json(behavior_results["summary_path"], typ="series")
    behavior_table = pd.DataFrame(
        [
            {
                "analysis_family": "behavior_network_coupling",
                "effect_name": "PLS latent score correlation",
                "estimate": float(behavior_summary["observed_score_correlation"]),
                "p_value": float(behavior_summary["permutation_p_two_sided"]),
                "q_value": np.nan,
                "effect_size": float(behavior_summary["brain_score_vs_behavior_vigor_spearman"]),
            }
        ]
    )
    main_effects_df = pd.concat([dcm_table, graph_table, behavior_table], ignore_index=True)
    main_effects_path = out_dir / "table_main_effects.csv"
    main_effects_df.to_csv(main_effects_path, index=False)

    benchmark_df = pd.read_csv(benchmark_results["summary_path"])
    secondary_effects_path = out_dir / "table_secondary_effects.csv"
    benchmark_df.to_csv(secondary_effects_path, index=False)

    ksg_null_df = pd.read_csv(supplementary_results["null_summary_path"])
    ksg_consistency_df = pd.read_csv(supplementary_results["consistency_path"]).head(10)
    supplementary_df = pd.concat(
        [
            ksg_null_df.assign(section="ksg_random_null"),
            ksg_consistency_df.assign(section="ksg_edge_consistency"),
        ],
        ignore_index=True,
        sort=False,
    )
    supplementary_path = out_dir / "table_supplementary_metrics.csv"
    supplementary_df.to_csv(supplementary_path, index=False)

    return {
        "subjects_and_trials_path": subject_trials_path,
        "main_effects_path": main_effects_path,
        "secondary_effects_path": secondary_effects_path,
        "supplementary_metrics_path": supplementary_path,
    }


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(args.output_root)

    effective_dir = ensure_dir(output_root / "effective_connectivity")
    graph_dir = ensure_dir(output_root / "roi_graph_analysis")
    behavior_dir = ensure_dir(output_root / "behavior_network_coupling")
    benchmark_dir = ensure_dir(output_root / "metric_benchmark")
    supplementary_dir = ensure_dir(output_root / "supplementary_ksg")
    table_dir = ensure_dir(output_root / "paper_summary_tables")

    dcm_results = run_dcm_medication_analysis(effective_dir)
    graph_results = run_roi_graph_reorganization(graph_dir, metric=args.primary_metric)
    behavior_results = run_behavior_network_coupling(
        behavior_dir,
        dcm_results["subject_parameters_path"],
        graph_results["node_delta_path"],
        graph_results["module_summary_path"],
        n_permutations=args.pls_permutations,
    )
    benchmark_results = run_metric_benchmark(
        benchmark_dir,
        behavior_results["subject_behavior_deltas_path"],
        metrics=BENCHMARK_METRICS,
    )
    supplementary_results = run_supplementary_ksg_summary(supplementary_dir)

    included_subjects = sorted(
        set(dcm_results["subjects"])
        & set(graph_results["subjects"])
        & set(behavior_results["subjects"])
    )
    table_results = build_paper_tables(
        table_dir,
        included_subjects,
        dcm_results,
        graph_results,
        behavior_results,
        benchmark_results,
        supplementary_results,
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": REPO_ROOT,
        "output_root": output_root,
        "primary_metric": args.primary_metric,
        "benchmark_metrics": list(BENCHMARK_METRICS),
        "session_mapping": {"session_1": "OFF", "session_2": "ON"},
        "included_subjects": included_subjects,
        "excluded_nodes": ["L Brain-Stem (relative)", "R Brain-Stem (relative)"],
        "dcm_input_root": TMP_DCM_ROOT,
        "analysis_outputs": {
            "effective_connectivity": dcm_results,
            "roi_graph_analysis": graph_results,
            "behavior_network_coupling": behavior_results,
            "metric_benchmark": benchmark_results,
            "supplementary_ksg": supplementary_results,
            "paper_summary_tables": table_results,
        },
        "file_inventory": recursive_file_inventory(output_root),
    }
    write_json(output_root / "analysis_manifest.json", to_serializable(manifest))


if __name__ == "__main__":
    main()
