from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from behavior_network_coupling import _run_pls_analysis
from common_io import TASK_BEHAVIOR_COLUMN_SPECS, ensure_dir, safe_slug, write_json


LEGACY_BEHAVIOR_COLS = [
    "behavior_vigor_delta",
    "behavior_lag1_corr_delta",
    "behavior_consistency_improvement_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save separate PLS result folders for family-level models that passed a p-value threshold."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling"),
        help="Root folder containing expanded_all_rois_combined_behavior and family_level_subset_search.csv.",
    )
    parser.add_argument(
        "--family-search-csv",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling/family_level_subset_search.csv"),
        help="CSV table ranking family-level grouped PLS models.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling/family_level_saved_models"),
        help="Folder where separate result directories will be written.",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Raw permutation p-value threshold used to select models from the family-level search table.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=2000,
        help="Number of permutations for each saved PLS model.",
    )
    return parser.parse_args()


def _build_brain_family_columns(neural_df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "dcm": [col for col in neural_df.columns if col.startswith("dcm_outgoing_delta_")],
        "graph_strength": [
            col for col in neural_df.columns if col.startswith("graph_strength_delta_")
        ],
        "graph_participation": [
            col for col in neural_df.columns if col.startswith("graph_participation_delta_")
        ],
        "module": [col for col in neural_df.columns if col.startswith("module_delta_")],
    }


def _build_behavior_group_columns(behavior_df: pd.DataFrame) -> dict[str, list[str]]:
    task_timing_cols = [
        "task_1_pt_delta",
        "task_1_rt_delta",
        "task_1_mt_delta",
        "task_1_rt_mt_delta",
    ]
    task_peak_cols = ["task_vmax_delta", "task_pmax_delta"]
    available_cols = set(behavior_df.columns)
    return {
        "legacy": [col for col in LEGACY_BEHAVIOR_COLS if col in available_cols],
        "timing": [col for col in task_timing_cols if col in available_cols],
        "peaks": [col for col in task_peak_cols if col in available_cols],
    }


def _resolve_selected_columns(selection: str, group_map: dict[str, list[str]]) -> list[str]:
    cols: list[str] = []
    for key in selection.split("+"):
        if key not in group_map:
            raise KeyError(f"Unknown group key '{key}' in selection '{selection}'.")
        cols.extend(group_map[key])
    return cols


def _build_behavior_label_map() -> dict[str, str]:
    task_labels = {
        f"{spec['key']}_delta": f"{spec['label']} session mean delta (ON - OFF)"
        for spec in TASK_BEHAVIOR_COLUMN_SPECS
    }
    return {
        "behavior_vigor_delta": "Behavior vigor delta",
        "behavior_lag1_corr_delta": "Behavior lag-1 correlation delta",
        "behavior_consistency_improvement_delta": "Behavior consistency improvement delta",
        **task_labels,
    }


def run_family_level_model_exports(
    root_dir: Path,
    family_search_csv: Path,
    out_dir: Path,
    p_threshold: float,
    n_permutations: int,
) -> dict:
    analysis_dir = root_dir / "expanded_all_rois_combined_behavior"
    neural_df = pd.read_csv(analysis_dir / "subject_neural_deltas.csv")
    behavior_df = pd.read_csv(analysis_dir / "subject_behavior_deltas.csv")
    merged = neural_df.merge(behavior_df, on="subject", how="inner").sort_values("subject")
    family_df = pd.read_csv(family_search_csv).sort_values(
        ["permutation_p_two_sided", "observed_score_correlation"], ascending=[True, False]
    )
    selected_df = family_df[family_df["permutation_p_two_sided"] < p_threshold].reset_index(
        drop=True
    )
    if selected_df.empty:
        raise ValueError(f"No family-level models met p < {p_threshold}.")

    out_dir = ensure_dir(out_dir)
    brain_family_cols = _build_brain_family_columns(neural_df)
    behavior_group_cols = _build_behavior_group_columns(behavior_df)
    behavior_label_map = _build_behavior_label_map()

    saved_rows = []
    for rank, row in enumerate(selected_df.itertuples(index=False), start=1):
        feature_cols = _resolve_selected_columns(row.brain_set, brain_family_cols)
        behavior_cols = _resolve_selected_columns(row.behavior_set, behavior_group_cols)
        model_dir = ensure_dir(
            out_dir / f"{rank:02d}_{safe_slug(row.brain_set)}__{safe_slug(row.behavior_set)}"
        )
        results = _run_pls_analysis(
            model_dir,
            merged,
            feature_cols=feature_cols,
            behavior_cols=behavior_cols,
            n_permutations=n_permutations,
            scatter_col="y_score",
            scatter_label="PLS behavior score",
            scatter_title="Brain Score vs Behavior Score",
            scatter_corr_method="pearson",
            behavior_label_map=behavior_label_map,
        )
        saved_rows.append(
            {
                "rank": rank,
                "brain_set": row.brain_set,
                "behavior_set": row.behavior_set,
                "n_brain_features": int(row.n_brain_features),
                "n_behavior_features": int(row.n_behavior_features),
                "observed_score_correlation": float(row.observed_score_correlation),
                "permutation_p_two_sided": float(row.permutation_p_two_sided),
                "q_fdr": float(row.q_fdr),
                "model_dir": str(model_dir),
                "summary_path": str(results["summary_path"]),
                "loadings_path": str(results["loadings_path"]),
                "scores_path": str(results["scores_path"]),
                "permutation_path": str(results["permutation_path"]),
                "loo_path": str(results["loo_path"]),
                "figure_path": str(results["figure_path"]),
                "behavior_weight_path": str(results["behavior_weight_path"]),
            }
        )

    saved_df = pd.DataFrame(saved_rows)
    saved_index_path = out_dir / "saved_model_index.csv"
    saved_df.to_csv(saved_index_path, index=False)

    manifest = {
        "root_dir": str(root_dir),
        "analysis_dir": str(analysis_dir),
        "family_search_csv": str(family_search_csv),
        "p_threshold": float(p_threshold),
        "n_permutations": int(n_permutations),
        "n_saved_models": int(saved_df.shape[0]),
        "saved_model_index_path": str(saved_index_path),
        "saved_models": saved_df.to_dict(orient="records"),
    }
    write_json(out_dir / "saved_model_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_family_level_model_exports(
        root_dir=args.root_dir,
        family_search_csv=args.family_search_csv,
        out_dir=args.out_dir,
        p_threshold=args.p_threshold,
        n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
