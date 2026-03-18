from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from behavior_network_coupling import _run_pls_analysis
from common_io import (
    DEPENDENT_CONSECUTIVE_BEHAVIOR_COLS,
    TASK_BEHAVIOR_COLUMN_SPECS,
    behavior_subset_passes_dependency_rule,
    ensure_dir,
    safe_slug,
    write_json,
)


LEGACY_BEHAVIOR_COLS = list(DEPENDENT_CONSECUTIVE_BEHAVIOR_COLS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export all significant brain/behavior PLS models into standalone result folders."
        )
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path(
            "results/connectivity/GPT/behavior_network_coupling/expanded_all_rois_combined_behavior"
        ),
        help="Folder containing subject_neural_deltas.csv and subject_behavior_deltas.csv.",
    )
    parser.add_argument(
        "--family-search-csv",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling/family_level_subset_search.csv"),
        help="Coarse family-level search table.",
    )
    parser.add_argument(
        "--subset-search-csv",
        type=Path,
        default=Path(
            "results/connectivity/GPT/behavior_network_coupling/behavior_subset_family_search/significant_subset_all_brain_families_permutation.csv"
        ),
        help="Subset-level brain-family permutation table.",
    )
    parser.add_argument(
        "--category-recheck-csv",
        type=Path,
        default=Path(
            "results/connectivity/GPT/behavior_network_coupling/behavior_subset_family_search/category_recheck_brain_family_permutation.csv"
        ),
        help="Targeted category recheck table.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling/all_good_result"),
        help="Folder where exported model subdirectories will be written.",
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Raw permutation p-value threshold used to define significant models.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=2000,
        help="Number of permutations for each exported PLS model.",
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


def _sorted_unique_behavior_cols(text: str) -> list[str]:
    return text.split("|")


def _model_key(brain_set: str, behavior_cols: list[str]) -> tuple[str, tuple[str, ...]]:
    return brain_set, tuple(behavior_cols)


def _collect_significant_models(
    family_df: pd.DataFrame,
    subset_df: pd.DataFrame,
    category_df: pd.DataFrame,
    brain_group_map: dict[str, list[str]],
    behavior_group_map: dict[str, list[str]],
    p_threshold: float,
) -> list[dict]:
    selected: dict[tuple[str, tuple[str, ...]], dict] = {}

    def keep_row(row_dict: dict) -> None:
        key = _model_key(row_dict["brain_set"], row_dict["behavior_cols"])
        existing = selected.get(key)
        if existing is None:
            selected[key] = row_dict
            return
        if row_dict["permutation_p_two_sided"] < existing["permutation_p_two_sided"]:
            row_dict["sources"] = sorted(set(existing["sources"] + row_dict["sources"]))
            selected[key] = row_dict
            return
        existing["sources"] = sorted(set(existing["sources"] + row_dict["sources"]))

    for row in family_df[family_df["permutation_p_two_sided"] < p_threshold].itertuples(index=False):
        behavior_cols = _resolve_selected_columns(row.behavior_set, behavior_group_map)
        if not behavior_subset_passes_dependency_rule(behavior_cols):
            continue
        keep_row(
            {
                "source_label": "family_level",
                "sources": ["family_level"],
                "brain_set": row.brain_set,
                "behavior_name": row.behavior_set,
                "behavior_cols": behavior_cols,
                "feature_cols": _resolve_selected_columns(row.brain_set, brain_group_map),
                "n_behavior_features": int(row.n_behavior_features),
                "n_brain_features": int(row.n_brain_features),
                "observed_score_correlation": float(row.observed_score_correlation),
                "permutation_p_two_sided": float(row.permutation_p_two_sided),
            }
        )

    for source_name, frame in [
        ("subset_search", subset_df),
        ("category_recheck", category_df),
    ]:
        for row in frame[frame["permutation_p_two_sided"] < p_threshold].itertuples(index=False):
            behavior_cols = _sorted_unique_behavior_cols(row.behavior_subset)
            if not behavior_subset_passes_dependency_rule(behavior_cols):
                continue
            keep_row(
                {
                    "source_label": source_name,
                    "sources": [source_name],
                    "brain_set": row.brain_set,
                    "behavior_name": row.behavior_subset,
                    "behavior_cols": behavior_cols,
                    "feature_cols": _resolve_selected_columns(row.brain_set, brain_group_map),
                    "n_behavior_features": int(row.n_behavior_features),
                    "n_brain_features": int(row.n_brain_features),
                    "observed_score_correlation": float(row.observed_score_correlation),
                    "permutation_p_two_sided": float(row.permutation_p_two_sided),
                }
            )

    return sorted(
        selected.values(),
        key=lambda item: (
            item["permutation_p_two_sided"],
            -item["observed_score_correlation"],
            item["brain_set"],
            item["behavior_name"],
        ),
    )


def export_all_significant_models(
    analysis_dir: Path,
    family_search_csv: Path,
    subset_search_csv: Path,
    category_recheck_csv: Path,
    out_dir: Path,
    p_threshold: float,
    n_permutations: int,
) -> dict:
    neural_df = pd.read_csv(analysis_dir / "subject_neural_deltas.csv")
    behavior_df = pd.read_csv(analysis_dir / "subject_behavior_deltas.csv")
    merged = neural_df.merge(behavior_df, on="subject", how="inner").sort_values("subject")

    family_df = pd.read_csv(family_search_csv)
    subset_df = pd.read_csv(subset_search_csv)
    category_df = pd.read_csv(category_recheck_csv)

    brain_group_map = _build_brain_family_columns(neural_df)
    behavior_group_map = _build_behavior_group_columns(behavior_df)
    behavior_label_map = _build_behavior_label_map()

    models = _collect_significant_models(
        family_df=family_df,
        subset_df=subset_df,
        category_df=category_df,
        brain_group_map=brain_group_map,
        behavior_group_map=behavior_group_map,
        p_threshold=p_threshold,
    )
    if not models:
        raise ValueError(f"No significant models found at p < {p_threshold}.")

    out_dir = ensure_dir(out_dir)
    saved_rows = []
    for rank, model in enumerate(models, start=1):
        behavior_slug = safe_slug("__".join(model["behavior_cols"]))
        model_dir = ensure_dir(
            out_dir / f"{rank:03d}_{safe_slug(model['brain_set'])}__{behavior_slug}"
        )
        results = _run_pls_analysis(
            model_dir,
            merged,
            feature_cols=model["feature_cols"],
            behavior_cols=model["behavior_cols"],
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
                "brain_set": model["brain_set"],
                "behavior_name": model["behavior_name"],
                "behavior_cols": "|".join(model["behavior_cols"]),
                "sources": "|".join(model["sources"]),
                "n_brain_features": model["n_brain_features"],
                "n_behavior_features": model["n_behavior_features"],
                "observed_score_correlation": model["observed_score_correlation"],
                "permutation_p_two_sided": model["permutation_p_two_sided"],
                "model_dir": str(model_dir),
                "summary_path": str(results["summary_path"]),
                "figure_path": str(results["figure_path"]),
                "loadings_path": str(results["loadings_path"]),
                "scores_path": str(results["scores_path"]),
                "permutation_path": str(results["permutation_path"]),
                "loo_path": str(results["loo_path"]),
                "behavior_weight_path": str(results["behavior_weight_path"]),
            }
        )

    saved_df = pd.DataFrame(saved_rows)
    index_path = out_dir / "all_good_result_index.csv"
    saved_df.to_csv(index_path, index=False)

    manifest = {
        "analysis_dir": str(analysis_dir),
        "family_search_csv": str(family_search_csv),
        "subset_search_csv": str(subset_search_csv),
        "category_recheck_csv": str(category_recheck_csv),
        "p_threshold": float(p_threshold),
        "n_permutations": int(n_permutations),
        "n_saved_models": int(saved_df.shape[0]),
        "index_path": str(index_path),
        "saved_models": saved_df.to_dict(orient="records"),
    }
    write_json(out_dir / "all_good_result_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    export_all_significant_models(
        analysis_dir=args.analysis_dir,
        family_search_csv=args.family_search_csv,
        subset_search_csv=args.subset_search_csv,
        category_recheck_csv=args.category_recheck_csv,
        out_dir=args.out_dir,
        p_threshold=args.p_threshold,
        n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
