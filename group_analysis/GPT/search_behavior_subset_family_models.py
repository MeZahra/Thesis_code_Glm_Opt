from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

from common_io import ensure_dir, write_json


LEGACY_BEHAVIOR_COLS = [
    "behavior_vigor_delta",
    "behavior_lag1_corr_delta",
    "behavior_consistency_improvement_delta",
]

TIMING_BEHAVIOR_COLS = [
    "task_1_pt_delta",
    "task_1_rt_delta",
    "task_1_mt_delta",
    "task_1_rt_mt_delta",
]

PEAK_BEHAVIOR_COLS = [
    "task_vmax_delta",
    "task_pmax_delta",
]

ALL_BEHAVIOR_COLS = LEGACY_BEHAVIOR_COLS + TIMING_BEHAVIOR_COLS + PEAK_BEHAVIOR_COLS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Systematically scan behavior-feature subsets against brain-family combinations "
            "using single-component PLS."
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
        "--out-dir",
        type=Path,
        default=Path(
            "results/connectivity/GPT/behavior_network_coupling/behavior_subset_family_search"
        ),
        help="Output folder for exhaustive scan tables.",
    )
    parser.add_argument(
        "--min-behavior-subset-size",
        type=int,
        default=1,
        help="Minimum number of behavior features in a subset.",
    )
    parser.add_argument(
        "--max-behavior-subset-size",
        type=int,
        default=len(ALL_BEHAVIOR_COLS),
        help="Maximum number of behavior features in a subset.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=2000,
        help="Number of Y-row permutations for the best brain family model per behavior subset.",
    )
    return parser.parse_args()


def _zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    centered = df - df.mean(axis=0)
    scale = df.std(axis=0, ddof=0).replace(0.0, 1.0)
    return centered / scale


def _pls_score_correlation(
    merged: pd.DataFrame,
    neural_cols: list[str],
    behavior_cols: list[str],
) -> float:
    X = _zscore_frame(merged[neural_cols])
    Y = _zscore_frame(merged[behavior_cols])
    model = PLSRegression(n_components=1)
    model.fit(X, Y)
    return float(stats.pearsonr(model.x_scores_.ravel(), model.y_scores_.ravel()).statistic)


def _pls_permutation_pvalue(
    merged: pd.DataFrame,
    neural_cols: list[str],
    behavior_cols: list[str],
    permutations: np.ndarray,
) -> tuple[float, float]:
    X = _zscore_frame(merged[neural_cols])
    Y = _zscore_frame(merged[behavior_cols])
    model = PLSRegression(n_components=1)
    model.fit(X, Y)
    observed = float(stats.pearsonr(model.x_scores_.ravel(), model.y_scores_.ravel()).statistic)

    ge = 0
    for perm in permutations:
        perm_model = PLSRegression(n_components=1)
        perm_model.fit(X, Y.iloc[perm].reset_index(drop=True))
        perm_corr = float(
            stats.pearsonr(perm_model.x_scores_.ravel(), perm_model.y_scores_.ravel()).statistic
        )
        ge += abs(perm_corr) >= abs(observed)

    pvalue = float((ge + 1) / (len(permutations) + 1))
    return observed, pvalue


def _bh_fdr(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    scaled = ranked * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    out = np.empty_like(scaled)
    out[order] = scaled
    return pd.Series(out, index=pvalues.index)


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


def _iter_behavior_subsets(
    behavior_cols: list[str],
    min_size: int,
    max_size: int,
):
    upper = min(max_size, len(behavior_cols))
    for size in range(max(min_size, 1), upper + 1):
        yield from combinations(behavior_cols, size)


def _iter_brain_sets(group_map: dict[str, list[str]]):
    keys = list(group_map)
    for size in range(1, len(keys) + 1):
        for combo in combinations(keys, size):
            cols: list[str] = []
            for key in combo:
                cols.extend(group_map[key])
            yield "+".join(combo), cols


def _count_subset_membership(behavior_subset: list[str]) -> dict[str, int]:
    subset = set(behavior_subset)
    return {
        "n_legacy_features": sum(col in subset for col in LEGACY_BEHAVIOR_COLS),
        "n_timing_features": sum(col in subset for col in TIMING_BEHAVIOR_COLS),
        "n_peak_features": sum(col in subset for col in PEAK_BEHAVIOR_COLS),
    }


def _subset_family_label(behavior_subset: list[str]) -> str:
    counts = _count_subset_membership(behavior_subset)
    labels = []
    if counts["n_legacy_features"]:
        labels.append("legacy")
    if counts["n_timing_features"]:
        labels.append("timing")
    if counts["n_peak_features"]:
        labels.append("peaks")
    return "+".join(labels)


def run_behavior_subset_family_search(
    analysis_dir: Path,
    out_dir: Path,
    min_behavior_subset_size: int,
    max_behavior_subset_size: int,
    n_permutations: int,
) -> dict:
    out_dir = ensure_dir(out_dir)

    neural_df = pd.read_csv(analysis_dir / "subject_neural_deltas.csv")
    behavior_df = pd.read_csv(analysis_dir / "subject_behavior_deltas.csv")

    available_behavior_cols = [col for col in ALL_BEHAVIOR_COLS if col in behavior_df.columns]
    merged = neural_df.merge(
        behavior_df[["subject"] + available_behavior_cols],
        on="subject",
        how="inner",
    ).sort_values("subject")

    brain_group_map = _build_brain_family_columns(neural_df)
    brain_sets = list(_iter_brain_sets(brain_group_map))

    observed_rows = []
    best_rows = []
    for behavior_subset in _iter_behavior_subsets(
        available_behavior_cols,
        min_size=min_behavior_subset_size,
        max_size=max_behavior_subset_size,
    ):
        behavior_cols = list(behavior_subset)
        subset_counts = _count_subset_membership(behavior_cols)
        subset_key = "|".join(behavior_cols)
        subset_family_label = _subset_family_label(behavior_cols)

        best_row = None
        for brain_set_name, neural_cols in brain_sets:
            usable = merged.dropna(subset=neural_cols + behavior_cols).reset_index(drop=True)
            if usable.empty:
                continue
            score_corr = _pls_score_correlation(usable, neural_cols, behavior_cols)
            row = {
                "brain_set": brain_set_name,
                "behavior_subset": subset_key,
                "behavior_family_label": subset_family_label,
                "n_brain_features": len(neural_cols),
                "n_behavior_features": len(behavior_cols),
                "n_subjects": int(usable.shape[0]),
                "observed_score_correlation": score_corr,
                **subset_counts,
            }
            observed_rows.append(row)
            if best_row is None or (
                row["observed_score_correlation"] > best_row["observed_score_correlation"]
            ):
                best_row = row

        if best_row is not None:
            best_rows.append(best_row)

    observed_df = (
        pd.DataFrame(observed_rows)
        .sort_values(
            ["observed_score_correlation", "n_brain_features", "n_behavior_features"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    observed_path = out_dir / "observed_all_models.csv"
    observed_df.to_csv(observed_path, index=False)

    best_df = (
        pd.DataFrame(best_rows)
        .sort_values(
            ["observed_score_correlation", "n_brain_features", "n_behavior_features"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )
    best_observed_path = out_dir / "best_brain_set_per_behavior_subset.csv"
    best_df.to_csv(best_observed_path, index=False)

    rng = np.random.default_rng(0)
    permutations = np.array([rng.permutation(len(merged)) for _ in range(n_permutations)])

    permutation_rows = []
    brain_set_to_cols = {name: cols for name, cols in brain_sets}
    for row in best_df.itertuples(index=False):
        behavior_cols = row.behavior_subset.split("|")
        neural_cols = brain_set_to_cols[row.brain_set]
        usable = merged.dropna(subset=neural_cols + behavior_cols).reset_index(drop=True)
        observed, pvalue = _pls_permutation_pvalue(
            usable,
            neural_cols=neural_cols,
            behavior_cols=behavior_cols,
            permutations=permutations,
        )
        permutation_rows.append(
            {
                **row._asdict(),
                "refit_observed_score_correlation": observed,
                "permutation_p_two_sided": pvalue,
            }
        )

    permutation_df = (
        pd.DataFrame(permutation_rows)
        .sort_values(
            ["permutation_p_two_sided", "refit_observed_score_correlation"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )
    permutation_df["q_fdr_best_subset_models"] = _bh_fdr(
        permutation_df["permutation_p_two_sided"]
    )
    permutation_path = out_dir / "best_brain_set_per_behavior_subset_permutation.csv"
    permutation_df.to_csv(permutation_path, index=False)

    summary = {
        "analysis_dir": str(analysis_dir),
        "n_subjects": int(merged.shape[0]),
        "n_behavior_features_available": len(available_behavior_cols),
        "behavior_features_available": available_behavior_cols,
        "n_brain_families": len(brain_group_map),
        "brain_family_feature_counts": {key: len(value) for key, value in brain_group_map.items()},
        "n_brain_family_combinations": len(brain_sets),
        "min_behavior_subset_size": int(min_behavior_subset_size),
        "max_behavior_subset_size": int(min(max_behavior_subset_size, len(available_behavior_cols))),
        "n_behavior_subsets_tested": int(best_df.shape[0]),
        "n_observed_models": int(observed_df.shape[0]),
        "n_permutations": int(n_permutations),
        "n_best_subset_models_p_below_0_05": int(
            (permutation_df["permutation_p_two_sided"] < 0.05).sum()
        ),
        "n_best_subset_models_q_below_0_05": int(
            (permutation_df["q_fdr_best_subset_models"] < 0.05).sum()
        ),
        "best_observed_model": observed_df.head(1).to_dict(orient="records"),
        "best_permuted_subset_model": permutation_df.head(1).to_dict(orient="records"),
        "note": (
            "Permutation p-values are computed only for the best brain family combination "
            "picked separately for each behavior subset. They remain exploratory because "
            "brain-family selection is data-driven."
        ),
        "output_files": {
            "observed_all_models": str(observed_path),
            "best_brain_set_per_behavior_subset": str(best_observed_path),
            "best_brain_set_per_behavior_subset_permutation": str(permutation_path),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    run_behavior_subset_family_search(
        analysis_dir=args.analysis_dir,
        out_dir=args.out_dir,
        min_behavior_subset_size=args.min_behavior_subset_size,
        max_behavior_subset_size=args.max_behavior_subset_size,
        n_permutations=args.n_permutations,
    )


if __name__ == "__main__":
    main()
