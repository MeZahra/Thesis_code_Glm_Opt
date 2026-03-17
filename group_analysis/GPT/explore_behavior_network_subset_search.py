from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

from common_io import ensure_dir, write_json


COMBINED_BEHAVIOR_COLS = [
    "behavior_vigor_delta",
    "behavior_lag1_corr_delta",
    "behavior_consistency_improvement_delta",
    "task_1_pt_delta",
    "task_1_rt_delta",
    "task_1_mt_delta",
    "task_1_rt_mt_delta",
    "task_vmax_delta",
    "task_pmax_delta",
]

TASK_ONLY_BEHAVIOR_COLS = [
    "task_1_pt_delta",
    "task_1_rt_delta",
    "task_1_mt_delta",
    "task_1_rt_mt_delta",
    "task_vmax_delta",
    "task_pmax_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory subset search for brain-behavior PLS models."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling"),
        help="Root folder containing expanded_all_rois_* analysis outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling/exploratory_subset_search"),
        help="Output folder for exploratory search results.",
    )
    parser.add_argument(
        "--top-neural-features",
        type=int,
        default=10,
        help="Neural features shortlisted from the 1x1 screen for subset search.",
    )
    parser.add_argument(
        "--max-neural-subset-size",
        type=int,
        default=3,
        help="Largest neural subset size to consider in the observed-correlation search.",
    )
    parser.add_argument(
        "--max-behavior-subset-size",
        type=int,
        default=3,
        help="Largest behavior subset size to consider in the observed-correlation search.",
    )
    parser.add_argument(
        "--n-finalists",
        type=int,
        default=80,
        help="Number of top observed-correlation subsets to refit with permutation testing.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=5000,
        help="Number of Y-row permutations for finalist PLS models.",
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
        Y_perm = Y.iloc[perm].reset_index(drop=True)
        perm_model = PLSRegression(n_components=1)
        perm_model.fit(X, Y_perm)
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


def _iter_subsets(items: list[str], max_size: int):
    for size in range(1, min(max_size, len(items)) + 1):
        yield from combinations(items, size)


def _one_by_one_screen(
    merged: pd.DataFrame,
    neural_cols: list[str],
    behavior_cols: list[str],
) -> pd.DataFrame:
    rows = []
    for neural in neural_cols:
        for behavior in behavior_cols:
            pearson = stats.pearsonr(merged[neural], merged[behavior])
            rows.append(
                {
                    "neural_feature": neural,
                    "behavior_feature": behavior,
                    "pearson_r": float(pearson.statistic),
                    "pearson_p_two_sided": float(pearson.pvalue),
                    "abs_pearson_r": float(abs(pearson.statistic)),
                }
            )
    screen_df = pd.DataFrame(rows).sort_values(
        ["pearson_p_two_sided", "abs_pearson_r"], ascending=[True, False]
    )
    screen_df["q_fdr"] = _bh_fdr(screen_df["pearson_p_two_sided"])
    return screen_df.reset_index(drop=True)


def _rank_neural_features(screen_df: pd.DataFrame, top_n: int) -> list[str]:
    ranked = (
        screen_df.groupby("neural_feature", as_index=False)
        .agg(best_p=("pearson_p_two_sided", "min"), best_abs_r=("abs_pearson_r", "max"))
        .sort_values(["best_p", "best_abs_r"], ascending=[True, False])
    )
    return ranked.head(top_n)["neural_feature"].tolist()


def _observed_subset_search(
    merged: pd.DataFrame,
    neural_candidates: list[str],
    behavior_cols: list[str],
    max_neural_subset_size: int,
    max_behavior_subset_size: int,
) -> pd.DataFrame:
    rows = []
    for neural_subset in _iter_subsets(neural_candidates, max_neural_subset_size):
        neural_list = list(neural_subset)
        for behavior_subset in _iter_subsets(behavior_cols, max_behavior_subset_size):
            behavior_list = list(behavior_subset)
            score_corr = _pls_score_correlation(merged, neural_list, behavior_list)
            rows.append(
                {
                    "n_neural_features": len(neural_list),
                    "n_behavior_features": len(behavior_list),
                    "neural_features": "|".join(neural_list),
                    "behavior_features": "|".join(behavior_list),
                    "observed_score_correlation": score_corr,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["observed_score_correlation", "n_neural_features", "n_behavior_features"],
            ascending=[False, True, True],
        )
        .reset_index(drop=True)
    )


def _permutation_refit_finalists(
    merged: pd.DataFrame,
    observed_df: pd.DataFrame,
    n_finalists: int,
    n_permutations: int,
) -> pd.DataFrame:
    finalists = observed_df.head(n_finalists).copy()
    rng = np.random.default_rng(0)
    permutations = np.array([rng.permutation(len(merged)) for _ in range(n_permutations)])

    rows = []
    for _, row in finalists.iterrows():
        neural_cols = row["neural_features"].split("|")
        behavior_cols = row["behavior_features"].split("|")
        observed, pvalue = _pls_permutation_pvalue(
            merged, neural_cols, behavior_cols, permutations
        )
        rows.append(
            {
                **row.to_dict(),
                "refit_observed_score_correlation": observed,
                "permutation_p_two_sided": pvalue,
            }
        )

    results = pd.DataFrame(rows).sort_values(
        ["permutation_p_two_sided", "refit_observed_score_correlation"],
        ascending=[True, False],
    )
    results["q_fdr_finalists_only"] = _bh_fdr(results["permutation_p_two_sided"])
    return results.reset_index(drop=True)


def run_search_for_analysis(
    analysis_name: str,
    root_dir: Path,
    out_dir: Path,
    behavior_cols: list[str],
    top_neural_features: int,
    max_neural_subset_size: int,
    max_behavior_subset_size: int,
    n_finalists: int,
    n_permutations: int,
) -> dict:
    analysis_dir = root_dir / analysis_name
    out_dir = ensure_dir(out_dir / analysis_name)

    neural_df = pd.read_csv(analysis_dir / "subject_neural_deltas.csv")
    behavior_df = pd.read_csv(analysis_dir / "subject_behavior_deltas.csv")
    merged = neural_df.merge(behavior_df, on="subject", how="inner").sort_values("subject")
    neural_cols = [col for col in neural_df.columns if col != "subject"]
    merged = merged.dropna(subset=neural_cols + behavior_cols).reset_index(drop=True)

    screen_df = _one_by_one_screen(merged, neural_cols, behavior_cols)
    screen_path = out_dir / "screen_1x1.csv"
    screen_df.to_csv(screen_path, index=False)

    neural_candidates = _rank_neural_features(screen_df, top_neural_features)
    observed_df = _observed_subset_search(
        merged,
        neural_candidates,
        behavior_cols,
        max_neural_subset_size=max_neural_subset_size,
        max_behavior_subset_size=max_behavior_subset_size,
    )
    observed_path = out_dir / "observed_subset_rankings.csv"
    observed_df.to_csv(observed_path, index=False)

    finalist_df = _permutation_refit_finalists(
        merged,
        observed_df,
        n_finalists=n_finalists,
        n_permutations=n_permutations,
    )
    finalist_path = out_dir / "finalist_permutation_results.csv"
    finalist_df.to_csv(finalist_path, index=False)

    summary = {
        "analysis_name": analysis_name,
        "n_subjects": int(merged.shape[0]),
        "n_neural_features_total": int(len(neural_cols)),
        "n_behavior_features_total": int(len(behavior_cols)),
        "behavior_features_tested": behavior_cols,
        "top_neural_candidates": neural_candidates,
        "n_1x1_models": int(screen_df.shape[0]),
        "n_observed_subset_models": int(observed_df.shape[0]),
        "n_finalists_permutation_tested": int(finalist_df.shape[0]),
        "best_1x1_model": screen_df.head(1).to_dict(orient="records"),
        "best_finalist": finalist_df.head(1).to_dict(orient="records"),
        "n_finalists_p_below_0_05": int((finalist_df["permutation_p_two_sided"] < 0.05).sum()),
        "n_finalists_q_below_0_05": int(
            (finalist_df["q_fdr_finalists_only"] < 0.05).sum()
        ),
        "note": (
            "This is exploratory subset selection on the same dataset used for testing. "
            "Permutation p-values here are useful for ranking candidates but are not "
            "confirmatory after post-hoc model search."
        ),
        "output_files": {
            "screen_1x1": str(screen_path),
            "observed_subset_rankings": str(observed_path),
            "finalist_permutation_results": str(finalist_path),
        },
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    out_root = ensure_dir(args.out_dir)

    task_summary = run_search_for_analysis(
        analysis_name="expanded_all_rois_task_only",
        root_dir=args.root_dir,
        out_dir=out_root,
        behavior_cols=TASK_ONLY_BEHAVIOR_COLS,
        top_neural_features=args.top_neural_features,
        max_neural_subset_size=args.max_neural_subset_size,
        max_behavior_subset_size=args.max_behavior_subset_size,
        n_finalists=args.n_finalists,
        n_permutations=args.n_permutations,
    )
    combined_summary = run_search_for_analysis(
        analysis_name="expanded_all_rois_combined_behavior",
        root_dir=args.root_dir,
        out_dir=out_root,
        behavior_cols=COMBINED_BEHAVIOR_COLS,
        top_neural_features=args.top_neural_features,
        max_neural_subset_size=args.max_neural_subset_size,
        max_behavior_subset_size=args.max_behavior_subset_size,
        n_finalists=args.n_finalists,
        n_permutations=args.n_permutations,
    )

    write_json(
        out_root / "summary.json",
        {
            "task_only": task_summary,
            "combined_behavior": combined_summary,
        },
    )


if __name__ == "__main__":
    main()
