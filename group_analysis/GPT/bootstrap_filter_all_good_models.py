from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning

from behavior_network_coupling import _align_vector_sign, _zscore_frame
from common_io import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bootstrap neural-weight stability for all exported significant PLS models and "
            "copy the passing models into a new folder."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("results/connectivity/GPT/behavior_network_coupling/all_good_result"),
        help="Folder containing exported significant model subdirectories.",
    )
    parser.add_argument(
        "--pass-dir",
        type=Path,
        default=Path(
            "results/connectivity/GPT/behavior_network_coupling/all_good_result_bootstrap_pass"
        ),
        help="Folder where bootstrap-passing model subdirectories will be copied.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of subject-level bootstrap resamples per model.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Base random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many top absolute neural weights to inspect for CI-based stability.",
    )
    parser.add_argument(
        "--min-bootstrap-median-corr",
        type=float,
        default=0.85,
        help="Minimum median bootstrap correlation between resampled and full neural weight vectors.",
    )
    parser.add_argument(
        "--min-bootstrap-q05-corr",
        type=float,
        default=0.05,
        help="Minimum 5th-percentile bootstrap correlation to full neural weights.",
    )
    parser.add_argument(
        "--min-topk-ci-same-sign-count",
        type=int,
        default=2,
        help=(
            "Minimum number of top-k absolute neural weights whose 95% bootstrap CI excludes "
            "zero in the same direction as the full-model weight."
        ),
    )
    return parser.parse_args()


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _fit_bootstrap_model(
    merged: pd.DataFrame,
    feature_cols: list[str],
    behavior_cols: list[str],
    full_x_weights: np.ndarray,
    sample_idx: np.ndarray,
) -> tuple[np.ndarray, float]:
    X = _zscore_frame(merged.iloc[sample_idx][feature_cols].reset_index(drop=True))
    Y = _zscore_frame(merged.iloc[sample_idx][behavior_cols].reset_index(drop=True))
    model = PLSRegression(n_components=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(X, Y)
    x_weights = _align_vector_sign(full_x_weights, model.x_weights_.ravel())
    score_corr = float(stats.pearsonr(model.x_scores_.ravel(), model.y_scores_.ravel()).statistic)
    return x_weights, score_corr


def _bootstrap_model_dir(
    model_dir: Path,
    n_bootstrap: int,
    random_seed: int,
    top_k: int,
    min_bootstrap_median_corr: float,
    min_bootstrap_q05_corr: float,
    min_topk_ci_same_sign_count: int,
) -> dict:
    summary = _load_json(model_dir / "behavioral_pls_summary.json")
    loadings_df = pd.read_csv(model_dir / "behavioral_pls_loadings.csv")
    neural_df = pd.read_csv(model_dir / "subject_neural_deltas.csv")
    behavior_df = pd.read_csv(model_dir / "subject_behavior_deltas.csv")

    feature_df = loadings_df[loadings_df["set"] == "X"].copy().reset_index(drop=True)
    feature_cols = feature_df["name"].tolist()
    full_x_weights = feature_df["weight"].to_numpy(dtype=float)
    behavior_cols = list(summary["behavior_features"])

    merged = neural_df.merge(behavior_df, on="subject", how="inner").sort_values("subject")
    merged = merged.dropna(subset=feature_cols + behavior_cols).reset_index(drop=True)
    if merged.empty:
        raise ValueError(f"No usable rows for bootstrap in {model_dir}.")

    rng = np.random.default_rng(random_seed)
    x_weight_samples = np.zeros((n_bootstrap, len(feature_cols)), dtype=float)
    vector_rows = []
    for bootstrap_id in range(n_bootstrap):
        sample_idx = rng.integers(0, len(merged), size=len(merged))
        x_weights, score_corr = _fit_bootstrap_model(
            merged,
            feature_cols=feature_cols,
            behavior_cols=behavior_cols,
            full_x_weights=full_x_weights,
            sample_idx=sample_idx,
        )
        x_weight_samples[bootstrap_id] = x_weights
        vector_rows.append(
            {
                "bootstrap_id": bootstrap_id,
                "weights_corr_to_full": _safe_corr(full_x_weights, x_weights),
                "bootstrap_score_correlation": score_corr,
            }
        )

    vector_df = pd.DataFrame(vector_rows)
    vector_path = model_dir / "behavioral_pls_bootstrap_vector_stability.csv"
    vector_df.to_csv(vector_path, index=False)

    x_mean = x_weight_samples.mean(axis=0)
    x_sd = x_weight_samples.std(axis=0, ddof=1)
    ci_low = np.percentile(x_weight_samples, 2.5, axis=0)
    ci_high = np.percentile(x_weight_samples, 97.5, axis=0)
    full_sign = np.sign(full_x_weights)
    same_sign_fraction = np.mean(np.sign(x_weight_samples) == full_sign, axis=0)
    ci_excludes_zero = (ci_low > 0.0) | (ci_high < 0.0)
    ci_same_sign_as_full = ((ci_low > 0.0) & (full_x_weights > 0.0)) | (
        (ci_high < 0.0) & (full_x_weights < 0.0)
    )

    feature_summary_df = pd.DataFrame(
        {
            "name": feature_cols,
            "full_weight": full_x_weights,
            "abs_full_weight": np.abs(full_x_weights),
            "bootstrap_mean_weight": x_mean,
            "bootstrap_sd_weight": x_sd,
            "bootstrap_ratio_mean_over_sd": np.divide(
                x_mean,
                x_sd,
                out=np.full_like(x_mean, np.nan),
                where=x_sd > 0.0,
            ),
            "bootstrap_ci95_low": ci_low,
            "bootstrap_ci95_high": ci_high,
            "bootstrap_same_sign_fraction": same_sign_fraction,
            "bootstrap_ci95_excludes_zero": ci_excludes_zero,
            "bootstrap_ci95_same_sign_as_full": ci_same_sign_as_full,
        }
    ).sort_values("abs_full_weight", ascending=False)
    feature_summary_df["full_abs_rank"] = np.arange(1, len(feature_summary_df) + 1)
    feature_path = model_dir / "behavioral_pls_bootstrap_neural_weight_stability.csv"
    feature_summary_df.to_csv(feature_path, index=False)

    top_feature_df = feature_summary_df.head(min(top_k, len(feature_summary_df))).copy()
    median_corr = float(vector_df["weights_corr_to_full"].median())
    q05_corr = float(vector_df["weights_corr_to_full"].quantile(0.05))
    topk_ci_same_sign_count = int(top_feature_df["bootstrap_ci95_same_sign_as_full"].sum())

    passes_bootstrap = (
        (median_corr >= min_bootstrap_median_corr)
        and (q05_corr >= min_bootstrap_q05_corr)
        and (topk_ci_same_sign_count >= min_topk_ci_same_sign_count)
    )

    bootstrap_summary = {
        "model_dir": str(model_dir),
        "n_subjects": int(merged.shape[0]),
        "n_bootstrap": int(n_bootstrap),
        "top_k": int(min(top_k, len(feature_summary_df))),
        "bootstrap_vector_corr_mean": float(vector_df["weights_corr_to_full"].mean()),
        "bootstrap_vector_corr_median": median_corr,
        "bootstrap_vector_corr_q05": q05_corr,
        "bootstrap_vector_corr_q25": float(vector_df["weights_corr_to_full"].quantile(0.25)),
        "bootstrap_vector_corr_q75": float(vector_df["weights_corr_to_full"].quantile(0.75)),
        "bootstrap_vector_corr_min": float(vector_df["weights_corr_to_full"].min()),
        "bootstrap_score_corr_mean": float(vector_df["bootstrap_score_correlation"].mean()),
        "bootstrap_score_corr_median": float(vector_df["bootstrap_score_correlation"].median()),
        "n_neural_features_ci95_same_sign_as_full": int(
            feature_summary_df["bootstrap_ci95_same_sign_as_full"].sum()
        ),
        "fraction_neural_features_ci95_same_sign_as_full": float(
            feature_summary_df["bootstrap_ci95_same_sign_as_full"].mean()
        ),
        "top_k_feature_names": top_feature_df["name"].tolist(),
        "top_k_ci95_same_sign_as_full_count": topk_ci_same_sign_count,
        "top_k_ci95_same_sign_as_full_fraction": float(
            top_feature_df["bootstrap_ci95_same_sign_as_full"].mean()
        ),
        "pass_criteria": {
            "min_bootstrap_median_corr": float(min_bootstrap_median_corr),
            "min_bootstrap_q05_corr": float(min_bootstrap_q05_corr),
            "min_topk_ci_same_sign_count": int(min_topk_ci_same_sign_count),
        },
        "passes_bootstrap_stability": bool(passes_bootstrap),
        "output_files": {
            "bootstrap_vector_stability": str(vector_path),
            "bootstrap_neural_weight_stability": str(feature_path),
        },
    }
    summary_path = model_dir / "behavioral_pls_bootstrap_summary.json"
    write_json(summary_path, bootstrap_summary)

    return {
        "model_dir": str(model_dir),
        "bootstrap_summary_path": str(summary_path),
        "bootstrap_vector_path": str(vector_path),
        "bootstrap_feature_path": str(feature_path),
        "bootstrap_vector_corr_mean": bootstrap_summary["bootstrap_vector_corr_mean"],
        "bootstrap_vector_corr_median": bootstrap_summary["bootstrap_vector_corr_median"],
        "bootstrap_vector_corr_q05": bootstrap_summary["bootstrap_vector_corr_q05"],
        "top_k_ci95_same_sign_as_full_count": bootstrap_summary[
            "top_k_ci95_same_sign_as_full_count"
        ],
        "top_k_ci95_same_sign_as_full_fraction": bootstrap_summary[
            "top_k_ci95_same_sign_as_full_fraction"
        ],
        "n_neural_features_ci95_same_sign_as_full": bootstrap_summary[
            "n_neural_features_ci95_same_sign_as_full"
        ],
        "fraction_neural_features_ci95_same_sign_as_full": bootstrap_summary[
            "fraction_neural_features_ci95_same_sign_as_full"
        ],
        "passes_bootstrap_stability": bootstrap_summary["passes_bootstrap_stability"],
    }


def run_bootstrap_filter(
    source_dir: Path,
    pass_dir: Path,
    n_bootstrap: int,
    random_seed: int,
    top_k: int,
    min_bootstrap_median_corr: float,
    min_bootstrap_q05_corr: float,
    min_topk_ci_same_sign_count: int,
) -> dict:
    source_dir = Path(source_dir)
    pass_dir = Path(pass_dir)
    if pass_dir.exists():
        raise FileExistsError(f"Pass directory already exists: {pass_dir}")

    index_path = source_dir / "all_good_result_index.csv"
    index_df = pd.read_csv(index_path)

    bootstrap_rows = []
    for row_idx, row in enumerate(index_df.itertuples(index=False), start=1):
        bootstrap_rows.append(
            {
                **row._asdict(),
                **_bootstrap_model_dir(
                    model_dir=Path(row.model_dir),
                    n_bootstrap=n_bootstrap,
                    random_seed=random_seed + row_idx,
                    top_k=top_k,
                    min_bootstrap_median_corr=min_bootstrap_median_corr,
                    min_bootstrap_q05_corr=min_bootstrap_q05_corr,
                    min_topk_ci_same_sign_count=min_topk_ci_same_sign_count,
                ),
            }
        )

    bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(
        [
            "passes_bootstrap_stability",
            "bootstrap_vector_corr_q05",
            "bootstrap_vector_corr_median",
            "permutation_p_two_sided",
        ],
        ascending=[False, False, False, True],
    )
    bootstrap_summary_path = source_dir / "all_good_result_bootstrap_summary.csv"
    bootstrap_df.to_csv(bootstrap_summary_path, index=False)

    pass_df = bootstrap_df[bootstrap_df["passes_bootstrap_stability"]].reset_index(drop=True)
    pass_dir = ensure_dir(pass_dir)
    copied_rows = []
    for rank, row in enumerate(pass_df.itertuples(index=False), start=1):
        source_model_dir = Path(row.model_dir)
        target_dir = pass_dir / f"{rank:03d}_{source_model_dir.name}"
        shutil.copytree(source_model_dir, target_dir)
        copied_rows.append(
            {
                "rank": rank,
                "source_model_dir": str(source_model_dir),
                "pass_model_dir": str(target_dir),
                "brain_set": row.brain_set,
                "behavior_name": row.behavior_name,
                "behavior_cols": row.behavior_cols,
                "permutation_p_two_sided": float(row.permutation_p_two_sided),
                "bootstrap_vector_corr_median": float(row.bootstrap_vector_corr_median),
                "bootstrap_vector_corr_q05": float(row.bootstrap_vector_corr_q05),
                "top_k_ci95_same_sign_as_full_count": int(
                    row.top_k_ci95_same_sign_as_full_count
                ),
            }
        )

    pass_index_df = pd.DataFrame(
        copied_rows,
        columns=[
            "rank",
            "source_model_dir",
            "pass_model_dir",
            "brain_set",
            "behavior_name",
            "behavior_cols",
            "permutation_p_two_sided",
            "bootstrap_vector_corr_median",
            "bootstrap_vector_corr_q05",
            "top_k_ci95_same_sign_as_full_count",
        ],
    )
    pass_index_path = pass_dir / "bootstrap_pass_index.csv"
    pass_index_df.to_csv(pass_index_path, index=False)

    manifest = {
        "source_dir": str(source_dir),
        "source_index_path": str(index_path),
        "source_bootstrap_summary_path": str(bootstrap_summary_path),
        "pass_dir": str(pass_dir),
        "n_source_models": int(index_df.shape[0]),
        "n_pass_models": int(pass_index_df.shape[0]),
        "n_bootstrap": int(n_bootstrap),
        "criteria": {
            "top_k": int(top_k),
            "min_bootstrap_median_corr": float(min_bootstrap_median_corr),
            "min_bootstrap_q05_corr": float(min_bootstrap_q05_corr),
            "min_topk_ci_same_sign_count": int(min_topk_ci_same_sign_count),
        },
        "pass_index_path": str(pass_index_path),
        "pass_models": pass_index_df.to_dict(orient="records"),
    }
    write_json(pass_dir / "bootstrap_pass_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    run_bootstrap_filter(
        source_dir=args.source_dir,
        pass_dir=args.pass_dir,
        n_bootstrap=args.n_bootstrap,
        random_seed=args.random_seed,
        top_k=args.top_k,
        min_bootstrap_median_corr=args.min_bootstrap_median_corr,
        min_bootstrap_q05_corr=args.min_bootstrap_q05_corr,
        min_topk_ci_same_sign_count=args.min_topk_ci_same_sign_count,
    )


if __name__ == "__main__":
    main()
