#!/usr/bin/env python3
"""Subject-level comparison plots for projection metrics by ses/run and session-level KS.

This script mirrors the projection/run bookkeeping from `motor_brain_com.py`,
but focuses only on projection metrics (no behavior comparison).

Outputs:
1) Per-run/session variance CSV.
2) Per-run/session std/rms CSV.
3) Variability figure (2x2): ses1-run1, ses1-run2, ses2-run1, ses2-run2.
4) std/rms figure (2x2): ses1-run1, ses1-run2, ses2-run1, ses2-run2.
5) Per-subject KS stats CSV (session 1 vs 2, runs 1+2 pooled within session).
6) Group-level KS inference CSV.
7) KS summary figure (3x1): KS statistic, IQR diff, MAD diff.
8) Per-subject pooled-trial metric CSV (session 1 vs 2, runs 1+2 pooled).
9) Group-level pooled-trial metric session comparison stats CSV.
10) Pooled-trial metric overlay figures with KS/IQR/MAD summary text (non-CV metrics).
11) Subject distribution grid figure (2x7): session 1 vs 2 projected-signal densities.
12) Pooled-trial CV paired box plot with subject-wise lines (session 1 vs 2).
13) Consecutive-trial scatter plots for all paired subjects (left=session 1, right=session 2).
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import combine_pvalues, ks_2samp, ttest_rel, wilcoxon

from motor_brain_com import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TRIAL_KEEP_ROOT,
    compute_run_variances,
    split_projection_by_run,
)

VARIANCE_SCALE = 1e7
PAIRWISE_ALPHA = 0.05
KS_ALPHA = 0.05

POOLED_METRIC_SPECS = [
    {
        "key": "cv_projection",
        "label": "CV",
        "formula": "std(X) / |mean(X)|",
        "file_stub": "pooled_cv",
        "x_label": "Pooled-trial CV projection (runs 1+2)",
        "title": "Session 1 vs Session 2 pooled-trial CV distribution",
    },
    {
        "key": "mad_mean_projection",
        "label": "MAD(mean)",
        "formula": "mean(|X - mean(X)|)",
        "file_stub": "pooled_mad_mean",
        "x_label": "Pooled-trial MAD(mean) projection (runs 1+2)",
        "title": "Session 1 vs Session 2 pooled-trial MAD(mean) distribution",
    },
    {
        "key": "mad_mean_over_median_projection",
        "label": "MAD(mean)/|median|",
        "formula": "mean(|X - mean(X)|) / |median(X)|",
        "file_stub": "pooled_mad_mean_over_median",
        "x_label": "Pooled-trial MAD(mean)/|median| projection (runs 1+2)",
        "title": "Session 1 vs Session 2 pooled-trial MAD(mean)/|median| distribution",
    },
    {
        "key": "std_centered_range_projection",
        "label": "std((X-mean)/(max-min))",
        "formula": "std((X - mean(X)) / (max(X) - min(X)))",
        "file_stub": "pooled_std_centered_range",
        "x_label": "Pooled-trial std((X-mean)/(max-min)) projection (runs 1+2)",
        "title": "Session 1 vs Session 2 pooled-trial centered-range-normalized std",
    },
]


def build_projection_variability_stats_table(
    run_variance_df,
    value_col="variance_projection",
    alpha=PAIRWISE_ALPHA,
):
    comparisons = [
        ("session1_run1_vs_run2", (1, 1), (1, 2)),
        ("session2_run1_vs_run2", (2, 1), (2, 2)),
        ("run1_session1_vs_session2", (1, 1), (2, 1)),
        ("run2_session1_vs_session2", (1, 2), (2, 2)),
    ]

    df = run_variance_df[["sub_tag", "ses", "run", value_col]].copy()
    df["sub_tag"] = df["sub_tag"].astype(str)
    df["ses"] = pd.to_numeric(df["ses"], errors="coerce")
    df["run"] = pd.to_numeric(df["run"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    finite_mask = (
        np.isfinite(df["ses"].to_numpy(dtype=np.float64))
        & np.isfinite(df["run"].to_numpy(dtype=np.float64))
    )
    df = df.loc[finite_mask].copy()
    df["ses"] = df["ses"].astype(int)
    df["run"] = df["run"].astype(int)

    pivot = df.pivot_table(
        index="sub_tag",
        columns=["ses", "run"],
        values=value_col,
        aggfunc="mean",
    )

    rows = []
    for comparison_name, key_a, key_b in comparisons:
        row = {
            "comparison": comparison_name,
            "group_a": f"ses{key_a[0]}_run{key_a[1]}",
            "group_b": f"ses{key_b[0]}_run{key_b[1]}",
            "n_subjects_paired": 0,
            "mean_group_a": np.nan,
            "mean_group_b": np.nan,
            "mean_diff_group_a_minus_b": np.nan,
            "median_diff_group_a_minus_b": np.nan,
            "ttest_stat": np.nan,
            "ttest_p_two_sided": np.nan,
            "wilcoxon_stat": np.nan,
            "wilcoxon_p_two_sided": np.nan,
        }

        if key_a not in pivot.columns or key_b not in pivot.columns:
            rows.append(row)
            continue

        paired = pivot.loc[:, [key_a, key_b]].dropna()
        n_pairs = int(len(paired))
        row["n_subjects_paired"] = n_pairs
        if n_pairs == 0:
            rows.append(row)
            continue

        values_a = paired.iloc[:, 0].to_numpy(dtype=np.float64)
        values_b = paired.iloc[:, 1].to_numpy(dtype=np.float64)
        diff = values_a - values_b

        row["mean_group_a"] = float(np.mean(values_a))
        row["mean_group_b"] = float(np.mean(values_b))
        row["mean_diff_group_a_minus_b"] = float(np.mean(diff))
        row["median_diff_group_a_minus_b"] = float(np.median(diff))

        if n_pairs >= 2:
            try:
                t_result = ttest_rel(values_a, values_b, nan_policy="omit")
            except TypeError:
                t_result = ttest_rel(values_a, values_b)
            row["ttest_stat"] = float(t_result.statistic)
            row["ttest_p_two_sided"] = float(t_result.pvalue)

        if np.allclose(diff, 0.0):
            row["wilcoxon_stat"] = 0.0
            row["wilcoxon_p_two_sided"] = 1.0
        else:
            try:
                w_result = wilcoxon(values_a, values_b, alternative="two-sided")
                row["wilcoxon_stat"] = float(w_result.statistic)
                row["wilcoxon_p_two_sided"] = float(w_result.pvalue)
            except ValueError:
                pass

        rows.append(row)

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        return stats_df

    for prefix in ("ttest", "wilcoxon"):
        p_col = f"{prefix}_p_two_sided"
        p_values = stats_df[p_col].to_numpy(dtype=np.float64)
        corrected = np.full(p_values.shape, np.nan, dtype=np.float64)
        finite = np.isfinite(p_values)
        n_tests = int(np.count_nonzero(finite))
        if n_tests > 0:
            corrected[finite] = np.minimum(p_values[finite] * float(n_tests), 1.0)
        stats_df[f"{prefix}_p_bonferroni"] = corrected

        sig_col = f"{prefix}_significant_bonferroni_alpha_{float(alpha):g}"
        stats_df[sig_col] = pd.Series(pd.NA, index=stats_df.index, dtype="boolean")
        valid = np.isfinite(corrected)
        stats_df.loc[valid, sig_col] = corrected[valid] < float(alpha)

    return stats_df


def _std_over_rms(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan, np.nan

    std_value = float(np.std(finite_values))
    rms_value = float(np.sqrt(np.mean(finite_values**2)))
    if not np.isfinite(rms_value) or np.isclose(rms_value, 0.0):
        ratio_value = np.nan
    else:
        ratio_value = float(std_value / rms_value)
    return rms_value, std_value, ratio_value


def _coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan, np.nan

    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values))
    mean_abs = abs(mean_value)
    if not np.isfinite(mean_abs) or np.isclose(mean_abs, 0.0):
        cv_value = np.nan
    else:
        cv_value = float(std_value / mean_abs)
    return mean_value, std_value, cv_value


def _mean_absolute_deviation(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    mean_value = float(np.mean(finite_values))
    return float(np.mean(np.abs(finite_values - mean_value)))


def _safe_abs_ratio(numerator, denominator):
    numerator = float(numerator)
    denominator_abs = abs(float(denominator))
    if not np.isfinite(numerator) or not np.isfinite(denominator_abs) or np.isclose(denominator_abs, 0.0):
        return np.nan
    return float(numerator / denominator_abs)


def _std_centered_over_range(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    value_range = float(vmax - vmin)
    if not np.isfinite(value_range) or np.isclose(value_range, 0.0):
        return np.nan

    mean_value = float(np.mean(finite_values))
    centered_scaled = (finite_values - mean_value) / value_range
    return float(np.std(centered_scaled))


def _compute_pooled_metric_values(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "mean_projection": np.nan,
            "median_projection": np.nan,
            "std_projection": np.nan,
            "cv_projection": np.nan,
            "mad_mean_projection": np.nan,
            "mad_mean_over_median_projection": np.nan,
            "std_centered_range_projection": np.nan,
        }

    mean_value = float(np.mean(finite_values))
    median_value = float(np.median(finite_values))
    std_value = float(np.std(finite_values))
    mad_mean_value = _mean_absolute_deviation(finite_values)
    _, _, cv_value = _coefficient_of_variation(finite_values)
    mad_over_median_value = _safe_abs_ratio(mad_mean_value, median_value)
    std_centered_range = _std_centered_over_range(finite_values)

    return {
        "mean_projection": mean_value,
        "median_projection": median_value,
        "std_projection": std_value,
        "cv_projection": cv_value,
        "mad_mean_projection": mad_mean_value,
        "mad_mean_over_median_projection": mad_over_median_value,
        "std_centered_range_projection": std_centered_range,
    }


def compute_run_projection_std_over_rms(run_segments):
    rows = []
    for segment in run_segments:
        values = np.asarray(segment["values"], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        rms_value, std_value, ratio_value = _std_over_rms(finite_values)
        rows.append(
            {
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_source": int(segment["n_trials_source"]),
                "n_trials_kept": int(finite_values.size),
                "rms_projection": rms_value,
                "std_projection": std_value,
                "std_over_rms_projection": ratio_value,
            }
        )
    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    corrected = np.full(p_values.shape, np.nan, dtype=np.float64)
    finite_idx = np.flatnonzero(np.isfinite(p_values))
    if finite_idx.size == 0:
        return corrected

    finite_p = p_values[finite_idx]
    order = np.argsort(finite_p)
    ranked_p = finite_p[order]
    m = float(ranked_p.size)
    ranks = np.arange(1, ranked_p.size + 1, dtype=np.float64)
    adjusted_sorted = ranked_p * (m / ranks)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted_finite = np.full(ranked_p.shape, np.nan, dtype=np.float64)
    adjusted_finite[order] = adjusted_sorted
    corrected[finite_idx] = adjusted_finite
    return corrected


def _collect_subject_session_pooled_values(
    run_segments,
    session_a=1,
    session_b=2,
    runs=(1, 2),
):
    session_a = int(session_a)
    session_b = int(session_b)
    run_set = {int(run) for run in runs}

    pooled_lists = {}
    run_counts = {}

    for segment in run_segments:
        sub_tag = str(segment["sub_tag"])
        ses = int(segment["ses"])
        run = int(segment["run"])
        if ses not in {session_a, session_b} or run not in run_set:
            continue

        values = np.asarray(segment["values"], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        pooled_lists.setdefault((sub_tag, ses), []).append(finite_values)
        run_counts[(sub_tag, ses)] = run_counts.get((sub_tag, ses), 0) + 1

    subjects = sorted(
        {sub for sub, _ in run_counts.keys()},
        key=lambda sub: (_subject_order(sub), str(sub)),
    )

    pooled = {}
    for sub_tag in subjects:
        for ses in (session_a, session_b):
            chunks = pooled_lists.get((sub_tag, ses), [])
            pooled[(sub_tag, ses)] = (
                np.concatenate(chunks).astype(np.float64, copy=False)
                if chunks
                else np.array([], dtype=np.float64)
            )

    return pooled, run_counts, subjects, run_set


def compute_subject_session_pooled_metrics(
    run_segments,
    session_a=1,
    session_b=2,
    runs=(1, 2),
):
    pooled, run_counts, subjects, run_set = _collect_subject_session_pooled_values(
        run_segments,
        session_a=session_a,
        session_b=session_b,
        runs=runs,
    )

    rows = []
    for sub_tag in subjects:
        values_a = pooled.get((sub_tag, int(session_a)), np.array([], dtype=np.float64))
        values_b = pooled.get((sub_tag, int(session_b)), np.array([], dtype=np.float64))
        metrics_a = _compute_pooled_metric_values(values_a)
        metrics_b = _compute_pooled_metric_values(values_b)

        row = {
            "sub_tag": str(sub_tag),
            "session_a": session_a,
            "session_b": session_b,
            "runs_pooled": ",".join(str(run) for run in sorted(run_set)),
            "n_runs_session_a": int(run_counts.get((sub_tag, session_a), 0)),
            "n_runs_session_b": int(run_counts.get((sub_tag, session_b), 0)),
            "n_trials_session_a": int(values_a.size),
            "n_trials_session_b": int(values_b.size),
            "session_a_mean_projection": metrics_a["mean_projection"],
            "session_b_mean_projection": metrics_b["mean_projection"],
            "session_a_median_projection": metrics_a["median_projection"],
            "session_b_median_projection": metrics_b["median_projection"],
            "session_a_std_projection": metrics_a["std_projection"],
            "session_b_std_projection": metrics_b["std_projection"],
        }

        for metric_spec in POOLED_METRIC_SPECS:
            metric_key = metric_spec["key"]
            value_a = float(metrics_a.get(metric_key, np.nan))
            value_b = float(metrics_b.get(metric_key, np.nan))
            diff_key = f"{metric_key}_diff_session_b_minus_a"
            higher_key = f"higher_{metric_key}_session"

            row[f"session_a_{metric_key}"] = value_a
            row[f"session_b_{metric_key}"] = value_b
            row[diff_key] = np.nan
            row[higher_key] = pd.NA

            if np.isfinite(value_a) and np.isfinite(value_b):
                diff_value = float(value_b - value_a)
                row[diff_key] = diff_value
                if np.isclose(diff_value, 0.0):
                    row[higher_key] = "tie"
                elif diff_value > 0:
                    row[higher_key] = f"session_{session_b}"
                else:
                    row[higher_key] = f"session_{session_a}"

        rows.append(row)

    return pd.DataFrame(rows)


def compute_subject_session_pooled_cv(
    run_segments,
    session_a=1,
    session_b=2,
    runs=(1, 2),
):
    return compute_subject_session_pooled_metrics(
        run_segments,
        session_a=session_a,
        session_b=session_b,
        runs=runs,
    )


def build_group_level_session_metric_stats(
    subject_metric_df,
    session_a=1,
    session_b=2,
    metric_specs=POOLED_METRIC_SPECS,
):
    rows = []
    subject_count = 0 if (subject_metric_df is None or subject_metric_df.empty) else int(len(subject_metric_df))
    runs_label = "1,2"

    for metric_spec in metric_specs:
        metric_key = str(metric_spec["key"])
        row = {
            "metric": metric_key,
            "metric_label": str(metric_spec["label"]),
            "metric_formula": str(metric_spec["formula"]),
            "session_a": int(session_a),
            "session_b": int(session_b),
            "runs_pooled": runs_label,
            "n_subjects_total": subject_count,
            "n_subjects_paired_finite_metric": 0,
            "mean_session_a_metric": np.nan,
            "mean_session_b_metric": np.nan,
            "median_session_a_metric": np.nan,
            "median_session_b_metric": np.nan,
            "ks_statistic": np.nan,
            "ks_p_two_sided": np.nan,
            "iqr_session_a_metric": np.nan,
            "iqr_session_b_metric": np.nan,
            "iqr_diff_session_b_minus_a": np.nan,
            "mad_session_a_metric": np.nan,
            "mad_session_b_metric": np.nan,
            "mad_diff_session_b_minus_a": np.nan,
        }

        if subject_metric_df is None or subject_metric_df.empty:
            rows.append(row)
            continue

        df = subject_metric_df.copy()
        col_a = f"session_a_{metric_key}"
        col_b = f"session_b_{metric_key}"
        if col_a not in df.columns or col_b not in df.columns:
            rows.append(row)
            continue

        metric_a = pd.to_numeric(df[col_a], errors="coerce").to_numpy(dtype=np.float64)
        metric_b = pd.to_numeric(df[col_b], errors="coerce").to_numpy(dtype=np.float64)
        paired = np.isfinite(metric_a) & np.isfinite(metric_b)
        row["n_subjects_paired_finite_metric"] = int(np.count_nonzero(paired))
        if not np.any(paired):
            rows.append(row)
            continue

        values_a = metric_a[paired]
        values_b = metric_b[paired]
        row["mean_session_a_metric"] = float(np.mean(values_a))
        row["mean_session_b_metric"] = float(np.mean(values_b))
        row["median_session_a_metric"] = float(np.median(values_a))
        row["median_session_b_metric"] = float(np.median(values_b))

        ks_result = ks_2samp(values_a, values_b, alternative="two-sided", method="auto")
        row["ks_statistic"] = float(ks_result.statistic)
        row["ks_p_two_sided"] = float(ks_result.pvalue)

        iqr_a = _iqr(values_a)
        iqr_b = _iqr(values_b)
        mad_a = _mad(values_a)
        mad_b = _mad(values_b)
        row["iqr_session_a_metric"] = iqr_a
        row["iqr_session_b_metric"] = iqr_b
        row["mad_session_a_metric"] = mad_a
        row["mad_session_b_metric"] = mad_b
        if np.isfinite(iqr_a) and np.isfinite(iqr_b):
            row["iqr_diff_session_b_minus_a"] = float(iqr_b - iqr_a)
        if np.isfinite(mad_a) and np.isfinite(mad_b):
            row["mad_diff_session_b_minus_a"] = float(mad_b - mad_a)

        rows.append(row)

    return pd.DataFrame(rows)


def _iqr(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    q1, q3 = np.percentile(finite_values, [25.0, 75.0])
    return float(q3 - q1)


def _mad(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    median_value = float(np.median(finite_values))
    return float(np.median(np.abs(finite_values - median_value)))


def _ks_2samp_min_sign_flip(values_a, values_b):
    ks_direct = ks_2samp(values_a, values_b, alternative="two-sided", method="auto")
    ks_sign_flip = ks_2samp(values_a, -values_b, alternative="two-sided", method="auto")

    d_direct = float(ks_direct.statistic)
    d_sign_flip = float(ks_sign_flip.statistic)
    if d_sign_flip < d_direct:
        return ks_sign_flip
    if d_sign_flip > d_direct:
        return ks_direct

    # If D ties, prefer the larger p-value so the choice is deterministic.
    if float(ks_sign_flip.pvalue) > float(ks_direct.pvalue):
        return ks_sign_flip
    return ks_direct


def build_group_level_session_cv_stats(subject_cv_df, session_a=1, session_b=2):
    group_metric_df = build_group_level_session_metric_stats(
        subject_cv_df,
        session_a=session_a,
        session_b=session_b,
        metric_specs=[spec for spec in POOLED_METRIC_SPECS if spec["key"] == "cv_projection"],
    )
    if group_metric_df.empty:
        return pd.DataFrame(
            [
                {
                    "session_a": int(session_a),
                    "session_b": int(session_b),
                    "runs_pooled": "1,2",
                    "n_subjects_total": 0,
                    "n_subjects_paired_finite_cv": 0,
                    "mean_session_a_cv": np.nan,
                    "mean_session_b_cv": np.nan,
                    "median_session_a_cv": np.nan,
                    "median_session_b_cv": np.nan,
                    "ks_statistic": np.nan,
                    "ks_p_two_sided": np.nan,
                    "iqr_session_a_cv": np.nan,
                    "iqr_session_b_cv": np.nan,
                    "iqr_diff_session_b_minus_a": np.nan,
                    "mad_session_a_cv": np.nan,
                    "mad_session_b_cv": np.nan,
                    "mad_diff_session_b_minus_a": np.nan,
                }
            ]
        )

    src = group_metric_df.iloc[0].to_dict()
    return pd.DataFrame(
        [
            {
                "session_a": int(src.get("session_a", session_a)),
                "session_b": int(src.get("session_b", session_b)),
                "runs_pooled": str(src.get("runs_pooled", "1,2")),
                "n_subjects_total": int(src.get("n_subjects_total", 0)),
                "n_subjects_paired_finite_cv": int(src.get("n_subjects_paired_finite_metric", 0)),
                "mean_session_a_cv": float(src.get("mean_session_a_metric", np.nan)),
                "mean_session_b_cv": float(src.get("mean_session_b_metric", np.nan)),
                "median_session_a_cv": float(src.get("median_session_a_metric", np.nan)),
                "median_session_b_cv": float(src.get("median_session_b_metric", np.nan)),
                "ks_statistic": float(src.get("ks_statistic", np.nan)),
                "ks_p_two_sided": float(src.get("ks_p_two_sided", np.nan)),
                "iqr_session_a_cv": float(src.get("iqr_session_a_metric", np.nan)),
                "iqr_session_b_cv": float(src.get("iqr_session_b_metric", np.nan)),
                "iqr_diff_session_b_minus_a": float(src.get("iqr_diff_session_b_minus_a", np.nan)),
                "mad_session_a_cv": float(src.get("mad_session_a_metric", np.nan)),
                "mad_session_b_cv": float(src.get("mad_session_b_metric", np.nan)),
                "mad_diff_session_b_minus_a": float(src.get("mad_diff_session_b_minus_a", np.nan)),
            }
        ]
    )


def compute_subject_session_ks(
    run_segments,
    session_a=1,
    session_b=2,
    runs=(1, 2),
    alpha=KS_ALPHA,
):
    pooled, run_counts, subjects, run_set = _collect_subject_session_pooled_values(
        run_segments,
        session_a=session_a,
        session_b=session_b,
        runs=runs,
    )

    rows = []
    for sub_tag in subjects:
        values_a = pooled.get((sub_tag, int(session_a)), np.array([], dtype=np.float64))
        values_b = pooled.get((sub_tag, int(session_b)), np.array([], dtype=np.float64))

        row = {
            "sub_tag": str(sub_tag),
            "session_a": session_a,
            "session_b": session_b,
            "runs_pooled": ",".join(str(run) for run in sorted(run_set)),
            "n_runs_session_a": int(run_counts.get((sub_tag, session_a), 0)),
            "n_runs_session_b": int(run_counts.get((sub_tag, session_b), 0)),
            "n_trials_session_a": int(values_a.size),
            "n_trials_session_b": int(values_b.size),
            "session_a_mean_projection": float(np.mean(values_a)) if values_a.size else np.nan,
            "session_b_mean_projection": float(np.mean(values_b)) if values_b.size else np.nan,
            "session_a_median_projection": float(np.median(values_a)) if values_a.size else np.nan,
            "session_b_median_projection": float(np.median(values_b)) if values_b.size else np.nan,
            "session_a_variance_projection": float(np.var(values_a)) if values_a.size else np.nan,
            "session_b_variance_projection": float(np.var(values_b)) if values_b.size else np.nan,
            "session_a_std_projection": float(np.std(values_a)) if values_a.size else np.nan,
            "session_b_std_projection": float(np.std(values_b)) if values_b.size else np.nan,
            "variability_std_diff_session_b_minus_a": np.nan,
            "higher_variability_session": pd.NA,
            "session_a_iqr_projection": _iqr(values_a),
            "session_b_iqr_projection": _iqr(values_b),
            "variability_iqr_diff_session_b_minus_a": np.nan,
            "higher_iqr_session": pd.NA,
            "session_a_mad_projection": _mad(values_a),
            "session_b_mad_projection": _mad(values_b),
            "variability_mad_diff_session_b_minus_a": np.nan,
            "higher_mad_session": pd.NA,
            "ks_statistic": np.nan,
            "ks_p_two_sided": np.nan,
        }

        std_a = row["session_a_std_projection"]
        std_b = row["session_b_std_projection"]
        if np.isfinite(std_a) and np.isfinite(std_b):
            std_diff = float(std_b - std_a)
            row["variability_std_diff_session_b_minus_a"] = std_diff
            if np.isclose(std_diff, 0.0):
                row["higher_variability_session"] = "tie"
            elif std_diff > 0:
                row["higher_variability_session"] = f"session_{session_b}"
            else:
                row["higher_variability_session"] = f"session_{session_a}"

        iqr_a = row["session_a_iqr_projection"]
        iqr_b = row["session_b_iqr_projection"]
        if np.isfinite(iqr_a) and np.isfinite(iqr_b):
            iqr_diff = float(iqr_b - iqr_a)
            row["variability_iqr_diff_session_b_minus_a"] = iqr_diff
            if np.isclose(iqr_diff, 0.0):
                row["higher_iqr_session"] = "tie"
            elif iqr_diff > 0:
                row["higher_iqr_session"] = f"session_{session_b}"
            else:
                row["higher_iqr_session"] = f"session_{session_a}"

        mad_a = row["session_a_mad_projection"]
        mad_b = row["session_b_mad_projection"]
        if np.isfinite(mad_a) and np.isfinite(mad_b):
            mad_diff = float(mad_b - mad_a)
            row["variability_mad_diff_session_b_minus_a"] = mad_diff
            if np.isclose(mad_diff, 0.0):
                row["higher_mad_session"] = "tie"
            elif mad_diff > 0:
                row["higher_mad_session"] = f"session_{session_b}"
            else:
                row["higher_mad_session"] = f"session_{session_a}"

        if values_a.size > 0 and values_b.size > 0:
            ks_result = _ks_2samp_min_sign_flip(values_a, values_b)
            row["ks_statistic"] = float(ks_result.statistic)
            row["ks_p_two_sided"] = float(ks_result.pvalue)

        rows.append(row)

    ks_df = pd.DataFrame(rows)
    if ks_df.empty:
        return ks_df

    p_values = ks_df["ks_p_two_sided"].to_numpy(dtype=np.float64)
    ks_df["ks_p_fdr_bh"] = _benjamini_hochberg(p_values)
    sig_col = f"ks_significant_fdr_bh_alpha_{float(alpha):g}"
    ks_df[sig_col] = pd.Series(pd.NA, index=ks_df.index, dtype="boolean")
    finite = np.isfinite(ks_df["ks_p_fdr_bh"].to_numpy(dtype=np.float64))
    ks_df.loc[finite, sig_col] = (
        ks_df.loc[finite, "ks_p_fdr_bh"].to_numpy(dtype=np.float64) < float(alpha)
    )
    return ks_df


def build_group_level_ks_inference(ks_df, alpha=KS_ALPHA):
    row = {
        "session_a": 1,
        "session_b": 2,
        "runs_pooled": "1,2",
        "n_subjects_total": 0,
        "n_subjects_tested": 0,
        "mean_ks_statistic": np.nan,
        "median_ks_statistic": np.nan,
        "fisher_chi2_stat": np.nan,
        "fisher_p_two_sided": np.nan,
        f"n_subjects_uncorrected_p_lt_{float(alpha):g}": 0,
        f"fraction_subjects_uncorrected_p_lt_{float(alpha):g}": np.nan,
        f"n_subjects_fdr_bh_lt_{float(alpha):g}": 0,
    }

    if ks_df is None or ks_df.empty:
        return pd.DataFrame([row])

    df = ks_df.copy()
    row["n_subjects_total"] = int(len(df))
    p_values = pd.to_numeric(df["ks_p_two_sided"], errors="coerce").to_numpy(dtype=np.float64)
    d_values = pd.to_numeric(df["ks_statistic"], errors="coerce").to_numpy(dtype=np.float64)
    finite_p = np.isfinite(p_values)
    finite_d = np.isfinite(d_values)

    n_tested = int(np.count_nonzero(finite_p))
    row["n_subjects_tested"] = n_tested
    if np.any(finite_d):
        row["mean_ks_statistic"] = float(np.mean(d_values[finite_d]))
        row["median_ks_statistic"] = float(np.median(d_values[finite_d]))

    if n_tested > 0:
        fisher_stat, fisher_p = combine_pvalues(p_values[finite_p], method="fisher")
        row["fisher_chi2_stat"] = float(fisher_stat)
        row["fisher_p_two_sided"] = float(fisher_p)

        n_uncorrected = int(np.count_nonzero(p_values[finite_p] < float(alpha)))
        row[f"n_subjects_uncorrected_p_lt_{float(alpha):g}"] = n_uncorrected
        row[f"fraction_subjects_uncorrected_p_lt_{float(alpha):g}"] = float(
            n_uncorrected / n_tested
        )

    fdr_col = "ks_p_fdr_bh"
    if fdr_col in df.columns:
        fdr_values = pd.to_numeric(df[fdr_col], errors="coerce").to_numpy(dtype=np.float64)
        finite_fdr = np.isfinite(fdr_values)
        row[f"n_subjects_fdr_bh_lt_{float(alpha):g}"] = int(
            np.count_nonzero(fdr_values[finite_fdr] < float(alpha))
        )

    return pd.DataFrame([row])


def _subject_order(sub_tag):
    match = re.search(r"(\d+)$", str(sub_tag))
    if match:
        return int(match.group(1))
    return int(1e9)


def _density_grid(values, grid_points=512, pad_fraction=0.1, fallback_pad=0.25):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No finite values available for density-grid construction.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = vmax - vmin
    pad = (pad_fraction * vrange) if vrange > 0 else fallback_pad
    return np.linspace(vmin - pad, vmax + pad, int(grid_points))


def _evaluate_density(values, x):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No finite values available for density estimation.")

    if values.size < 2 or np.allclose(values, values[0]):
        width = max(np.std(values), max(abs(values[0]) * 0.05, 1e-6))
        density = (
            np.exp(-0.5 * ((x - values[0]) / width) ** 2)
            / (width * np.sqrt(2.0 * np.pi))
        )
    else:
        density = gaussian_kde(values).evaluate(x)

    area = np.trapezoid(density, x)
    if area > 0:
        density = density / area
    return density


def _format_p_for_label(p_value):
    if not np.isfinite(p_value):
        return "p=NA"
    if p_value < 1e-4:
        return f"p={p_value:.1e}"
    return f"p={p_value:.4f}"


def _plot_subject_session_ks_summary(ks_df, group_ks_df, out_path, alpha=KS_ALPHA):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11.2), sharex=True)

    if ks_df is None or ks_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No KS data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    ax = axes[0]
    ax_iqr = axes[1]
    ax_mad = axes[2]

    df = ks_df.copy()
    df["sub_tag"] = df["sub_tag"].astype(str)
    df["_order"] = df["sub_tag"].map(_subject_order)
    df = df[df["_order"] != 17].copy()
    df = df.sort_values(["_order", "sub_tag"]).reset_index(drop=True)

    labels = df["sub_tag"].tolist()
    x = np.arange(len(df), dtype=np.float64)
    d_values = pd.to_numeric(df["ks_statistic"], errors="coerce").to_numpy(dtype=np.float64)
    p_values = pd.to_numeric(df["ks_p_two_sided"], errors="coerce").to_numpy(dtype=np.float64)
    finite_d = np.isfinite(d_values)

    sig_col = f"ks_significant_fdr_bh_alpha_{float(alpha):g}"
    if sig_col in df.columns:
        significant_mask = df[sig_col].fillna(False).to_numpy(dtype=bool)
    else:
        significant_mask = np.zeros(len(df), dtype=bool)

    if np.any(finite_d):
        bars = ax.bar(
            x[finite_d],
            d_values[finite_d],
            width=0.8,
            color="tab:orange",
            alpha=0.85,
            edgecolor="black",
            linewidth=0.3,
        )
        d_max = float(np.max(d_values[finite_d]))
        y_offset = max(0.01, 0.03 * d_max)
        ax.set_ylim(0.0, d_max + (5.0 * y_offset))

        finite_indices = np.flatnonzero(finite_d)
        for bar, idx in zip(bars, finite_indices):
            y_top = float(bar.get_height())
            label_text = _format_p_for_label(p_values[idx])
            ax.text(
                float(bar.get_x() + bar.get_width() / 2.0),
                y_top + y_offset,
                label_text,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=0,
            )

        sig_d = finite_d & significant_mask
        if np.any(sig_d):
            ax.scatter(
                x[sig_d],
                d_values[sig_d],
                marker="*",
                color="black",
                s=90,
                label=f"FDR BH p < {float(alpha):g}",
                zorder=3,
            )
            ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "No finite KS statistics", ha="center", va="center", transform=ax.transAxes)

    ax.set_ylabel("KS statistic (D)")
    ax.grid(axis="y", alpha=0.2)

    def _plot_variability_diff_row(ax_row, diff_values, ylabel):
        finite_mask = np.isfinite(diff_values)
        if np.any(finite_mask):
            colors = np.where(diff_values[finite_mask] >= 0.0, "tab:red", "tab:blue")
            bars_var = ax_row.bar(
                x[finite_mask],
                diff_values[finite_mask],
                width=0.8,
                color=colors,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.3,
            )
            ax_row.axhline(0.0, color="black", linewidth=1.0)
            finite_indices = np.flatnonzero(finite_mask)
            max_abs = float(np.nanmax(np.abs(diff_values[finite_mask])))
            y_pad = max(1e-8, 0.04 * max_abs)
            for bar, idx in zip(bars_var, finite_indices):
                diff_val = float(diff_values[idx])
                if np.isclose(diff_val, 0.0):
                    winner = "tie"
                elif diff_val > 0:
                    winner = "s2"
                else:
                    winner = "s1"
                y_text = diff_val + (y_pad if diff_val >= 0 else -y_pad)
                va = "bottom" if diff_val >= 0 else "top"
                ax_row.text(
                    float(bar.get_x() + bar.get_width() / 2.0),
                    y_text,
                    winner,
                    ha="center",
                    va=va,
                    fontsize=6,
                )
            ax_row.text(
                0.01,
                0.98,
                "red: session 2 higher, blue: session 1 higher",
                transform=ax_row.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
        else:
            ax_row.text(
                0.5,
                0.5,
                "No finite variability data",
                ha="center",
                va="center",
                transform=ax_row.transAxes,
            )

        ax_row.set_ylabel(ylabel)
        ax_row.grid(axis="y", alpha=0.2)

    iqr_a = pd.to_numeric(df["session_a_iqr_projection"], errors="coerce").to_numpy(dtype=np.float64)
    iqr_b = pd.to_numeric(df["session_b_iqr_projection"], errors="coerce").to_numpy(dtype=np.float64)
    iqr_diff = iqr_b - iqr_a
    _plot_variability_diff_row(ax_iqr, iqr_diff, "IQR diff (session 2 - session 1)")

    mad_a = pd.to_numeric(df["session_a_mad_projection"], errors="coerce").to_numpy(dtype=np.float64)
    mad_b = pd.to_numeric(df["session_b_mad_projection"], errors="coerce").to_numpy(dtype=np.float64)
    mad_diff = mad_b - mad_a
    _plot_variability_diff_row(ax_mad, mad_diff, "MAD diff (session 2 - session 1)")

    ax_mad.set_xlabel("Subject")

    ax.set_xticks(x)
    ax_iqr.set_xticks(x)
    ax_mad.set_xticks(x)
    ax_mad.set_xticklabels(labels, rotation=45, ha="right")

    fig.suptitle(
        "KS test and variability direction per subject: session 1 vs session 2 (runs 1+2 pooled)"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_subject_session_projection_distribution_grid(
    run_segments,
    out_path,
    session_a=1,
    session_b=2,
    runs=(1, 2),
    n_rows=2,
    n_cols=7,
):
    session_a = int(session_a)
    session_b = int(session_b)
    n_rows = int(n_rows)
    n_cols = int(n_cols)
    max_panels = int(n_rows * n_cols)

    pooled, run_counts, subjects, _ = _collect_subject_session_pooled_values(
        run_segments,
        session_a=session_a,
        session_b=session_b,
        runs=runs,
    )
    paired_subjects = [
        sub_tag
        for sub_tag in subjects
        if pooled.get((sub_tag, session_a), np.array([], dtype=np.float64)).size > 0
        and pooled.get((sub_tag, session_b), np.array([], dtype=np.float64)).size > 0
    ]

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, 3.0 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).ravel()

    if len(paired_subjects) == 0:
        for ax in axes:
            ax.set_axis_off()
        axes[0].text(
            0.5,
            0.5,
            "No paired session data for KS plotting",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    all_values = []
    for sub_tag in paired_subjects:
        values_a = pooled.get((sub_tag, session_a), np.array([], dtype=np.float64))
        values_b = pooled.get((sub_tag, session_b), np.array([], dtype=np.float64))
        all_values.append(values_a)
        all_values.append(values_b)

    shared_x = None
    shared_xlim = None
    if all_values:
        combined_values = np.concatenate(all_values).astype(np.float64, copy=False)
        shared_x = _density_grid(combined_values, grid_points=512, fallback_pad=1e-6)
        shared_xlim = (float(shared_x[0]), float(shared_x[-1]))

    n_to_plot = min(len(paired_subjects), max_panels)
    for panel_idx, ax in enumerate(axes):
        if panel_idx >= n_to_plot:
            ax.set_axis_off()
            continue

        sub_tag = paired_subjects[panel_idx]
        values_a = pooled.get((sub_tag, session_a), np.array([], dtype=np.float64))
        values_b = pooled.get((sub_tag, session_b), np.array([], dtype=np.float64))

        title = str(sub_tag)
        x = shared_x
        if x is None:
            x = _density_grid(
                np.concatenate([values_a, values_b]).astype(np.float64, copy=False),
                grid_points=512,
                fallback_pad=1e-6,
            )

        density_a = _evaluate_density(values_a, x)
        ax.plot(x, density_a, color="tab:blue", linewidth=1.5, label=f"Session {session_a}")
        ax.fill_between(x, density_a, color="tab:blue", alpha=0.16)
        ax.axvline(float(np.median(values_a)), color="tab:blue", linestyle=":", linewidth=0.9)

        density_b = _evaluate_density(values_b, x)
        ax.plot(
            x,
            density_b,
            color="tab:red",
            linewidth=1.5,
            linestyle="--",
            label=f"Session {session_b}",
        )
        ax.fill_between(x, density_b, color="tab:red", alpha=0.12)
        ax.axvline(float(np.median(values_b)), color="tab:red", linestyle=":", linewidth=0.9)

        ks_result = ks_2samp(values_a, values_b, alternative="two-sided", method="auto")
        ks_text = f"D={float(ks_result.statistic):.3g}\np={float(ks_result.pvalue):.3g}"

        info_text = ks_text
        ax.text(
            0.98,
            0.98,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.7,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.75"},
        )

        if shared_xlim is not None:
            ax.set_xlim(shared_xlim)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.2)

        if panel_idx % n_cols == 0:
            ax.set_ylabel("Density")
        if panel_idx // n_cols == (n_rows - 1):
            ax.set_xlabel("Projected signal")

    n_subjects = len(paired_subjects)
    if n_subjects > max_panels:
        suffix = f" (first {max_panels} of {n_subjects} subjects)"
    else:
        suffix = ""
    legend_handles = [
        Line2D([0], [0], color="tab:blue", linewidth=2.0, linestyle="-", label=f"Session {session_a}"),
        Line2D([0], [0], color="tab:red", linewidth=2.0, linestyle="--", label=f"Session {session_b}"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),
        frameon=True,
        framealpha=0.9,
        edgecolor="0.75",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_session_metric_distribution_summary(
    subject_metric_df,
    group_metric_stats_df,
    metric_spec,
    out_path,
):
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.0))
    metric_key = str(metric_spec["key"])
    metric_label = str(metric_spec["label"])
    col_a = f"session_a_{metric_key}"
    col_b = f"session_b_{metric_key}"

    if subject_metric_df is None or subject_metric_df.empty:
        ax.text(
            0.5,
            0.5,
            f"No pooled {metric_label} data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    df = subject_metric_df.copy()
    if col_a not in df.columns or col_b not in df.columns:
        ax.text(
            0.5,
            0.5,
            f"Missing columns for {metric_label}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    value_a = pd.to_numeric(df[col_a], errors="coerce").to_numpy(dtype=np.float64)
    value_b = pd.to_numeric(df[col_b], errors="coerce").to_numpy(dtype=np.float64)
    paired = np.isfinite(value_a) & np.isfinite(value_b)

    if not np.any(paired):
        ax.text(
            0.5,
            0.5,
            f"No finite pooled {metric_label} pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    values_a = value_a[paired]
    values_b = value_b[paired]
    all_values = np.concatenate([values_a, values_b])
    x = _density_grid(all_values, grid_points=512, fallback_pad=1e-6)
    density_a = _evaluate_density(values_a, x)
    density_b = _evaluate_density(values_b, x)

    ax.plot(
        x,
        density_a,
        color="tab:blue",
        linewidth=2.0,
        label=f"Session 1 pooled {metric_label}",
    )
    ax.fill_between(x, density_a, color="tab:blue", alpha=0.2)
    ax.plot(
        x,
        density_b,
        color="tab:red",
        linewidth=2.0,
        linestyle="--",
        label=f"Session 2 pooled {metric_label}",
    )
    ax.fill_between(x, density_b, color="tab:red", alpha=0.15)

    median_a = float(np.median(values_a))
    median_b = float(np.median(values_b))
    ax.axvline(median_a, color="tab:blue", linestyle=":", linewidth=1.2)
    ax.axvline(median_b, color="tab:red", linestyle=":", linewidth=1.2)

    summary = {}
    if group_metric_stats_df is not None and not group_metric_stats_df.empty:
        row_match = group_metric_stats_df[group_metric_stats_df["metric"] == metric_key]
        if not row_match.empty:
            summary = row_match.iloc[0].to_dict()

    ks_d = float(summary.get("ks_statistic", np.nan))
    ks_p = float(summary.get("ks_p_two_sided", np.nan))
    iqr_a = float(summary.get("iqr_session_a_metric", np.nan))
    iqr_b = float(summary.get("iqr_session_b_metric", np.nan))
    iqr_diff = float(summary.get("iqr_diff_session_b_minus_a", np.nan))
    mad_a = float(summary.get("mad_session_a_metric", np.nan))
    mad_b = float(summary.get("mad_session_b_metric", np.nan))
    mad_diff = float(summary.get("mad_diff_session_b_minus_a", np.nan))

    stats_text = (
        f"n paired subjects = {int(np.count_nonzero(paired))}\n"
        f"KS: D={ks_d:.3g}, p={ks_p:.3g}\n"
        f"IQR: s1={iqr_a:.3g}, s2={iqr_b:.3g}, s2-s1={iqr_diff:.3g}\n"
        f"MAD: s1={mad_a:.3g}, s2={mad_b:.3g}, s2-s1={mad_diff:.3g}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    ax.set_xlabel(str(metric_spec["x_label"]))
    ax.set_ylabel("Probability density")
    ax.set_title(str(metric_spec["title"]))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_session_cv_distribution_summary(subject_cv_df, group_cv_stats_df, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 6.2))

    if subject_cv_df is None or subject_cv_df.empty:
        ax.text(
            0.5,
            0.5,
            "No pooled CV data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    df = subject_cv_df.copy()
    if "session_a_cv_projection" not in df.columns or "session_b_cv_projection" not in df.columns:
        ax.text(
            0.5,
            0.5,
            "Missing pooled CV columns",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    if "sub_tag" in df.columns:
        df["sub_tag"] = df["sub_tag"].astype(str)
        df["_order"] = df["sub_tag"].map(_subject_order)
        df = df.sort_values(["_order", "sub_tag"])

    value_a = pd.to_numeric(df["session_a_cv_projection"], errors="coerce").to_numpy(dtype=np.float64)
    value_b = pd.to_numeric(df["session_b_cv_projection"], errors="coerce").to_numpy(dtype=np.float64)
    paired = np.isfinite(value_a) & np.isfinite(value_b)

    if not np.any(paired):
        ax.text(
            0.5,
            0.5,
            "No finite pooled CV pairs",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    values_a = value_a[paired]
    values_b = value_b[paired]
    rng = np.random.default_rng(0)
    jitter_a = rng.uniform(-0.035, 0.035, size=values_a.size)
    jitter_b = rng.uniform(-0.035, 0.035, size=values_b.size)

    box = ax.boxplot(
        [values_a, values_b],
        positions=[1.0, 2.0],
        widths=0.45,
        patch_artist=True,
        boxprops={"linewidth": 1.1, "edgecolor": "black"},
        medianprops={"linewidth": 1.5, "color": "black"},
        whiskerprops={"linewidth": 1.0, "color": "0.3"},
        capprops={"linewidth": 1.0, "color": "0.3"},
    )
    box["boxes"][0].set_facecolor("tab:blue")
    box["boxes"][0].set_alpha(0.25)
    box["boxes"][1].set_facecolor("tab:red")
    box["boxes"][1].set_alpha(0.25)

    for cv_a, cv_b in zip(values_a, values_b):
        ax.plot([1.0, 2.0], [cv_a, cv_b], color="0.65", linewidth=0.8, alpha=0.75, zorder=1)

    ax.scatter(
        np.full(values_a.shape, 1.0) + jitter_a,
        values_a,
        color="tab:blue",
        s=22,
        alpha=0.88,
        zorder=2,
        label="Session 1 subjects",
    )
    ax.scatter(
        np.full(values_b.shape, 2.0) + jitter_b,
        values_b,
        color="tab:red",
        s=22,
        alpha=0.88,
        zorder=2,
        label="Session 2 subjects",
    )

    summary = {}
    if group_cv_stats_df is not None and not group_cv_stats_df.empty:
        summary = group_cv_stats_df.iloc[0].to_dict()

    ks_d = float(summary.get("ks_statistic", np.nan))
    ks_p = float(summary.get("ks_p_two_sided", np.nan))
    iqr_a = float(summary.get("iqr_session_a_cv", np.nan))
    iqr_b = float(summary.get("iqr_session_b_cv", np.nan))
    iqr_diff = float(summary.get("iqr_diff_session_b_minus_a", np.nan))
    mad_a = float(summary.get("mad_session_a_cv", np.nan))
    mad_b = float(summary.get("mad_session_b_cv", np.nan))
    mad_diff = float(summary.get("mad_diff_session_b_minus_a", np.nan))

    stats_text = (
        f"n paired subjects = {int(np.count_nonzero(paired))}\n"
        f"KS: D={ks_d:.3g}, p={ks_p:.3g}\n"
        f"IQR: s1={iqr_a:.3g}, s2={iqr_b:.3g}, s2-s1={iqr_diff:.3g}\n"
        f"MAD: s1={mad_a:.3g}, s2={mad_b:.3g}, s2-s1={mad_diff:.3g}"
    )
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    ax.set_xticks([1.0, 2.0])
    ax.set_xticklabels(["Session 1", "Session 2"])
    ax.set_ylabel("Pooled-trial CV projection (runs 1+2)")
    ax.set_title("Session 1 vs Session 2 pooled-trial CV by subject")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_subject_consecutive_trial_scatter(
    run_segments,
    out_dir,
    file_stem,
    session_a=1,
    session_b=2,
    runs=(1, 2),
):
    pooled, _, subjects, _ = _collect_subject_session_pooled_values(
        run_segments,
        session_a=session_a,
        session_b=session_b,
        runs=runs,
    )
    paired_subjects = [
        sub_tag
        for sub_tag in subjects
        if pooled.get((sub_tag, int(session_a)), np.array([], dtype=np.float64)).size >= 2
        and pooled.get((sub_tag, int(session_b)), np.array([], dtype=np.float64)).size >= 2
    ]
    if len(paired_subjects) == 0:
        return [], []

    selected_subjects = [str(sub_tag) for sub_tag in paired_subjects]

    os.makedirs(out_dir, exist_ok=True)
    output_paths = []
    for sub_tag in selected_subjects:
        values_a = pooled.get((sub_tag, int(session_a)), np.array([], dtype=np.float64))
        values_b = pooled.get((sub_tag, int(session_b)), np.array([], dtype=np.float64))

        x_a = values_a[:-1]
        y_a = values_a[1:]
        x_b = values_b[:-1]
        y_b = values_b[1:]

        combined = np.concatenate([values_a, values_b]).astype(np.float64, copy=False)
        finite_combined = combined[np.isfinite(combined)]
        if finite_combined.size > 0:
            vmin = float(np.min(finite_combined))
            vmax = float(np.max(finite_combined))
            vrange = vmax - vmin
            pad = (0.08 * vrange) if vrange > 0 else max(1e-6, 0.08 * max(abs(vmin), 1.0))
            lims = (vmin - pad, vmax + pad)
        else:
            lims = (-1.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
        session_specs = [
            (axes[0], x_a, y_a, int(session_a), "tab:blue"),
            (axes[1], x_b, y_b, int(session_b), "tab:red"),
        ]

        for ax, x_vals, y_vals, ses, color in session_specs:
            pair_idx = np.arange(x_vals.size, dtype=np.float64)
            if x_vals.size > 0:
                ax.scatter(
                    x_vals,
                    y_vals,
                    c=pair_idx,
                    cmap="viridis",
                    s=20,
                    alpha=0.9,
                    edgecolors="none",
                )
            ax.plot(lims, lims, linestyle="--", color=color, linewidth=1.0, alpha=0.85)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(f"Session {ses} (n pairs={x_vals.size})")
            ax.set_xlabel("trial(i)")
            ax.set_ylabel("tria(i+1)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 2), useMathText=True)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(alpha=0.2)

        fig.suptitle(f"{sub_tag}: Consecutive-trial Y scatter (runs 1+2 pooled)")
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        safe_sub = re.sub(r"[^A-Za-z0-9._-]+", "_", str(sub_tag))
        out_path = os.path.join(
            out_dir,
            f"{file_stem}_sub_{safe_sub}_session12_consecutive_trial_scatter.png",
        )
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        output_paths.append(out_path)

    return selected_subjects, output_paths


def _plot_metric_by_ses_run(df, value_col, x_label, title, out_path, color):
    combos = [(1, 1), (1, 2), (2, 1), (2, 2)]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)

    finite_all = df[value_col].to_numpy(dtype=np.float64)
    finite_all = finite_all[np.isfinite(finite_all)]
    use_sci = bool(finite_all.size and np.nanmax(np.abs(finite_all)) < 1e-3)
    if finite_all.size:
        shared_x = _density_grid(finite_all, grid_points=512, fallback_pad=1e-6)
        shared_xlim = (float(shared_x[0]), float(shared_x[-1]))
    else:
        shared_x = None
        shared_xlim = None

    for ax, (ses, run) in zip(axes.ravel(), combos):
        subset = df[(df["ses"] == ses) & (df["run"] == run)].copy()
        subset["sub_tag"] = subset["sub_tag"].astype(str)
        subset["_order"] = subset["sub_tag"].map(_subject_order)
        subset = subset.sort_values(["_order", "sub_tag"])
        subset = subset[np.isfinite(subset[value_col].to_numpy(dtype=np.float64))]

        if subset.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Session {ses} - Run {run}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Probability density")
            if shared_xlim is not None:
                ax.set_xlim(shared_xlim)
            continue

        y = subset[value_col].to_numpy(dtype=np.float64)
        x = shared_x if shared_x is not None else _density_grid(y, grid_points=512, fallback_pad=1e-6)
        density = _evaluate_density(y, x)

        ax.plot(x, density, color=color, linewidth=2.0)
        ax.fill_between(x, density, color=color, alpha=0.2)
        ax.set_title(f"Session {ses} - Run {run} (n={len(subset)})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Probability density")
        if shared_xlim is not None:
            ax.set_xlim(shared_xlim)
        if use_sci:
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

        mean_value = float(np.mean(y))
        median_value = float(np.median(y))
        ax.axvline(mean_value, color="black", linestyle="--", linewidth=1.0)
        ax.text(
            0.98,
            0.98,
            f"mean={mean_value:.3g}\nmedian={median_value:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
        )

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _read_subject_list_file(path):
    subjects = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = [token.strip() for token in re.split(r"[,\s]+", line) if token.strip()]
            subjects.extend(tokens)
    return sorted(set(subjects), key=lambda sub: (_subject_order(sub), str(sub)))


def _infer_projection_path_from_subject_list_path(subject_list_path):
    subject_list_path = os.path.abspath(os.path.expanduser(subject_list_path))
    base_name = os.path.basename(subject_list_path)
    stem = os.path.splitext(base_name)[0]
    directory = os.path.dirname(subject_list_path)

    candidates = []
    if "_excluded_subjects" in stem:
        prefix = stem.split("_excluded_subjects", 1)[0]
        candidates.append(os.path.join(directory, f"{prefix}.npy"))
    candidates.append(os.path.join(directory, f"{stem}.npy"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_projection_vector(path):
    path = os.path.abspath(os.path.expanduser(path))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.asarray(np.load(path)).ravel()

    # Numeric text support (one value per row or whitespace-delimited)
    try:
        return np.asarray(np.loadtxt(path, dtype=np.float64)).ravel()
    except Exception as exc:
        raise RuntimeError(f"Failed to load numeric projection from '{path}'.") from exc


def _resolve_projection_and_excluded_subjects(projection_path, excluded_subjects_path=None):
    projection_path = os.path.abspath(os.path.expanduser(projection_path))
    excluded_subjects = set()
    resolved_projection_path = projection_path
    exclusion_sources = []

    if excluded_subjects_path is not None:
        excluded_subjects_path = os.path.abspath(os.path.expanduser(excluded_subjects_path))
        explicit_subjects = _read_subject_list_file(excluded_subjects_path)
        excluded_subjects.update(explicit_subjects)
        exclusion_sources.append(excluded_subjects_path)

    ext = os.path.splitext(projection_path)[1].lower()
    if ext == ".txt":
        try:
            projection = _load_projection_vector(projection_path)
        except RuntimeError:
            inferred_projection_path = _infer_projection_path_from_subject_list_path(projection_path)
            if inferred_projection_path is None:
                raise RuntimeError(
                    "Projection path points to non-numeric text and no matching .npy file was found: "
                    f"{projection_path}"
                )
            projection = _load_projection_vector(inferred_projection_path)
            resolved_projection_path = inferred_projection_path
            inferred_subjects = _read_subject_list_file(projection_path)
            excluded_subjects.update(inferred_subjects)
            exclusion_sources.append(projection_path)
    else:
        projection = _load_projection_vector(projection_path)

    return projection, resolved_projection_path, sorted(excluded_subjects), exclusion_sources


def _exclude_subjects_from_segments(run_segments, excluded_subjects):
    excluded_set = {str(sub) for sub in (excluded_subjects or [])}
    if not excluded_set:
        return list(run_segments), 0, []

    filtered_segments = []
    removed_subjects = set()
    for segment in run_segments:
        sub_tag = str(segment["sub_tag"])
        if sub_tag in excluded_set:
            removed_subjects.add(sub_tag)
            continue
        filtered_segments.append(segment)

    removed_count = int(len(run_segments) - len(filtered_segments))
    removed_subjects = sorted(removed_subjects, key=lambda sub: (_subject_order(sub), str(sub)))
    return filtered_segments, removed_count, removed_subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection-path",
        default=(
            "results/behave_vs_bold/"
            "projection_voxel_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_thr90.npy"
        ),
        help="Path to projection vector (.npy or numeric text). Can also be a subject-list .txt.",
    )
    parser.add_argument(
        "--excluded-subjects-path",
        default=None,
        help="Optional path to a text file listing subject tags to exclude.",
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help="Path to concat manifest TSV.",
    )
    parser.add_argument(
        "--trial-keep-root",
        default=DEFAULT_TRIAL_KEEP_ROOT,
        help="Root containing trial_keep_run*.npy files.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    args = parser.parse_args()

    projection_input_path = os.path.abspath(os.path.expanduser(args.projection_path))
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest_path))
    trial_keep_root = os.path.abspath(os.path.expanduser(args.trial_keep_root))
    excluded_subjects_path = (
        None
        if args.excluded_subjects_path in {None, ""}
        else os.path.abspath(os.path.expanduser(args.excluded_subjects_path))
    )

    projection, resolved_projection_path, excluded_subjects, exclusion_sources = (
        _resolve_projection_and_excluded_subjects(
            projection_input_path,
            excluded_subjects_path=excluded_subjects_path,
        )
    )
    manifest_df = pd.read_csv(manifest_path, sep="\t")

    run_segments_all, layout = split_projection_by_run(projection, manifest_df, trial_keep_root)
    run_segments, removed_segment_count, removed_subjects = _exclude_subjects_from_segments(
        run_segments_all,
        excluded_subjects=excluded_subjects,
    )
    if len(run_segments) == 0:
        raise RuntimeError("No run segments remaining after exclusions.")

    run_variance_df = compute_run_variances(run_segments)
    run_variance_df["variance_projection"] = (
        run_variance_df["variance_projection"].astype(np.float64) * float(VARIANCE_SCALE)
    )
    run_std_rms_df = compute_run_projection_std_over_rms(run_segments)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(projection_input_path) or os.getcwd()
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(projection_input_path))[0]
    variance_csv_path = os.path.join(out_dir, f"{stem}_sub_ses_run_projection_variability.csv")
    std_rms_csv_path = os.path.join(out_dir, f"{stem}_sub_ses_run_projection_std_over_rms.csv")
    stats_csv_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_variability_pairwise_stats.csv"
    )
    ks_subject_csv_path = os.path.join(out_dir, f"{stem}_sub_session12_ks_stats.csv")
    ks_group_csv_path = os.path.join(out_dir, f"{stem}_group_session12_ks_inference.csv")
    pooled_metrics_subject_csv_path = os.path.join(
        out_dir, f"{stem}_sub_session12_pooled_metrics_stats.csv"
    )
    pooled_metrics_group_csv_path = os.path.join(
        out_dir, f"{stem}_group_session12_pooled_metrics_stats.csv"
    )
    pooled_cv_subject_csv_path = os.path.join(out_dir, f"{stem}_sub_session12_pooled_cv_stats.csv")
    pooled_cv_group_csv_path = os.path.join(out_dir, f"{stem}_group_session12_pooled_cv_stats.csv")
    variance_plot_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_variability_subject_comp.png"
    )
    std_rms_plot_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_std_over_rms_subject_comp.png"
    )
    ks_plot_path = os.path.join(out_dir, f"{stem}_sub_session12_ks_summary.png")
    ks_subject_grid_plot_path = os.path.join(
        out_dir, f"{stem}_sub_session12_ks_subject_distribution_grid_2x7.png"
    )
    pooled_metric_plot_paths = {
        spec["key"]: os.path.join(out_dir, f"{stem}_sub_session12_{spec['file_stub']}_summary.png")
        for spec in POOLED_METRIC_SPECS
    }
    pooled_cv_plot_path = os.path.join(out_dir, f"{stem}_sub_session12_pooled_cv_boxplot_summary.png")
    pooled_metric_plot_paths["cv_projection"] = pooled_cv_plot_path
    consecutive_trial_scatter_dir = os.path.join(
        out_dir,
        f"{stem}_sub_session12_consecutive_trial_scatter_random_subjects",
    )

    pairwise_stats_df = build_projection_variability_stats_table(run_variance_df)
    subject_ks_df = compute_subject_session_ks(
        run_segments,
        session_a=1,
        session_b=2,
        runs=(1, 2),
        alpha=KS_ALPHA,
    )
    group_ks_df = build_group_level_ks_inference(subject_ks_df, alpha=KS_ALPHA)
    pooled_metrics_subject_df = compute_subject_session_pooled_metrics(
        run_segments,
        session_a=1,
        session_b=2,
        runs=(1, 2),
    )
    pooled_metrics_group_df = build_group_level_session_metric_stats(
        pooled_metrics_subject_df,
        session_a=1,
        session_b=2,
    )
    pooled_cv_subject_df = pooled_metrics_subject_df.copy()
    pooled_cv_group_df = build_group_level_session_cv_stats(
        pooled_metrics_subject_df,
        session_a=1,
        session_b=2,
    )

    run_variance_df.to_csv(variance_csv_path, index=False)
    run_std_rms_df.to_csv(std_rms_csv_path, index=False)
    pairwise_stats_df.to_csv(stats_csv_path, index=False)
    subject_ks_df.to_csv(ks_subject_csv_path, index=False)
    group_ks_df.to_csv(ks_group_csv_path, index=False)
    pooled_metrics_subject_df.to_csv(pooled_metrics_subject_csv_path, index=False)
    pooled_metrics_group_df.to_csv(pooled_metrics_group_csv_path, index=False)
    pooled_cv_subject_df.to_csv(pooled_cv_subject_csv_path, index=False)
    pooled_cv_group_df.to_csv(pooled_cv_group_csv_path, index=False)

    _plot_metric_by_ses_run(
        run_variance_df,
        value_col="variance_projection",
        x_label=f"Projection variability",
        title=f"Projection variability",
        out_path=variance_plot_path,
        color="tab:blue",
    )
    _plot_metric_by_ses_run(
        run_std_rms_df,
        value_col="std_over_rms_projection",
        x_label="Projection std/rms",
        title="Projection std/rms",
        out_path=std_rms_plot_path,
        color="tab:green",
    )
    _plot_subject_session_ks_summary(
        subject_ks_df,
        group_ks_df,
        out_path=ks_plot_path,
        alpha=KS_ALPHA,
    )
    _plot_subject_session_projection_distribution_grid(
        run_segments,
        out_path=ks_subject_grid_plot_path,
        session_a=1,
        session_b=2,
        runs=(1, 2),
        n_rows=2,
        n_cols=7,
    )
    for metric_spec in POOLED_METRIC_SPECS:
        metric_key = str(metric_spec["key"])
        if metric_key == "cv_projection":
            _plot_session_cv_distribution_summary(
                pooled_cv_subject_df,
                pooled_cv_group_df,
                out_path=pooled_metric_plot_paths[metric_key],
            )
            continue
        _plot_session_metric_distribution_summary(
            pooled_metrics_subject_df,
            pooled_metrics_group_df,
            metric_spec=metric_spec,
            out_path=pooled_metric_plot_paths[metric_key],
        )
    selected_subjects, consecutive_trial_scatter_paths = _plot_subject_consecutive_trial_scatter(
        run_segments,
        out_dir=consecutive_trial_scatter_dir,
        file_stem=stem,
        session_a=1,
        session_b=2,
        runs=(1, 2),
    )

    print(f"Projection length: {projection.size}")
    print(f"Projection source path: {resolved_projection_path}")
    print(f"Projection layout: {layout}")
    print(f"Input run segments: {len(run_segments_all)}")
    print(f"Run segments after exclusions: {len(run_segments)}")
    print(f"Excluded segments count: {removed_segment_count}")
    if exclusion_sources:
        print(f"Exclusion source files: {', '.join(exclusion_sources)}")
    if removed_subjects:
        print(f"Excluded subjects present in data: {', '.join(removed_subjects)}")
    print(f"Rows (variance): {len(run_variance_df)}")
    print(f"Rows (std/rms): {len(run_std_rms_df)}")
    print(f"Rows (subject KS): {len(subject_ks_df)}")
    print(f"Rows (subject pooled metrics): {len(pooled_metrics_subject_df)}")
    print(f"Saved variance CSV: {variance_csv_path}")
    print(f"Saved std/rms CSV:  {std_rms_csv_path}")
    print(f"Saved stats CSV:    {stats_csv_path}")
    print(f"Saved subject KS CSV: {ks_subject_csv_path}")
    print(f"Saved group KS CSV:   {ks_group_csv_path}")
    print(f"Saved subject pooled metrics CSV: {pooled_metrics_subject_csv_path}")
    print(f"Saved group pooled metrics CSV:   {pooled_metrics_group_csv_path}")
    print(f"Saved subject pooled CV CSV: {pooled_cv_subject_csv_path}")
    print(f"Saved group pooled CV CSV:   {pooled_cv_group_csv_path}")
    print(f"Saved variability figure: {variance_plot_path}")
    print(f"Saved std/rms figure:    {std_rms_plot_path}")
    print(f"Saved KS figure:         {ks_plot_path}")
    print(f"Saved KS 2x7 grid figure:{ks_subject_grid_plot_path}")
    print(f"Saved pooled CV box-plot figure:  {pooled_cv_plot_path}")
    print(
        "Saved consecutive-trial scatter directory: "
        f"{consecutive_trial_scatter_dir}"
    )
    if selected_subjects:
        print(
            "Subjects with paired session data for consecutive-trial scatter: "
            f"{', '.join(selected_subjects)}"
        )
    for scatter_path in consecutive_trial_scatter_paths:
        print(f"Saved consecutive-trial scatter figure: {scatter_path}")
    for metric_spec in POOLED_METRIC_SPECS:
        metric_key = metric_spec["key"]
        if metric_key == "cv_projection":
            continue
        print(
            f"Saved pooled {metric_spec['label']} figure: {pooled_metric_plot_paths[metric_key]}"
        )
    print("\nProjection variability pairwise statistics:")
    print(pairwise_stats_df.to_string(index=False))
    print("\nSubject-level KS statistics (session 1 vs session 2, runs 1+2 pooled):")
    print(subject_ks_df.to_string(index=False))
    print("\nGroup-level KS inference:")
    print(group_ks_df.to_string(index=False))
    print("\nSubject-level pooled metric statistics (session 1 vs session 2, runs 1+2 pooled):")
    print(pooled_metrics_subject_df.to_string(index=False))
    print("\nGroup-level pooled metric session comparison:")
    print(pooled_metrics_group_df.to_string(index=False))


if __name__ == "__main__":
    main()
