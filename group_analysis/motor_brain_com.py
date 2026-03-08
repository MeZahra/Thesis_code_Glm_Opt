import argparse
import os
import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ttest_rel, wilcoxon
import statsmodels.formula.api as smf

DEFAULT_MANIFEST_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv"
DEFAULT_TRIAL_KEEP_ROOT = "/Data/zahra/results_glm"
DEFAULT_BEHAVIOR_ROOT = "/Data/zahra/behaviour"

METRIC_SPECS = [
    {
        "key": "variance",
        "label": "variance",
        "projection_col": "variance_projection",
        "behavior_col": "variance_behavior_col2",
        "projection_plot_scale": 1e7,
        "file_stub": "variance",
    },
    {
        "key": "cv",
        "label": "CV",
        "projection_col": "cv_projection",
        "behavior_col": "cv_behavior_col2",
        "projection_plot_scale": 1.0,
        "file_stub": "cv",
    },
    {
        "key": "qcd",
        "label": "QCD",
        "projection_col": "qcd_projection",
        "behavior_col": "qcd_behavior_col2",
        "projection_plot_scale": 1.0,
        "file_stub": "qcd",
    },
    {
        "key": "adjacent_diff_ratio_sum",
        "label": "sum((x_i-x_(i+1))^2/(x_i^2+x_(i+1)^2))",
        "projection_col": "adjacent_diff_ratio_sum_projection",
        "behavior_col": "adjacent_diff_ratio_sum_behavior_col2",
        "projection_plot_scale": 1.0,
        "file_stub": "adjacent_diff_ratio_sum",
    },
    {
        "key": "mad_mean",
        "label": "MAD(mean)",
        "projection_col": "mad_mean_projection",
        "behavior_col": "mad_mean_behavior_col2",
        "projection_plot_scale": 1.0,
        "file_stub": "mad_mean",
    },
    {
        "key": "mad_mean_over_median",
        "label": "MAD(mean)/|median|",
        "projection_col": "mad_mean_over_median_projection",
        "behavior_col": "mad_mean_over_median_behavior_col2",
        "projection_plot_scale": 1.0,
        "file_stub": "mad_mean_over_median",
    },
    {
        "key": "std_centered_range",
        "label": "std((X-mean)/(max-min))",
        "projection_col": "std_centered_range_projection",
        "behavior_col": "std_centered_range_behavior_col2",
        "projection_plot_scale": 1.0,
        "file_stub": "std_centered_range",
    },
]

RUN_OUTLIER_METRIC_KEYS = [
    "cv",
    "qcd",
    "adjacent_diff_ratio_sum",
    "mad_mean",
    "mad_mean_over_median",
    "std_centered_range",
]


def _resolve_trial_keep_path(row, trial_keep_root):
    from_manifest = str(getattr(row, "trial_keep_path", "") or "").strip()
    if from_manifest and os.path.exists(from_manifest):
        return from_manifest
    candidate = os.path.join(trial_keep_root, str(row.sub_tag), f"ses-{int(row.ses)}", "GLMOutputs-mni-std",
        f"trial_keep_run{int(row.run)}.npy")
    if os.path.exists(candidate):
        return candidate


def _load_keep_mask(path):
    keep = np.load(path)
    keep = np.asarray(keep, dtype=bool)
    return keep


def _extract_subject_digits(sub_tag):
    match = re.search(r"(\d+)$", str(sub_tag))
    return match.group(1)


def _resolve_behavior_path(sub_tag, ses, run, behavior_root):
    subject_digits = _extract_subject_digits(sub_tag)
    behavior_path = os.path.join(behavior_root, f"PSPD{subject_digits}_ses_{int(ses)}_run_{int(run)}.npy")
    return behavior_path


def _load_behavior_column(path, behavior_column):
    behavior = np.asarray(np.load(path), dtype=np.float64)
    return behavior[:, int(behavior_column)]


def _demean_finite(values):
    values = np.asarray(values, dtype=np.float64)
    centered = values.astype(np.float64, copy=True)
    finite_mask = np.isfinite(centered)
    if not np.any(finite_mask):
        return centered
    centered[finite_mask] = centered[finite_mask] - float(np.mean(centered[finite_mask]))
    return centered


def _load_run_kept_projection_behavior(segment, behavior_root, behavior_column):
    behavior_path = _resolve_behavior_path(segment["sub_tag"], segment["ses"], segment["run"], behavior_root)
    behavior_column_values = _load_behavior_column(behavior_path, behavior_column)
    n_trials_source = int(segment["n_trials_source"])
    if behavior_column_values.size > n_trials_source:
        behavior_column_values = behavior_column_values[:n_trials_source]
    keep_mask = np.asarray(segment["keep_mask"], dtype=bool)

    kept_behavior = _demean_finite(behavior_column_values[keep_mask])
    kept_projection = _demean_finite(np.asarray(segment["values"], dtype=np.float64))
    return kept_projection, kept_behavior, behavior_path


def _prepare_run_entries(manifest_df, trial_keep_root):
    run_entries = []
    rows = manifest_df.sort_values("offset_start").itertuples(index=False)
    for row in rows:
        keep_path = _resolve_trial_keep_path(row, trial_keep_root)
        keep_mask = _load_keep_mask(keep_path)
        run_entries.append({"sub_tag": str(row.sub_tag), "ses": int(row.ses), "run": int(row.run),
                            "keep_path": keep_path, "keep_mask": keep_mask,
                            "keep_count": int(np.count_nonzero(keep_mask)), "source_count": int(keep_mask.size)})
    return run_entries


def _infer_projection_layout(projection_len, run_entries):
    total_kept = int(sum(entry["keep_count"] for entry in run_entries))
    total_source = int(sum(entry["source_count"] for entry in run_entries))

    if projection_len == total_kept:
        return "kept_only"
    if projection_len == total_source:
        return "source_all"


def split_projection_by_run(projection, manifest_df, trial_keep_root):
    run_entries = _prepare_run_entries(manifest_df, trial_keep_root)
    layout = _infer_projection_layout(int(projection.size), run_entries)

    run_segments = []
    cursor = 0

    for entry in run_entries:
        keep_mask = entry["keep_mask"]
        if layout == "kept_only":
            seg_len = entry["keep_count"]
            run_values = projection[cursor : cursor + seg_len]
            cursor += seg_len
        else:
            seg_len = entry["source_count"]
            run_chunk = projection[cursor : cursor + seg_len]
            run_values = run_chunk[keep_mask]
            cursor += seg_len

        run_segments.append({"sub_tag": entry["sub_tag"], "ses": entry["ses"], "run": entry["run"],
                             "n_trials_source": entry["source_count"], "n_trials_kept": int(run_values.size),
                             "keep_mask": entry["keep_mask"].copy(), "values": run_values})

    return run_segments, layout


def _coefficient_of_variation(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan, np.nan, np.nan
    mean_value = float(np.mean(finite_values))
    std_value = float(np.std(finite_values))
    cv_value = _safe_abs_ratio(std_value, mean_value)
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


def _quartile_coefficient_of_dispersion(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan

    q1, q3 = np.percentile(finite_values, [25.0, 75.0])
    iqr = float(q3 - q1)
    return _safe_abs_ratio(iqr, float(q3 + q1))


def _adjacent_diff_ratio_sum(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size < 2:
        return np.nan

    x_i = finite_values[:-1]
    x_next = finite_values[1:]
    denom = (x_i ** 2) + (x_next ** 2)
    valid_mask = np.isfinite(denom) & (~np.isclose(denom, 0.0))
    if not np.any(valid_mask):
        return np.nan

    numerator = (x_i - x_next) ** 2
    terms = numerator[valid_mask] / denom[valid_mask]
    return float(np.sum(terms))


def _compute_variability_summary(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "variance": np.nan,
            "cv": np.nan,
            "qcd": np.nan,
            "adjacent_diff_ratio_sum": np.nan,
            "mad_mean": np.nan,
            "mad_mean_over_median": np.nan,
            "std_centered_range": np.nan,
        }

    mean_value = float(np.mean(finite_values))
    median_value = float(np.median(finite_values))
    std_value = float(np.std(finite_values))
    variance_value = float(np.var(finite_values))
    _, _, cv_value = _coefficient_of_variation(finite_values)
    qcd_value = _quartile_coefficient_of_dispersion(finite_values)
    adjacent_diff_ratio_sum_value = _adjacent_diff_ratio_sum(finite_values)
    mad_mean_value = _mean_absolute_deviation(finite_values)
    mad_mean_over_median_value = _safe_abs_ratio(mad_mean_value, median_value)
    std_centered_range_value = _std_centered_over_range(finite_values)
    return {
        "mean": mean_value,
        "median": median_value,
        "std": std_value,
        "variance": variance_value,
        "cv": cv_value,
        "qcd": qcd_value,
        "adjacent_diff_ratio_sum": adjacent_diff_ratio_sum_value,
        "mad_mean": mad_mean_value,
        "mad_mean_over_median": mad_mean_over_median_value,
        "std_centered_range": std_centered_range_value,
    }


def compute_run_variances(run_segments):
    """Compatibility helper for scripts that only need projection run variances."""
    rows = []
    for segment in run_segments:
        proj_values = np.asarray(segment["values"], dtype=np.float64)
        proj_finite = proj_values[np.isfinite(proj_values)]
        if proj_finite.size == 0:
            proj_mean = np.nan
            proj_var = np.nan
        else:
            proj_mean = float(np.mean(proj_finite))
            proj_var = float(np.var(proj_finite))
        rows.append({"sub_tag": str(segment["sub_tag"]), "ses": int(segment["ses"]), "run": int(segment["run"]),
                     "n_trials_source": int(segment["n_trials_source"]), "n_trials_kept": int(proj_finite.size),
                     "mean_projection": proj_mean, "variance_projection": proj_var})
    return pd.DataFrame(rows)


def compute_run_behavior_tables(run_segments, behavior_root, behavior_column):
    run_rows = []
    behavior_rows = []
    metric_rows = []

    for segment in run_segments:
        kept_projection, kept_behavior, behavior_path = _load_run_kept_projection_behavior(segment, behavior_root, behavior_column)
        finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
        projection_values = np.asarray(kept_projection[finite_mask], dtype=np.float64)
        behavior_values = np.asarray(kept_behavior[finite_mask], dtype=np.float64)

        proj_stats = _compute_variability_summary(projection_values)
        beh_stats = _compute_variability_summary(behavior_values)

        run_rows.append({
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_source": int(segment["n_trials_source"]),
                "n_trials_kept": int(projection_values.size),
                "mean_projection": proj_stats["mean"],
                "variance_projection": proj_stats["variance"]})

        behavior_rows.append({
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_behavior_source": int(segment["n_trials_source"]),
                "n_trials_behavior_kept": int(behavior_values.size),
                "mean_behavior_col2": beh_stats["mean"],
                "variance_behavior_col2": beh_stats["variance"],
                "behavior_path": behavior_path})

        metric_rows.append({"sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_paired_finite": int(projection_values.size),
                "mean_projection": proj_stats["mean"],
                "median_projection": proj_stats["median"],
                "std_projection": proj_stats["std"],
                "variance_projection": proj_stats["variance"],
                "cv_projection": proj_stats["cv"],
                "qcd_projection": proj_stats["qcd"],
                "adjacent_diff_ratio_sum_projection": proj_stats["adjacent_diff_ratio_sum"],
                "mad_mean_projection": proj_stats["mad_mean"],
                "mad_mean_over_median_projection": proj_stats["mad_mean_over_median"],
                "std_centered_range_projection": proj_stats["std_centered_range"],
                "mean_behavior_col2": beh_stats["mean"],
                "median_behavior_col2": beh_stats["median"],
                "std_behavior_col2": beh_stats["std"],
                "variance_behavior_col2": beh_stats["variance"],
                "cv_behavior_col2": beh_stats["cv"],
                "qcd_behavior_col2": beh_stats["qcd"],
                "adjacent_diff_ratio_sum_behavior_col2": beh_stats["adjacent_diff_ratio_sum"],
                "mad_mean_behavior_col2": beh_stats["mad_mean"],
                "mad_mean_over_median_behavior_col2": beh_stats["mad_mean_over_median"],
                "std_centered_range_behavior_col2": beh_stats["std_centered_range"]})

    return (pd.DataFrame(run_rows), pd.DataFrame(behavior_rows), pd.DataFrame(metric_rows))


def _scale_values(values, method):
    scaled = np.full(values.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return scaled

    finite_values = values[finite_mask]
    method = str(method).lower()
    if method == "zscore":
        center = float(np.mean(finite_values))
        spread = float(np.std(finite_values))
        if not np.isfinite(spread) or spread <= 0:
            scaled[finite_mask] = 0.0
        else:
            scaled[finite_mask] = (finite_values - center) / spread
        return scaled
    if method == "minmax":
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
        spread = upper - lower
        if spread <= 0:
            scaled[finite_mask] = 0.0
        else:
            scaled[finite_mask] = (finite_values - lower) / spread
        return scaled


def build_projection_behavior_comparison(run_df, behavior_df, scale_method):
    comparison_df = run_df.merge(behavior_df, on=["sub_tag", "ses", "run"], how="inner", validate="one_to_one")
    comparison_df["variance_projection_scaled"] = _scale_values(comparison_df["variance_projection"],
                                                                method=scale_method)
    comparison_df["variance_behavior_scaled"] = _scale_values(comparison_df["variance_behavior_col2"],
                                                              method=scale_method)
    return comparison_df


def _finite_median(values):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.median(finite_values))


def add_subject_median_normalized_variance_features(comparison_df):
    """Add subject-wise median baselines and log-relative variance features."""
    df = comparison_df.copy()

    df["subject_median_variance_projection"] = df.groupby("sub_tag")["variance_projection"].transform(lambda s: _finite_median(s))
    df["subject_median_variance_behavior_col2"] = df.groupby("sub_tag")[
        "variance_behavior_col2"
    ].transform(lambda s: _finite_median(s))

    proj = df["variance_projection"]
    beh = df["variance_behavior_col2"]
    proj_med = df["subject_median_variance_projection"]
    beh_med = df["subject_median_variance_behavior_col2"]

    proj_rel = np.full(proj.shape, np.nan)
    beh_rel = np.full(beh.shape, np.nan)

    proj_rel_mask = np.isfinite(proj) & np.isfinite(proj_med) & (proj_med > 0.0)
    beh_rel_mask = np.isfinite(beh) & np.isfinite(beh_med) & (beh_med > 0.0)
    proj_rel[proj_rel_mask] = proj[proj_rel_mask] / proj_med[proj_rel_mask]
    beh_rel[beh_rel_mask] = beh[beh_rel_mask] / beh_med[beh_rel_mask]

    proj_log_rel = np.full(proj.shape, np.nan)
    beh_log_rel = np.full(beh.shape, np.nan)
    proj_log_mask = np.isfinite(proj_rel) & (proj_rel > 0.0)
    beh_log_mask = np.isfinite(beh_rel) & (beh_rel > 0.0)
    proj_log_rel[proj_log_mask] = np.log(proj_rel[proj_log_mask])
    beh_log_rel[beh_log_mask] = np.log(beh_rel[beh_log_mask])

    log_rel_diff = np.full(proj.shape, np.nan)
    paired_log_mask = np.isfinite(proj_log_rel) & np.isfinite(beh_log_rel)
    log_rel_diff[paired_log_mask] = proj_log_rel[paired_log_mask] - beh_log_rel[paired_log_mask]

    df["variance_projection_rel_subject_median"] = proj_rel
    df["variance_behavior_rel_subject_median"] = beh_rel
    df["log_variance_projection_rel_subject_median"] = proj_log_rel
    df["log_variance_behavior_rel_subject_median"] = beh_log_rel
    df["log_rel_diff_proj_minus_beh"] = log_rel_diff
    return df


# def _one_sided_p_less_from_two_sided(two_sided_p, estimate):
#     if not np.isfinite(two_sided_p) or not np.isfinite(estimate):
#         return np.nan
#     if estimate < 0:
#         return float(two_sided_p / 2.0)
#     return float(1.0 - (two_sided_p / 2.0))


def _iqr_outlier_mask(values, iqr_multiplier=3.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(values.shape, dtype=bool)

    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1)
    lower = float(q1 - (float(iqr_multiplier) * iqr))
    upper = float(q3 + (float(iqr_multiplier) * iqr))
    return (values < lower) | (values > upper)


def _build_pair_labels(metric_df):
    if {"sub_tag", "ses", "run"}.issubset(metric_df.columns):
        ses_values = pd.to_numeric(metric_df["ses"], errors="coerce").to_numpy(dtype=np.float64)
        run_values = pd.to_numeric(metric_df["run"], errors="coerce").to_numpy(dtype=np.float64)
        sub_values = metric_df["sub_tag"].astype(str).to_numpy()
        labels = []
        for sub_tag, ses_num, run_num in zip(sub_values, ses_values, run_values):
            if np.isfinite(ses_num) and np.isfinite(run_num):
                labels.append(f"{str(sub_tag)}_ses{int(ses_num)}_run{int(run_num)}")
            else:
                labels.append(f"{str(sub_tag)}_ses{ses_num}_run{run_num}")
        return np.asarray(labels, dtype=str)

    if "sub_tag" in metric_df.columns:
        return metric_df["sub_tag"].astype(str).to_numpy()

    return np.arange(len(metric_df)).astype(str)


def _paired_outlier_filtered(values_a, values_b, labels=None, iqr_multiplier=3.0):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)

    if labels is None:
        labels = np.arange(values_a.size).astype(str)
    else:
        labels = np.asarray(labels).astype(str)

    paired_mask = np.isfinite(values_a) & np.isfinite(values_b)
    paired_a = values_a[paired_mask]
    paired_b = values_b[paired_mask]
    paired_labels = labels[paired_mask]

    if paired_a.size == 0:
        out_any = np.zeros(paired_a.shape, dtype=bool)
    elif iqr_multiplier is None:
        out_any = np.zeros(paired_a.shape, dtype=bool)
    else:
        out_a = _iqr_outlier_mask(paired_a, iqr_multiplier=iqr_multiplier)
        out_b = _iqr_outlier_mask(paired_b, iqr_multiplier=iqr_multiplier)
        out_any = out_a | out_b
    keep_mask = ~out_any

    return {"input_paired_mask": paired_mask,
        "paired_outlier_mask": out_any,
        "paired_a_all": paired_a,
        "paired_b_all": paired_b,
        "paired_labels_all": paired_labels,
        "paired_a_kept": paired_a[keep_mask],
        "paired_b_kept": paired_b[keep_mask],
        "paired_labels_kept": paired_labels[keep_mask],
        "paired_labels_removed": paired_labels[out_any],
        "n_pairs_total": int(paired_a.size),
        "n_pairs_removed": int(np.count_nonzero(out_any)),
        "n_pairs_kept": int(np.count_nonzero(keep_mask))}


def _paired_tests_two_sided(values_a, values_b):
    row = {"n_pairs": int(values_a.size),
        "mean_diff_a_minus_b": np.nan,
        "median_diff_a_minus_b": np.nan,
        "ttest_stat": np.nan,
        "ttest_p_two_sided": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p_two_sided": np.nan}

    diff = values_a - values_b
    row["mean_diff_a_minus_b"] = float(np.mean(diff))
    row["median_diff_a_minus_b"] = float(np.median(diff))

    if values_a.size >= 2:
        t_res = ttest_rel(values_a, values_b, nan_policy="omit")
        row["ttest_stat"] = float(t_res.statistic)
        row["ttest_p_two_sided"] = float(t_res.pvalue)

    if np.allclose(diff, 0.0):
        row["wilcoxon_stat"] = 0.0
        row["wilcoxon_p_two_sided"] = 1.0
    else:
        w_res = wilcoxon(values_a, values_b, alternative="two-sided")
        row["wilcoxon_stat"] = float(w_res.statistic)
        row["wilcoxon_p_two_sided"] = float(w_res.pvalue)
    return row


def _mixedlm_projection_effect(
    paired_df,
    behavior_value_col="behavior_raw",
    projection_value_col="projection_raw",
    subject_col="sub_tag",
):
    required_cols = {subject_col, behavior_value_col, projection_value_col}
    missing_cols = required_cols.difference(paired_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for mixed-effects model: {sorted(missing_cols)}"
        )

    model_df = paired_df.loc[
        :,
        [subject_col, behavior_value_col, projection_value_col],
    ].copy()
    model_df = model_df.rename(
        columns={
            subject_col: "subject_id",
            behavior_value_col: "behavior_value",
            projection_value_col: "projection_value",
        }
    )
    model_df["subject_id"] = model_df["subject_id"].astype(str)
    model_df = model_df.loc[
        np.isfinite(model_df["behavior_value"]) & np.isfinite(model_df["projection_value"])
    ].reset_index(drop=True)

    row = {
        "n_runs": int(model_df.shape[0]),
        "n_subjects": int(model_df["subject_id"].nunique()),
        "mean_diff_projection_minus_behavior": np.nan,
        "median_diff_projection_minus_behavior": np.nan,
        "lme_coef_projection_minus_behavior": np.nan,
        "lme_se_projection_minus_behavior": np.nan,
        "lme_z_projection_minus_behavior": np.nan,
        "lme_p_two_sided": np.nan,
    }
    if model_df.empty:
        return row

    diff = (
        model_df["projection_value"].to_numpy(dtype=np.float64)
        - model_df["behavior_value"].to_numpy(dtype=np.float64)
    )
    row["mean_diff_projection_minus_behavior"] = float(np.mean(diff))
    row["median_diff_projection_minus_behavior"] = float(np.median(diff))

    if row["n_subjects"] < 2 or row["n_runs"] < 2:
        return row

    behavior_long = model_df.loc[:, ["subject_id", "behavior_value"]].copy()
    behavior_long["signal"] = "Behaviour"
    behavior_long["value"] = behavior_long.pop("behavior_value")

    projection_long = model_df.loc[:, ["subject_id", "projection_value"]].copy()
    projection_long["signal"] = "Projection"
    projection_long["value"] = projection_long.pop("projection_value")

    long_df = pd.concat([behavior_long, projection_long], axis=0, ignore_index=True)
    long_df["signal"] = pd.Categorical(
        long_df["signal"],
        categories=["Behaviour", "Projection"],
        ordered=True,
    )

    fit = None
    for method in ("lbfgs", "powell", "bfgs", "cg", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    "value ~ signal",
                    data=long_df,
                    groups=long_df["subject_id"],
                    re_formula="1",
                ).fit(reml=False, method=method, disp=False)
            break
        except Exception:
            fit = None

    if fit is None:
        return row

    coef_name = "signal[T.Projection]"
    row["lme_coef_projection_minus_behavior"] = float(fit.params.get(coef_name, np.nan))
    row["lme_se_projection_minus_behavior"] = float(fit.bse.get(coef_name, np.nan))
    row["lme_z_projection_minus_behavior"] = float(fit.tvalues.get(coef_name, np.nan))
    row["lme_p_two_sided"] = float(fit.pvalues.get(coef_name, np.nan))

    return row


def _bootstrap_ci(values, statistic="mean", n_bootstrap=10000, ci_percent=95.0, random_seed=0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan

    ci_percent = float(ci_percent)
    n_bootstrap = int(n_bootstrap)

    rng = np.random.default_rng(int(random_seed))
    sample_idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    sampled = values[sample_idx]

    statistic = str(statistic).lower()
    if statistic == "mean":
        boot_stats = np.mean(sampled, axis=1)
    elif statistic == "median":
        boot_stats = np.median(sampled, axis=1)

    alpha = 100.0 - ci_percent
    lower_q = alpha / 2.0
    upper_q = 100.0 - lower_q
    return (np.percentile(boot_stats, lower_q), np.percentile(boot_stats, upper_q))


def _paired_sign_flip_permutation_test(diff_values, n_permutations=20000, random_seed=0):
    diff_values = np.asarray(diff_values, dtype=np.float64)
    diff_values = diff_values[np.isfinite(diff_values)]
    if diff_values.size == 0:
        return np.nan, np.nan

    observed = float(np.mean(diff_values))
    n_permutations = int(n_permutations)

    rng = np.random.default_rng(int(random_seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diff_values.size))
    permuted_means = np.mean(signs * diff_values[None, :], axis=1)
    p_two_sided = (np.count_nonzero(np.abs(permuted_means) >= abs(observed)) + 1.0) / (n_permutations + 1.0)
    return observed, float(p_two_sided)


def compute_subject_projection_behavior_metrics(run_segments, behavior_root,behavior_column, excluded_run_keys=None):
    excluded_keys = set(excluded_run_keys or [])
    rows = []
    subject_keys = sorted({(str(segment["sub_tag"])) for segment in run_segments},
        key=lambda sub: int(_extract_subject_digits(sub)))

    for sub_tag in subject_keys:
        subject_segments = [segment for segment in run_segments if str(segment["sub_tag"]) == str(sub_tag)]
        projection_chunks = []
        behavior_chunks = []
        n_runs = 0
        n_trials_paired = 0
        for segment in subject_segments:
            run_key = (str(segment["sub_tag"]), int(segment["ses"]), int(segment["run"]))
            if run_key in excluded_keys:
                continue

            kept_projection, kept_behavior, _ = _load_run_kept_projection_behavior(segment, behavior_root, behavior_column)
            finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
            projection_values = kept_projection[finite_mask]
            behavior_values = kept_behavior[finite_mask]
            if projection_values.size == 0:
                continue
            projection_chunks.append(np.asarray(projection_values, dtype=np.float64))
            behavior_chunks.append(np.asarray(behavior_values, dtype=np.float64))
            n_trials_paired += int(projection_values.size)
            n_runs += 1

        if n_trials_paired == 0:
            proj_values = np.asarray([], dtype=np.float64)
            beh_values = np.asarray([], dtype=np.float64)
        else:
            proj_values = np.concatenate(projection_chunks, axis=0)
            beh_values = np.concatenate(behavior_chunks, axis=0)

        proj_stats = _compute_variability_summary(proj_values)
        beh_stats = _compute_variability_summary(beh_values)

        rows.append(
            {"sub_tag": str(sub_tag),
                "n_runs_with_paired_trials": int(n_runs),
                "n_trials_paired_finite": int(n_trials_paired),
                "mean_projection": proj_stats["mean"],
                "median_projection": proj_stats["median"],
                "variance_projection": proj_stats["variance"],
                "std_projection": proj_stats["std"],
                "cv_projection": proj_stats["cv"],
                "qcd_projection": proj_stats["qcd"],
                "adjacent_diff_ratio_sum_projection": proj_stats["adjacent_diff_ratio_sum"],
                "mad_mean_projection": proj_stats["mad_mean"],
                "mad_mean_over_median_projection": proj_stats["mad_mean_over_median"],
                "std_centered_range_projection": proj_stats["std_centered_range"],
                "mean_behavior_col2": beh_stats["mean"],
                "median_behavior_col2": beh_stats["median"],
                "variance_behavior_col2": beh_stats["variance"],
                "std_behavior_col2": beh_stats["std"],
                "cv_behavior_col2": beh_stats["cv"],
                "qcd_behavior_col2": beh_stats["qcd"],
                "adjacent_diff_ratio_sum_behavior_col2": beh_stats["adjacent_diff_ratio_sum"],
                "mad_mean_behavior_col2": beh_stats["mad_mean"],
                "mad_mean_over_median_behavior_col2": beh_stats["mad_mean_over_median"],
                "std_centered_range_behavior_col2": beh_stats["std_centered_range"],
                "d_var_projection_minus_behavior": (
                    float(proj_stats["variance"] - beh_stats["variance"])
                    if np.isfinite(proj_stats["variance"]) and np.isfinite(beh_stats["variance"])
                    else np.nan
                ),
                "d_cv_projection_minus_behavior": (
                    float(proj_stats["cv"] - beh_stats["cv"])
                    if np.isfinite(proj_stats["cv"]) and np.isfinite(beh_stats["cv"])
                    else np.nan
                ),
                "d_qcd_projection_minus_behavior": (
                    float(proj_stats["qcd"] - beh_stats["qcd"])
                    if np.isfinite(proj_stats["qcd"]) and np.isfinite(beh_stats["qcd"])
                    else np.nan
                ),
                "d_adjacent_diff_ratio_sum_projection_minus_behavior": (
                    float(proj_stats["adjacent_diff_ratio_sum"] - beh_stats["adjacent_diff_ratio_sum"])
                    if (
                        np.isfinite(proj_stats["adjacent_diff_ratio_sum"])
                        and np.isfinite(beh_stats["adjacent_diff_ratio_sum"])
                    )
                    else np.nan
                ),
                "d_mad_mean_projection_minus_behavior": (
                    float(proj_stats["mad_mean"] - beh_stats["mad_mean"])
                    if np.isfinite(proj_stats["mad_mean"]) and np.isfinite(beh_stats["mad_mean"])
                    else np.nan
                ),
                "d_mad_mean_over_median_projection_minus_behavior": (
                    float(proj_stats["mad_mean_over_median"] - beh_stats["mad_mean_over_median"])
                    if (
                        np.isfinite(proj_stats["mad_mean_over_median"])
                        and np.isfinite(beh_stats["mad_mean_over_median"])
                    )
                    else np.nan
                ),
                "d_std_centered_range_projection_minus_behavior": (
                    float(proj_stats["std_centered_range"] - beh_stats["std_centered_range"])
                    if (
                        np.isfinite(proj_stats["std_centered_range"])
                        and np.isfinite(beh_stats["std_centered_range"])
                    )
                    else np.nan
                )})
    return pd.DataFrame(rows)


def identify_run_outliers(metric_df, projection_col, behavior_col, outlier_iqr_multiplier=3.0):
    required_cols = {"sub_tag", "ses", "run", projection_col, behavior_col}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for run outlier detection: {sorted(missing_cols)}"
        )

    labels = _build_pair_labels(metric_df)
    paired = _paired_outlier_filtered(metric_df[projection_col],metric_df[behavior_col],
                                      labels=labels, iqr_multiplier=outlier_iqr_multiplier)

    paired_rows = metric_df.loc[paired["input_paired_mask"], ["sub_tag", "ses", "run"]].copy()
    paired_rows[projection_col] = paired["paired_a_all"]
    paired_rows[behavior_col] = paired["paired_b_all"]
    paired_rows["pair_label"] = paired["paired_labels_all"]
    paired_rows["removed_outlier"] = paired["paired_outlier_mask"]

    outlier_rows = paired_rows.loc[paired_rows["removed_outlier"]].copy()
    outlier_keys = {(str(row.sub_tag), int(row.ses), int(row.run)) for row in outlier_rows.itertuples(index=False)}
    return outlier_keys, paired_rows


def _evaluate_density(values, x):
    values = values[np.isfinite(values)]

    if values.size < 2 or np.allclose(values, values[0]):
        width = max(np.std(values), max(abs(values[0]) * 0.05, 1e-6))
        density = (np.exp(-0.5 * ((x - values[0]) / width) ** 2)/ (width * np.sqrt(2.0 * np.pi)))
    else:
        density = gaussian_kde(values).evaluate(x)

    area = np.trapezoid(density, x)
    if area > 0:
        density = density / area
    return density


def _density_grid(values, grid_points=512, pad_fraction=0.1, fallback_pad=0.25):
    values = values[np.isfinite(values)]
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = vmax - vmin
    pad = pad_fraction * vrange if vrange > 0 else fallback_pad
    x = np.linspace(vmin - pad, vmax + pad, int(grid_points))
    return x


def plot_run_variance_density(run_df, out_path, grid_points=512):
    scale_factor = 1e7
    values = run_df["variance_projection"].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)] * scale_factor
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = vmax - vmin

    if values.size < 2 or np.allclose(values, values[0]):
        width = max(abs(vmin) * 0.05, 1e-12)
        x = np.linspace(max(0.0, vmin - 4.0 * width), vmax + 4.0 * width, int(grid_points))
        density = (np.exp(-0.5 * ((x - vmin) / width) ** 2) / (width * np.sqrt(2.0 * np.pi)))
    else:
        pad = 0.1 * vrange if vrange > 0 else max(abs(vmin) * 0.1, 1e-12)
        x = np.linspace(max(0.0, vmin - pad), vmax + pad, int(grid_points))
        kde = gaussian_kde(values)
        density = kde.evaluate(x)

    # Variance cannot be negative; enforce zero density below the observed minimum.
    # This avoids non-zero KDE tails at x=0 when no run/session has zero variance.
    if vmin > 0.0:
        density = np.where(x < vmin, 0.0, density)

    area = np.trapezoid(density, x)
    if area > 0:
        density = density / area

    plt.figure(figsize=(8, 5))
    plt.plot(x, density, color="tab:blue", linewidth=2.0, label="KDE")
    plt.fill_between(x, density, alpha=0.2, color="tab:blue")
    plt.xlabel(f"Variance (scaled by {scale_factor:.0e})")
    plt.ylabel("Probability density")
    plt.title("Smooth distribution of run/session variances")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _category_sort_key(value, category_name):
    if str(category_name) == "sub_tag":
        return (0, int(_extract_subject_digits(value)))
    return (0, float(value))


def _build_category_color_map(values, category_name):
    unique_values = sorted({value for value in values},
        key=lambda value: _category_sort_key(value, category_name))
    n_values = len(unique_values)
    if n_values == 0:
        return {}, []

    if str(category_name) == "run":
        run_colors = ["#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#ff7f0e",  # orange
            "#9467bd",  # purple
            "#17becf",  # cyan
            ]
        color_map = {value: run_colors[idx % len(run_colors)] for idx, value in enumerate(unique_values)}
        return color_map, unique_values

    if str(category_name) == "ses":
        session_colors = [
            "#1f77b4",  # blue
            "#ff7f0e",  # orange
            "#2ca02c",  # green
            "#d62728",  # red
            "#9467bd",  # purple
            "#8c564b",  # brown
        ]
        color_map = {value: session_colors[idx % len(session_colors)]
            for idx, value in enumerate(unique_values)}
        return color_map, unique_values

    if str(category_name) == "sub_tag":
        # Build a larger qualitative palette to minimize subject-color collisions.
        tab20 = list(plt.get_cmap("tab20").colors)
        tab20b = list(plt.get_cmap("tab20b").colors)
        tab20c = list(plt.get_cmap("tab20c").colors)
        subject_palette = tab20 + tab20b + tab20c
        if n_values <= len(subject_palette):
            step = 11  # coprime with 60, spreads adjacent picks across the palette
            spread_palette = [subject_palette[(idx * step) % len(subject_palette)]
                for idx in range(len(subject_palette))]
            color_map = {value: spread_palette[idx]
                for idx, value in enumerate(unique_values)}
            return color_map, unique_values

    if n_values <= 10:
        cmap = plt.get_cmap("tab10", n_values)
    elif n_values <= 20:
        cmap = plt.get_cmap("tab20", n_values)
    else:
        cmap = plt.get_cmap("gist_ncar", n_values)

    color_map = {value: cmap(idx) for idx, value in enumerate(unique_values)}
    return color_map, unique_values


def _build_paired_metric_zscore_df(metric_df, projection_col, behavior_col):
    required_cols = {"sub_tag", "ses", "run", projection_col, behavior_col}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for z-score paired metric table: {sorted(missing_cols)}"
        )

    projection_z = _scale_values(metric_df[projection_col], method="zscore")
    behavior_z = _scale_values(metric_df[behavior_col], method="zscore")
    paired_mask = np.isfinite(projection_z) & np.isfinite(behavior_z)

    paired_df = metric_df.loc[paired_mask, ["sub_tag", "ses", "run"]].copy()
    paired_df = paired_df.reset_index(drop=True)
    paired_df["projection_z"] = projection_z[paired_mask]
    paired_df["behavior_z"] = behavior_z[paired_mask]
    return paired_df


def _build_paired_metric_raw_df(metric_df, projection_col, behavior_col):
    required_cols = {"sub_tag", "ses", "run", projection_col, behavior_col}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for raw paired metric table: {sorted(missing_cols)}"
        )

    projection_raw = metric_df[projection_col]
    behavior_raw = metric_df[behavior_col]
    paired_mask = np.isfinite(projection_raw) & np.isfinite(behavior_raw)

    paired_df = metric_df.loc[paired_mask, ["sub_tag", "ses", "run"]].copy()
    paired_df = paired_df.reset_index(drop=True)
    paired_df["projection_raw"] = projection_raw[paired_mask]
    paired_df["behavior_raw"] = behavior_raw[paired_mask]
    return paired_df


def _plot_paired_box_with_connections(ax, paired_df, group_col, color_map, jitter_seed=0,
                                      y_limits=None, behavior_value_col="behavior_z",
                                      projection_value_col="projection_z", x_tick_labels=None):
    behavior_values = paired_df[behavior_value_col]
    projection_values = paired_df[projection_value_col]
    group_values = paired_df[group_col].tolist()

    box_parts = ax.boxplot([behavior_values, projection_values], positions=[0.0, 1.0], widths=0.5,
                           patch_artist=True, showfliers=False, showmeans=True,
                           meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black",
                                      "markersize": 4.0}, zorder=1)
    for box_patch in box_parts["boxes"]:
        box_patch.set(facecolor="0.94", edgecolor="0.35", linewidth=1.15)
    for whisker in box_parts["whiskers"]:
        whisker.set(color="0.4", linewidth=1.0)
    for cap in box_parts["caps"]:
        cap.set(color="0.4", linewidth=1.0)
    for median in box_parts["medians"]:
        median.set(color="0.1", linewidth=1.6)

    rng = np.random.default_rng(int(jitter_seed))
    jitter_base = rng.uniform(-0.12, 0.12, size=behavior_values.size)
    x_behavior = jitter_base + rng.uniform(-0.03, 0.03, size=behavior_values.size)
    x_projection = 1.0 + jitter_base + rng.uniform(-0.03, 0.03, size=behavior_values.size)

    clipped_count = 0
    if y_limits is None:
        y_min = None
        y_max = None
    else:
        y_min = float(y_limits[0])
        y_max = float(y_limits[1])

    for x0, x1, y0, y1, group_value in zip(x_behavior, x_projection, behavior_values, projection_values,
                                           group_values):
        color = color_map[group_value]
        if y_min is None or y_max is None:
            y0_plot = y0
            y1_plot = y1
            marker0 = "o"
            marker1 = "o"
        else:
            y0_plot = float(np.clip(y0, y_min, y_max))
            y1_plot = float(np.clip(y1, y_min, y_max))
            if y0 > y_max:
                marker0 = "^"
                clipped_count += 1
            elif y0 < y_min:
                marker0 = "v"
                clipped_count += 1
            else:
                marker0 = "o"
            if y1 > y_max:
                marker1 = "^"
                clipped_count += 1
            elif y1 < y_min:
                marker1 = "v"
                clipped_count += 1
            else:
                marker1 = "o"

        ax.plot([x0, x1], [y0_plot, y1_plot], color=color, linewidth=0.85, alpha=0.28, zorder=2)
        ax.scatter([x0], [y0_plot], s=34 if marker0 != "o" else 30, color=color, alpha=0.9, edgecolors="0.15",
                   linewidths=0.3, marker=marker0, zorder=3)
        ax.scatter([x1], [y1_plot], s=34 if marker1 != "o" else 30, color=color, alpha=0.9, edgecolors="0.15",
                   linewidths=0.3, marker=marker1, zorder=3)

    ax.set_xlim(-0.45, 1.45)
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xticks([0.0, 1.0])
    if x_tick_labels is None:
        x_tick_labels = ("Behavior variability", "BOLD variability")
    ax.set_xticklabels([str(x_tick_labels[0]), str(x_tick_labels[1])])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    return int(clipped_count)


def _add_category_legend(ax, color_map, category_values, title, loc="upper right"):
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_map[value],
            markeredgecolor="0.25", markersize=5.5, label=str(value)) for value in category_values]
    ncol = 1
    if len(handles) > 16:
        ncol = 2

    legend = ax.legend(handles=handles, title=str(title), loc=str(loc), fontsize=7.5, title_fontsize=8.5,
                       frameon=True, ncol=ncol, borderaxespad=0.4, handletextpad=0.35, columnspacing=0.8,
                       labelspacing=0.25)
    legend.get_frame().set_alpha(0.95)


def _metric_specs_from_keys(metric_keys=None):
    if metric_keys is None:
        return [dict(spec) for spec in METRIC_SPECS]

    by_key = {str(spec["key"]): dict(spec) for spec in METRIC_SPECS}
    specs = []
    for key in metric_keys:
        key = str(key)
        if key not in by_key:
            raise ValueError(f"Unknown metric key: {key}. Available keys: {sorted(by_key)}")
        specs.append(dict(by_key[key]))
    return specs


def _metric_z_column_title(metric_spec):
    return f"Z-scored {str(metric_spec['label'])}"


def _metric_raw_column_title(metric_spec):
    label = str(metric_spec["label"])
    if label == "MAD(mean)":
        return "MAD"
    if label == "MAD(mean)/|median|":
        return "MAD/|median|"
    return label


def _validate_metric_df(metric_df, metric_specs=None):
    specs = _metric_specs_from_keys(
        None if metric_specs is None else [spec["key"] for spec in metric_specs]
    )
    required_cols = {"sub_tag", "ses", "run"}
    for spec in specs:
        required_cols.add(str(spec["projection_col"]))
        required_cols.add(str(spec["behavior_col"]))
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for metric plots: {sorted(missing_cols)}"
        )


def _build_metric_column_dfs(metric_df, metric_specs=None, use_zscore=False):
    specs = _metric_specs_from_keys(
        None if metric_specs is None else [spec["key"] for spec in metric_specs]
    )
    _validate_metric_df(metric_df, metric_specs=specs)

    column_defs = []
    for spec in specs:
        if bool(use_zscore):
            paired_df = _build_paired_metric_zscore_df(
                metric_df=metric_df,
                projection_col=str(spec["projection_col"]),
                behavior_col=str(spec["behavior_col"]),
            )
            column_title = _metric_z_column_title(spec)
        else:
            paired_df = _build_paired_metric_raw_df(
                metric_df=metric_df,
                projection_col=str(spec["projection_col"]),
                behavior_col=str(spec["behavior_col"]),
            )
            column_title = _metric_raw_column_title(spec)
        if paired_df.empty:
            continue
        column_defs.append((column_title, paired_df, spec))
    if len(column_defs) == 0:
        raise RuntimeError("No metrics with finite paired values are available for plotting.")
    return column_defs


def _reshape_axes_grid(axes, n_rows, n_cols):
    if n_rows == 1 and n_cols == 1:
        return np.asarray([[axes]])
    if n_rows == 1:
        return np.asarray([axes])
    if n_cols == 1:
        return np.asarray(axes).reshape(n_rows, 1)
    return axes


def plot_variance_cv_subject_session_run_3x2(metric_df, out_path):
    column_defs = _build_metric_column_dfs(metric_df=metric_df, metric_specs=METRIC_SPECS[1:], use_zscore=False)
    row_defs = [("sub_tag", "Subject colors", "Subject"),
        ("ses", "Session colors", "Session"),
        ("run", "Run colors", "Run")]

    column_lme_stats = []
    for _, metric_col_df, _ in column_defs:
        lme_stats = _mixedlm_projection_effect(
            metric_col_df,
            behavior_value_col="behavior_raw",
            projection_value_col="projection_raw",
            subject_col="sub_tag",
        )
        column_lme_stats.append(lme_stats)

    column_limits = []
    for _, metric_col_df, _ in column_defs:
        column_values = np.concatenate([metric_col_df["behavior_raw"], metric_col_df["projection_raw"]])
        finite_values = column_values[np.isfinite(column_values)]
        if finite_values.size == 0:
            column_limits.append((-1.0, 1.0))
            continue
        q_low = float(np.percentile(finite_values, 2.0))
        q_high = float(np.percentile(finite_values, 98.0))
        q_span = q_high - q_low
        pad = 0.18 * q_span if q_span > 0 else 0.55
        y_low = q_low - pad
        y_high = q_high + pad
        column_limits.append((y_low, y_high))

    n_cols = len(column_defs)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(6.0 * n_cols, 12.0), 14.2))
    axes = _reshape_axes_grid(axes, n_rows=3, n_cols=n_cols)
    total_clipped = 0
    for row_idx, (group_col, row_title, legend_title) in enumerate(row_defs):
        for col_idx, (column_title, metric_col_df, metric_spec) in enumerate(column_defs):
            ax = axes[row_idx, col_idx]
            color_map, category_values = _build_category_color_map(metric_col_df[group_col].tolist(),
                                                                   category_name=group_col)
            clipped_here = _plot_paired_box_with_connections(ax=ax, paired_df=metric_col_df,
                                                             group_col=group_col, color_map=color_map,
                                                             jitter_seed=111 + 17 * row_idx + 29 * col_idx,
                                                             y_limits=column_limits[col_idx],
                                                             behavior_value_col="behavior_raw",
                                                             projection_value_col="projection_raw",
                                                             x_tick_labels=(
                                                                 f"Behavior {metric_spec['label']}",
                                                                 f"BOLD {metric_spec['label']}",
                                                             ))
            total_clipped += int(clipped_here)
            ax.set_ylim(column_limits[col_idx])
            ax.set_title(f"{column_title}", fontsize=11.0)
            if row_idx == 0:
                test_row = column_lme_stats[col_idx]
                if np.isfinite(test_row["lme_p_two_sided"]) and np.isfinite(
                    test_row["lme_z_projection_minus_behavior"]
                ):
                    ttest_text = (
                        f"LME (Projection-Behaviour)\n"
                        f"p={test_row['lme_p_two_sided']:.3g}, "
                        f"z={test_row['lme_z_projection_minus_behavior']:.3g}, "
                        f"beta={test_row['lme_coef_projection_minus_behavior']:.3g}\n"
                        f"subjects={int(test_row['n_subjects'])}, runs={int(test_row['n_runs'])}"
                    )
                else:
                    ttest_text = (
                        "LME (Projection-Behaviour)\n"
                        "fit unavailable"
                    )
                ax.text(
                    0.02,
                    0.98,
                    ttest_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8.3,
                    bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.75"},
                )
            if col_idx == 0:
                ax.set_ylabel("Variability")
            if col_idx == (n_cols - 1):
                legend_loc = "lower right" if row_idx == 0 else "upper right"
                _add_category_legend(
                    ax=ax,
                    color_map=color_map,
                    category_values=category_values,
                    title=legend_title,
                    loc=legend_loc,
                )

    fig.suptitle("Behavior vs BOLD variability (no z-score)", fontsize=13.0)
    tight_bottom = 0.0
    fig.tight_layout(rect=(0.0, tight_bottom, 0.995, 0.965))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_metric_grid_first_row_second_column(metric_df, out_path):
    column_defs = _build_metric_column_dfs(
        metric_df=metric_df,
        metric_specs=METRIC_SPECS[1:],
        use_zscore=False,
    )
    if len(column_defs) < 2:
        raise RuntimeError(
            "The metric grid does not have a second column to export as a standalone figure."
        )
    column_title, metric_col_df, metric_spec = column_defs[1]
    metric_col_df = metric_col_df.loc[
        metric_col_df["sub_tag"].astype(str).map(lambda value: int(_extract_subject_digits(value))) != 17
    ].reset_index(drop=True)
    if metric_col_df.empty:
        raise RuntimeError("No rows remain for the standalone panel after excluding subject 17.")

    combined_values = np.concatenate(
        [metric_col_df["behavior_raw"], metric_col_df["projection_raw"]]
    )
    finite_values = combined_values[np.isfinite(combined_values)]
    if finite_values.size == 0:
        y_limits = (-1.0, 1.0)
    else:
        q_low = float(np.percentile(finite_values, 2.0))
        q_high = float(np.percentile(finite_values, 98.0))
        q_span = q_high - q_low
        pad = 0.18 * q_span if q_span > 0 else 0.55
        y_limits = (q_low - pad, q_high + pad)

    color_map, category_values = _build_category_color_map(
        metric_col_df["sub_tag"].tolist(),
        category_name="sub_tag",
    )
    lme_stats = _mixedlm_projection_effect(
        metric_col_df,
        behavior_value_col="behavior_raw",
        projection_value_col="projection_raw",
        subject_col="sub_tag",
    )

    if str(metric_spec["key"]) == "adjacent_diff_ratio_sum":
        panel_title = "Behavior vs BOLD adjacent-trial change ratio across runs"
        x_tick_labels = (
            "Behaviour",
            "Projection",
        )
    else:
        panel_title = f"Behavior vs BOLD {column_title} across runs, colored by subject"
        x_tick_labels = (
            "Behaviour",
            "Projection",
        )

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    _plot_paired_box_with_connections(
        ax=ax,
        paired_df=metric_col_df,
        group_col="sub_tag",
        color_map=color_map,
        jitter_seed=140,
        y_limits=y_limits,
        behavior_value_col="behavior_raw",
        projection_value_col="projection_raw",
        x_tick_labels=x_tick_labels,
    )
    ax.set_ylim(y_limits)
    ax.set_ylabel("Variability")
    ax.set_title("")

    if np.isfinite(lme_stats["lme_p_two_sided"]) and np.isfinite(
        lme_stats["lme_z_projection_minus_behavior"]
    ):
        ttest_text = (
            "LME (Projection-Behaviour)\n"
            f"p={lme_stats['lme_p_two_sided']:.3g}, "
            f"z={lme_stats['lme_z_projection_minus_behavior']:.3g}, "
            f"beta={lme_stats['lme_coef_projection_minus_behavior']:.3g}"
        )
    else:
        ttest_text = "LME (Projection-Behaviour)\nfit unavailable"
    ax.text(
        0.70,
        0.98,
        ttest_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.3,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "0.75",
        },
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color_map[value],
            markeredgecolor="0.25",
            markersize=5.5,
            label=str(value),
        )
        for value in category_values
    ]
    ax.legend(
        handles=handles,
        title="Subject",
        loc="center left",
        bbox_to_anchor=(1.02, 0.42),
        fontsize=7.5,
        title_fontsize=8.5,
        frameon=True,
        ncol=2,
        borderaxespad=0.4,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.25,
    )

    fig.tight_layout(rect=(0.0, 0.0, 0.74, 1.0))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def plot_cv_subject_session_run_raw_3x1(metric_df, out_path):
    column_defs = _build_metric_column_dfs(metric_df=metric_df, metric_specs=METRIC_SPECS, use_zscore=False)

    column_limits = []
    for _, metric_col_df, _ in column_defs:
        combined_values = np.concatenate([metric_col_df["behavior_raw"], metric_col_df["projection_raw"]])
        finite_values = combined_values[np.isfinite(combined_values)]
        if finite_values.size == 0:
            column_limits.append((-1.0, 1.0))
            continue
        q_low = float(np.percentile(finite_values, 2.0))
        q_high = float(np.percentile(finite_values, 98.0))
        q_span = q_high - q_low
        pad = 0.18 * q_span if q_span > 0 else max(0.05 * abs(q_low), 1e-6)
        column_limits.append((q_low - pad, q_high + pad))

    row_defs = [("sub_tag", "Subject colors", "Subject"), ("ses", "Session colors", "Session"),
                ("run", "Run colors", "Run")]

    n_cols = len(column_defs)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(6.0 * n_cols, 12.0), 14.0))
    axes = _reshape_axes_grid(axes, n_rows=3, n_cols=n_cols)

    for row_idx, (group_col, row_title, legend_title) in enumerate(row_defs):
        for col_idx, (column_title, metric_col_df, metric_spec) in enumerate(column_defs):
            ax = axes[row_idx, col_idx]
            color_map, category_values = _build_category_color_map(metric_col_df[group_col].tolist(), category_name=group_col)
            _plot_paired_box_with_connections(ax=ax, paired_df=metric_col_df, group_col=group_col, color_map=color_map,
                jitter_seed=211 + 31 * row_idx + 19 * col_idx,
                y_limits=column_limits[col_idx],
                behavior_value_col="behavior_raw",
                projection_value_col="projection_raw",
                x_tick_labels=(f"Behavior {metric_spec['label']}", f"BOLD {metric_spec['label']}"))
            ax.set_ylim(column_limits[col_idx])
            ax.set_title(f"{row_title} | {column_title}", fontsize=11.2)
            if col_idx == 0:
                ax.set_ylabel("Value")
            if col_idx == (n_cols - 1):
                _add_category_legend(
                    ax=ax,
                    color_map=color_map,
                    category_values=category_values,
                    title=legend_title,
                )

    fig.suptitle("Behavior vs BOLD variability (no z-score)", fontsize=13.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.995, 0.975))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _validate_3x2_metric_df(metric_df):
    _validate_metric_df(metric_df, metric_specs=METRIC_SPECS)


def _build_3x2_column_metric_dfs(metric_df):
    return _build_metric_column_dfs(metric_df=metric_df, metric_specs=METRIC_SPECS, use_zscore=False)


def _sort_count_category_cell(cell_df, row_title, primary_subject_sort_col=None):
    cell_df = cell_df.copy()
    if cell_df.empty:
        return cell_df

    if (
        str(row_title) == "Subject colors"
        and primary_subject_sort_col is not None
        and primary_subject_sort_col in cell_df.columns
    ):
        subject_order = []
        for value in cell_df["category"].astype(str):
            try:
                subject_order.append(int(_extract_subject_digits(value)))
            except ValueError:
                subject_order.append(int(1e9))
        cell_df["_subject_order"] = subject_order
        cell_df = cell_df.sort_values(
            [primary_subject_sort_col, "_subject_order", "category"],
            ascending=[False, True, True],
        )
        return cell_df.drop(columns=["_subject_order"])

    cat_num = pd.to_numeric(cell_df["category"], errors="coerce")
    if np.all(np.isfinite(cat_num.to_numpy(dtype=np.float64))):
        cell_df["_cat_num"] = cat_num.to_numpy(dtype=np.float64)
        cell_df = cell_df.sort_values(["_cat_num", "category"], ascending=[True, True])
        return cell_df.drop(columns=["_cat_num"])

    return cell_df.sort_values("category")


def compute_outside_box_counts_behavior_vs_bold_3x2(metric_df):
    row_defs = [
        ("sub_tag", "Subject colors"),
        ("ses", "Session colors"),
        ("run", "Run colors"),
    ]
    column_defs = _build_3x2_column_metric_dfs(metric_df)

    rows = []
    for column_title, metric_col_df, metric_spec in column_defs:
        behavior_values = metric_col_df["behavior_raw"].to_numpy(dtype=np.float64)
        projection_values = metric_col_df["projection_raw"].to_numpy(dtype=np.float64)
        if behavior_values.size == 0 or projection_values.size == 0:
            continue

        bq1, bq3 = np.percentile(behavior_values, [25.0, 75.0])
        pq1, pq3 = np.percentile(projection_values, [25.0, 75.0])
        behavior_out = (behavior_values < bq1) | (behavior_values > bq3)
        projection_out = (projection_values < pq1) | (projection_values > pq3)

        for group_col, row_title in row_defs:
            group_values = metric_col_df[group_col].astype(str).to_numpy()
            categories = sorted(
                {value for value in group_values},
                key=lambda value: _category_sort_key(value, group_col),
            )
            for category in categories:
                category_mask = group_values == str(category)
                rows.append(
                    {
                        "row": row_title,
                        "column": column_title,
                        "metric_key": str(metric_spec["key"]),
                        "category": str(category),
                        "behavior_outside_box_count": int(
                            np.count_nonzero(behavior_out & category_mask)
                        ),
                        "bold_outside_box_count": int(
                            np.count_nonzero(projection_out & category_mask)
                        ),
                    }
                )

    return pd.DataFrame(rows)


def plot_outside_box_counts_behavior_vs_bold_3x2(
    metric_df,
    out_path,
    out_csv_path=None,
    sort_subject_by_behavior=True,
):
    counts_df = compute_outside_box_counts_behavior_vs_bold_3x2(metric_df)
    if out_csv_path is not None:
        counts_df.to_csv(out_csv_path, index=False)

    column_defs = _build_3x2_column_metric_dfs(metric_df)
    row_order = ["Subject colors", "Session colors", "Run colors"]
    column_order = [column_title for column_title, _, _ in column_defs]

    n_cols = len(column_order)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(6.5 * n_cols, 19.0), 14.0), sharey=False)
    axes = _reshape_axes_grid(axes, n_rows=3, n_cols=n_cols)
    for row_idx, row_title in enumerate(row_order):
        for col_idx, column_title in enumerate(column_order):
            ax = axes[row_idx, col_idx]
            cell = counts_df.loc[
                (counts_df["row"] == row_title) & (counts_df["column"] == column_title)
            ].copy()
            if cell.empty:
                ax.set_axis_off()
                continue

            sort_col = "behavior_outside_box_count" if bool(sort_subject_by_behavior) else None
            cell = _sort_count_category_cell(
                cell,
                row_title=row_title,
                primary_subject_sort_col=sort_col,
            )

            categories = cell["category"].astype(str).tolist()
            x = np.arange(len(categories), dtype=np.float64)
            width = 0.42
            behavior_counts = cell["behavior_outside_box_count"].to_numpy(dtype=np.float64)
            bold_counts = cell["bold_outside_box_count"].to_numpy(dtype=np.float64)

            ax.bar(
                x - (width / 2.0),
                behavior_counts,
                width=width,
                color="#1f77b4",
                label="Behavior",
            )
            ax.bar(
                x + (width / 2.0),
                bold_counts,
                width=width,
                color="#d62728",
                label="BOLD",
            )

            ax.set_title(f"{row_title} | {column_title}", fontsize=11.5)
            ax.set_ylabel("Outside-box count")
            ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
            ax.set_xticks(x)
            if row_title == "Subject colors":
                ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8.0)
            else:
                ax.set_xticklabels(categories, fontsize=9.5)

            ymax = float(np.max(np.concatenate([behavior_counts, bold_counts])))
            ax.set_ylim(0.0, max(3.0, ymax + 1.0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle("Outside-box counts (Q1-Q3): Behavior vs BOLD", fontsize=14.0, y=1.01)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return counts_df


def _classify_whisker_position(values, near_fraction=0.10):
    values = np.asarray(values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        raise RuntimeError("No finite values available for whisker classification.")

    q1, q3 = np.percentile(finite_values, [25.0, 75.0])
    iqr = float(q3 - q1)
    low_fence = float(q1 - (1.5 * iqr))
    high_fence = float(q3 + (1.5 * iqr))
    near_band = float(max(float(near_fraction) * iqr, 1e-12))

    below = values < low_fence
    above = values > high_fence
    near_low = (values >= low_fence) & (values <= (low_fence + near_band))
    near_high = (values <= high_fence) & (values >= (high_fence - near_band))
    near = (~below) & (~above) & (near_low | near_high)
    total = below | near | above

    return {
        "q1": float(q1),
        "q3": float(q3),
        "iqr": iqr,
        "low_fence": low_fence,
        "high_fence": high_fence,
        "near_band": near_band,
        "below": np.asarray(below, dtype=bool),
        "near": np.asarray(near, dtype=bool),
        "above": np.asarray(above, dtype=bool),
        "total": np.asarray(total, dtype=bool),
    }


def compute_whisker_outlier_counts_behavior_vs_bold_3x2(metric_df, near_fraction=0.10):
    near_fraction = float(near_fraction)
    if near_fraction < 0:
        raise ValueError(f"near_fraction must be >= 0, got {near_fraction}.")

    row_defs = [
        ("sub_tag", "Subject colors"),
        ("ses", "Session colors"),
        ("run", "Run colors"),
    ]
    column_defs = _build_3x2_column_metric_dfs(metric_df)

    rows = []
    thresholds = []
    for column_title, metric_col_df, metric_spec in column_defs:
        behavior_values = metric_col_df["behavior_raw"].to_numpy(dtype=np.float64)
        projection_values = metric_col_df["projection_raw"].to_numpy(dtype=np.float64)
        if behavior_values.size == 0 or projection_values.size == 0:
            continue
        behavior_cls = _classify_whisker_position(behavior_values, near_fraction=near_fraction)
        projection_cls = _classify_whisker_position(
            projection_values, near_fraction=near_fraction
        )

        thresholds.extend(
            [
                {
                    "column": column_title,
                    "metric_key": str(metric_spec["key"]),
                    "signal": "Behavior",
                    "q1": behavior_cls["q1"],
                    "q3": behavior_cls["q3"],
                    "iqr": behavior_cls["iqr"],
                    "low_fence": behavior_cls["low_fence"],
                    "high_fence": behavior_cls["high_fence"],
                    "near_band": behavior_cls["near_band"],
                },
                {
                    "column": column_title,
                    "metric_key": str(metric_spec["key"]),
                    "signal": "BOLD",
                    "q1": projection_cls["q1"],
                    "q3": projection_cls["q3"],
                    "iqr": projection_cls["iqr"],
                    "low_fence": projection_cls["low_fence"],
                    "high_fence": projection_cls["high_fence"],
                    "near_band": projection_cls["near_band"],
                },
            ]
        )

        for group_col, row_title in row_defs:
            group_values = metric_col_df[group_col].astype(str).to_numpy()
            categories = sorted(
                {value for value in group_values},
                key=lambda value: _category_sort_key(value, group_col),
            )
            for category in categories:
                category_mask = group_values == str(category)
                behavior_below = int(np.count_nonzero(behavior_cls["below"] & category_mask))
                behavior_near = int(np.count_nonzero(behavior_cls["near"] & category_mask))
                behavior_above = int(np.count_nonzero(behavior_cls["above"] & category_mask))

                bold_below = int(np.count_nonzero(projection_cls["below"] & category_mask))
                bold_near = int(np.count_nonzero(projection_cls["near"] & category_mask))
                bold_above = int(np.count_nonzero(projection_cls["above"] & category_mask))

                rows.append(
                    {
                        "row": row_title,
                        "column": column_title,
                        "metric_key": str(metric_spec["key"]),
                        "category": str(category),
                        "behavior_below_whisker_count": behavior_below,
                        "behavior_near_whisker_count": behavior_near,
                        "behavior_above_whisker_count": behavior_above,
                        "behavior_outlier_total": int(behavior_below + behavior_near + behavior_above),
                        "bold_below_whisker_count": bold_below,
                        "bold_near_whisker_count": bold_near,
                        "bold_above_whisker_count": bold_above,
                        "bold_outlier_total": int(bold_below + bold_near + bold_above),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def plot_whisker_outlier_counts_behavior_vs_bold_3x2(
    metric_df,
    out_path,
    out_csv_path=None,
    thresholds_csv_path=None,
    near_fraction=0.10,
    sort_subject_by_behavior=True,
):
    counts_df, thresholds_df = compute_whisker_outlier_counts_behavior_vs_bold_3x2(
        metric_df=metric_df,
        near_fraction=near_fraction,
    )
    if out_csv_path is not None:
        counts_df.to_csv(out_csv_path, index=False)
    if thresholds_csv_path is not None:
        thresholds_df.to_csv(thresholds_csv_path, index=False)

    column_defs = _build_3x2_column_metric_dfs(metric_df)
    row_order = ["Subject colors", "Session colors", "Run colors"]
    column_order = [column_title for column_title, _, _ in column_defs]

    n_cols = len(column_order)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(6.8 * n_cols, 20.0), 14.0), sharey=False)
    axes = _reshape_axes_grid(axes, n_rows=3, n_cols=n_cols)
    for row_idx, row_title in enumerate(row_order):
        for col_idx, column_title in enumerate(column_order):
            ax = axes[row_idx, col_idx]
            cell = counts_df.loc[
                (counts_df["row"] == row_title) & (counts_df["column"] == column_title)
            ].copy()
            if cell.empty:
                ax.set_axis_off()
                continue

            sort_col = "behavior_outlier_total" if bool(sort_subject_by_behavior) else None
            cell = _sort_count_category_cell(
                cell,
                row_title=row_title,
                primary_subject_sort_col=sort_col,
            )

            categories = cell["category"].astype(str).tolist()
            x = np.arange(len(categories), dtype=np.float64)
            width = 0.38

            behavior_below = cell["behavior_below_whisker_count"].to_numpy(dtype=np.float64)
            behavior_near = cell["behavior_near_whisker_count"].to_numpy(dtype=np.float64)
            behavior_above = cell["behavior_above_whisker_count"].to_numpy(dtype=np.float64)
            bold_below = cell["bold_below_whisker_count"].to_numpy(dtype=np.float64)
            bold_near = cell["bold_near_whisker_count"].to_numpy(dtype=np.float64)
            bold_above = cell["bold_above_whisker_count"].to_numpy(dtype=np.float64)

            ax.bar(
                x - (width / 2.0),
                behavior_below,
                width=width,
                color="#6baed6",
                label="Behavior below" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x - (width / 2.0),
                behavior_near,
                width=width,
                bottom=behavior_below,
                color="#3182bd",
                label="Behavior near" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x - (width / 2.0),
                behavior_above,
                width=width,
                bottom=(behavior_below + behavior_near),
                color="#08519c",
                label="Behavior above" if (row_idx, col_idx) == (0, 0) else None,
            )

            ax.bar(
                x + (width / 2.0),
                bold_below,
                width=width,
                color="#fcae91",
                label="BOLD below" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x + (width / 2.0),
                bold_near,
                width=width,
                bottom=bold_below,
                color="#fb6a4a",
                label="BOLD near" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x + (width / 2.0),
                bold_above,
                width=width,
                bottom=(bold_below + bold_near),
                color="#cb181d",
                label="BOLD above" if (row_idx, col_idx) == (0, 0) else None,
            )

            ax.set_title(f"{row_title} | {column_title}", fontsize=11.5)
            ax.set_ylabel("Whisker-based outlier count")
            ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
            ax.set_xticks(x)
            if row_title == "Subject colors":
                ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8.0)
            else:
                ax.set_xticklabels(categories, fontsize=9.5)

            behavior_total = behavior_below + behavior_near + behavior_above
            bold_total = bold_below + bold_near + bold_above
            ymax = float(np.max(np.concatenate([behavior_total, bold_total])))
            ax.set_ylim(0.0, max(2.0, ymax + 1.0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle(
        "Whisker outlier counts (below / near / above): Behavior vs BOLD",
        fontsize=14.0,
        y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return counts_df, thresholds_df


def compute_whisker_outlier_counts_behavior_vs_bold_cv_raw(metric_df, near_fraction=0.10):
    near_fraction = float(near_fraction)
    if near_fraction < 0:
        raise ValueError(f"near_fraction must be >= 0, got {near_fraction}.")

    row_defs = [
        ("sub_tag", "Subject colors"),
        ("ses", "Session colors"),
        ("run", "Run colors"),
    ]
    column_defs = _build_metric_column_dfs(metric_df=metric_df, metric_specs=METRIC_SPECS, use_zscore=False)

    rows = []
    thresholds = []
    for column_title, metric_col_df, metric_spec in column_defs:
        behavior_values = metric_col_df["behavior_raw"].to_numpy(dtype=np.float64)
        projection_values = metric_col_df["projection_raw"].to_numpy(dtype=np.float64)
        if behavior_values.size == 0 or projection_values.size == 0:
            continue
        behavior_cls = _classify_whisker_position(behavior_values, near_fraction=near_fraction)
        projection_cls = _classify_whisker_position(projection_values, near_fraction=near_fraction)

        thresholds.extend(
            [
                {
                    "column": column_title,
                    "metric_key": str(metric_spec["key"]),
                    "signal": "Behavior",
                    "q1": behavior_cls["q1"],
                    "q3": behavior_cls["q3"],
                    "iqr": behavior_cls["iqr"],
                    "low_fence": behavior_cls["low_fence"],
                    "high_fence": behavior_cls["high_fence"],
                    "near_band": behavior_cls["near_band"],
                },
                {
                    "column": column_title,
                    "metric_key": str(metric_spec["key"]),
                    "signal": "BOLD",
                    "q1": projection_cls["q1"],
                    "q3": projection_cls["q3"],
                    "iqr": projection_cls["iqr"],
                    "low_fence": projection_cls["low_fence"],
                    "high_fence": projection_cls["high_fence"],
                    "near_band": projection_cls["near_band"],
                },
            ]
        )

        for group_col, row_title in row_defs:
            group_values = metric_col_df[group_col].astype(str).to_numpy()
            categories = sorted(
                {value for value in group_values},
                key=lambda value: _category_sort_key(value, group_col),
            )
            for category in categories:
                category_mask = group_values == str(category)
                behavior_below = int(np.count_nonzero(behavior_cls["below"] & category_mask))
                behavior_near = int(np.count_nonzero(behavior_cls["near"] & category_mask))
                behavior_above = int(np.count_nonzero(behavior_cls["above"] & category_mask))

                bold_below = int(np.count_nonzero(projection_cls["below"] & category_mask))
                bold_near = int(np.count_nonzero(projection_cls["near"] & category_mask))
                bold_above = int(np.count_nonzero(projection_cls["above"] & category_mask))

                rows.append(
                    {
                        "row": row_title,
                        "column": column_title,
                        "metric_key": str(metric_spec["key"]),
                        "category": str(category),
                        "behavior_below_whisker_count": behavior_below,
                        "behavior_near_whisker_count": behavior_near,
                        "behavior_above_whisker_count": behavior_above,
                        "behavior_outlier_total": int(behavior_below + behavior_near + behavior_above),
                        "bold_below_whisker_count": bold_below,
                        "bold_near_whisker_count": bold_near,
                        "bold_above_whisker_count": bold_above,
                        "bold_outlier_total": int(bold_below + bold_near + bold_above),
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(thresholds)


def plot_whisker_outlier_counts_behavior_vs_bold_cv_raw(
    metric_df,
    out_path,
    out_csv_path=None,
    thresholds_csv_path=None,
    near_fraction=0.10,
    sort_subject_by_behavior=True,
):
    counts_df, thresholds_df = compute_whisker_outlier_counts_behavior_vs_bold_cv_raw(
        metric_df=metric_df,
        near_fraction=near_fraction,
    )
    if out_csv_path is not None:
        counts_df.to_csv(out_csv_path, index=False)
    if thresholds_csv_path is not None:
        thresholds_df.to_csv(thresholds_csv_path, index=False)

    column_defs = _build_metric_column_dfs(metric_df=metric_df, metric_specs=METRIC_SPECS, use_zscore=False)
    row_order = ["Subject colors", "Session colors", "Run colors"]
    column_order = [column_title for column_title, _, _ in column_defs]

    n_cols = len(column_order)
    fig, axes = plt.subplots(3, n_cols, figsize=(max(6.8 * n_cols, 20.0), 14.0), sharey=False)
    axes = _reshape_axes_grid(axes, n_rows=3, n_cols=n_cols)

    for row_idx, row_title in enumerate(row_order):
        for col_idx, column_title in enumerate(column_order):
            ax = axes[row_idx, col_idx]
            cell = counts_df.loc[
                (counts_df["row"] == row_title) & (counts_df["column"] == column_title)
            ].copy()
            if cell.empty:
                ax.set_axis_off()
                continue

            sort_col = "behavior_outlier_total" if bool(sort_subject_by_behavior) else None
            cell = _sort_count_category_cell(
                cell,
                row_title=row_title,
                primary_subject_sort_col=sort_col,
            )

            categories = cell["category"].astype(str).tolist()
            x = np.arange(len(categories), dtype=np.float64)
            width = 0.38

            behavior_below = cell["behavior_below_whisker_count"].to_numpy(dtype=np.float64)
            behavior_near = cell["behavior_near_whisker_count"].to_numpy(dtype=np.float64)
            behavior_above = cell["behavior_above_whisker_count"].to_numpy(dtype=np.float64)
            bold_below = cell["bold_below_whisker_count"].to_numpy(dtype=np.float64)
            bold_near = cell["bold_near_whisker_count"].to_numpy(dtype=np.float64)
            bold_above = cell["bold_above_whisker_count"].to_numpy(dtype=np.float64)

            ax.bar(
                x - (width / 2.0),
                behavior_below,
                width=width,
                color="#6baed6",
                label="Behavior below" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x - (width / 2.0),
                behavior_near,
                width=width,
                bottom=behavior_below,
                color="#3182bd",
                label="Behavior near" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x - (width / 2.0),
                behavior_above,
                width=width,
                bottom=(behavior_below + behavior_near),
                color="#08519c",
                label="Behavior above" if (row_idx, col_idx) == (0, 0) else None,
            )

            ax.bar(
                x + (width / 2.0),
                bold_below,
                width=width,
                color="#fcae91",
                label="BOLD below" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x + (width / 2.0),
                bold_near,
                width=width,
                bottom=bold_below,
                color="#fb6a4a",
                label="BOLD near" if (row_idx, col_idx) == (0, 0) else None,
            )
            ax.bar(
                x + (width / 2.0),
                bold_above,
                width=width,
                bottom=(bold_below + bold_near),
                color="#cb181d",
                label="BOLD above" if (row_idx, col_idx) == (0, 0) else None,
            )

            ax.set_title(f"{row_title} | {column_title}", fontsize=11.5)
            ax.set_ylabel("Whisker-based outlier count")
            ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.55)
            ax.set_xticks(x)
            if row_title == "Subject colors":
                ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8.0)
            else:
                ax.set_xticklabels(categories, fontsize=9.5)

            behavior_total = behavior_below + behavior_near + behavior_above
            bold_total = bold_below + bold_near + bold_above
            ymax = float(np.max(np.concatenate([behavior_total, bold_total])))
            ax.set_ylim(0.0, max(2.0, ymax + 1.0))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
            bbox_to_anchor=(0.5, 0.995),
        )
    fig.suptitle(
        "Whisker outlier counts for metrics (below / near / above): Behavior vs BOLD",
        fontsize=14.0,
        y=1.01,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return counts_df, thresholds_df


def _plot_subject_metric_comparison(
    subject_metrics_df,
    projection_col,
    behavior_col,
    metric_label,
    out_path,
    outlier_iqr_multiplier=3.0,
    projection_plot_scale=1.0,
    grid_points=512,
    use_2x2_layout=False,
    z_text_projection_col=None,
    z_text_behavior_col=None,
):
    projection_values = subject_metrics_df[projection_col].to_numpy(dtype=np.float64)
    behavior_values = subject_metrics_df[behavior_col].to_numpy(dtype=np.float64)
    subject_labels = _build_pair_labels(subject_metrics_df)

    finite_proj = projection_values[np.isfinite(projection_values)]
    finite_beh = behavior_values[np.isfinite(behavior_values)]
    if finite_proj.size == 0:
        raise RuntimeError(f"No finite subject-level projection {metric_label} values.")
    if finite_beh.size == 0:
        raise RuntimeError(f"No finite subject-level behavior {metric_label} values.")

    paired = _paired_outlier_filtered(
        projection_values,
        behavior_values,
        labels=subject_labels,
        iqr_multiplier=outlier_iqr_multiplier,
    )
    if paired["n_pairs_kept"] == 0:
        raise RuntimeError(
            f"No paired subject-level {metric_label} values after outlier filtering."
        )

    tests = _paired_tests_two_sided(paired["paired_a_kept"], paired["paired_b_kept"])

    base_cols = [col for col in ("sub_tag", "ses", "run") if col in subject_metrics_df.columns]
    if base_cols:
        paired_table = subject_metrics_df.loc[paired["input_paired_mask"], base_cols].copy()
        paired_table = paired_table.reset_index(drop=True)
    else:
        paired_table = pd.DataFrame(index=np.arange(paired["paired_a_all"].size))
    paired_table["pair_label"] = paired["paired_labels_all"]
    paired_table[projection_col] = paired["paired_a_all"]
    paired_table[behavior_col] = paired["paired_b_all"]
    paired_table["removed_outlier"] = paired["paired_outlier_mask"]

    projection_plot_values = finite_proj * float(projection_plot_scale)
    x_proj = _density_grid(projection_plot_values, grid_points=grid_points, fallback_pad=1e-6)
    proj_density = _evaluate_density(projection_plot_values, x_proj)

    x_beh = _density_grid(finite_beh, grid_points=grid_points, fallback_pad=1e-6)
    beh_density = _evaluate_density(finite_beh, x_beh)

    proj_paired = paired["paired_a_kept"]
    beh_paired = paired["paired_b_kept"]
    raw_all = np.concatenate([proj_paired, beh_paired])
    x_raw = _density_grid(raw_all, grid_points=grid_points, fallback_pad=1e-6)
    proj_raw_density = _evaluate_density(proj_paired, x_raw)
    beh_raw_density = _evaluate_density(beh_paired, x_raw)

    if bool(use_2x2_layout):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
        ax0, ax1, ax2, ax3 = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))
        ax0, ax1, ax2 = axes
        ax3 = None

    ax0.plot(x_proj, proj_density, color="tab:blue", linewidth=2.0)
    ax0.fill_between(x_proj, proj_density, alpha=0.2, color="tab:blue")
    proj_xlabel = f"Projection {metric_label}"
    if not np.isclose(projection_plot_scale, 1.0):
        proj_xlabel += f" (x{projection_plot_scale:.0e})"
    ax0.set_xlabel(proj_xlabel)
    ax0.set_ylabel("Probability density")
    ax0.set_title(f"1) Projection {metric_label}")

    ax1.plot(x_beh, beh_density, color="tab:orange", linewidth=2.0)
    ax1.fill_between(x_beh, beh_density, alpha=0.2, color="tab:orange")
    ax1.set_xlabel(f"Behavior {metric_label}")
    ax1.set_ylabel("Probability density")
    ax1.set_title(f"2) Behavior {metric_label}")

    ax2.plot(
        x_raw,
        proj_raw_density,
        color="tab:blue",
        linewidth=2.0,
        label=f"Projection {metric_label}",
    )
    ax2.fill_between(x_raw, proj_raw_density, alpha=0.15, color="tab:blue")
    ax2.plot(
        x_raw,
        beh_raw_density,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label=f"Behavior {metric_label}",
    )
    ax2.fill_between(x_raw, beh_raw_density, alpha=0.15, color="tab:orange")
    ax2.set_xlabel(f"{metric_label}")
    ax2.set_ylabel("Probability density")
    ax2.set_title(f"3) {metric_label} + paired tests")
    ax2.legend(fontsize=8, loc="upper right")

    removed_labels = paired_table.loc[paired_table["removed_outlier"], "pair_label"].astype(str).tolist()

    stats_text = (
        f"paired t p(two-sided)={tests['ttest_p_two_sided']:.3g}, t={tests['ttest_stat']:.3g}\n"
        f"Wilcoxon p(two-sided)={tests['wilcoxon_p_two_sided']:.3g}, W={tests['wilcoxon_stat']:.3g}"
    )
    ax2.text(
        0.98,
        0.52,
        stats_text,
        transform=ax2.transAxes,
        va="center",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.7"},
    )

    if ax3 is not None:
        ax3.scatter(
            beh_paired,
            proj_paired,
            s=36,
            alpha=0.85,
            color="tab:green",
            edgecolors="none",
        )
        raw_min = float(np.min(np.concatenate([beh_paired, proj_paired])))
        raw_max = float(np.max(np.concatenate([beh_paired, proj_paired])))
        raw_span = raw_max - raw_min
        raw_pad = 0.06 * raw_span if raw_span > 0 else 0.05
        line_min = raw_min - raw_pad
        line_max = raw_max + raw_pad
        ax3.plot(
            [line_min, line_max],
            [line_min, line_max],
            color="0.25",
            linestyle="--",
            linewidth=1.2,
        )
        ax3.set_xlim(line_min, line_max)
        ax3.set_ylim(line_min, line_max)
        ax3.set_xlabel(f"Behavior {metric_label}")
        ax3.set_ylabel(f"Projection {metric_label}")
        ax3.set_title(f"4) {metric_label} scatter")

        text_projection_col = z_text_projection_col or projection_col
        text_behavior_col = z_text_behavior_col or behavior_col
        text_diff_mean = np.nan
        text_diff_median = np.nan
        if (
            text_projection_col in subject_metrics_df.columns
            and text_behavior_col in subject_metrics_df.columns
        ):
            text_paired = _paired_outlier_filtered(
                subject_metrics_df[text_projection_col].to_numpy(dtype=np.float64),
                subject_metrics_df[text_behavior_col].to_numpy(dtype=np.float64),
                labels=subject_labels,
                iqr_multiplier=outlier_iqr_multiplier,
            )
            if text_paired["n_pairs_kept"] > 0:
                text_diff = text_paired["paired_a_kept"] - text_paired["paired_b_kept"]
                text_diff_mean = float(np.mean(text_diff))
                text_diff_median = float(np.median(text_diff))

        if np.isfinite(text_diff_mean):
            z_diff_text = (
                f"mean({text_projection_col}-\n{text_behavior_col}) = {text_diff_mean:.3g}\n"
                f"median = {text_diff_median:.3g}"
            )
        else:
            z_diff_text = (
                f"mean({text_projection_col}-\n{text_behavior_col}) = NaN"
            )
        ax3.text(
            0.98,
            0.06,
            z_diff_text,
            transform=ax3.transAxes,
            va="bottom",
            ha="right",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "0.7"},
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    summary_row = {
        "metric": str(metric_label),
        "n_pairs_total": int(paired["n_pairs_total"]),
        "n_pairs_removed_outlier": int(paired["n_pairs_removed"]),
        "n_pairs_used": int(paired["n_pairs_kept"]),
        "removed_subjects": ";".join(removed_labels),
        "mean_diff_projection_minus_behavior": float(tests["mean_diff_a_minus_b"]),
        "median_diff_projection_minus_behavior": float(tests["median_diff_a_minus_b"]),
        "ttest_stat": float(tests["ttest_stat"]),
        "ttest_p_two_sided": float(tests["ttest_p_two_sided"]),
        "wilcoxon_stat": float(tests["wilcoxon_stat"]),
        "wilcoxon_p_two_sided": float(tests["wilcoxon_p_two_sided"]),
    }
    return summary_row, paired_table


def plot_scaled_variance_comparison_density(
    subject_metrics_df,
    out_path,
    grid_points=512,
    outlier_iqr_multiplier=3.0,
):
    metric_spec = next(spec for spec in METRIC_SPECS if str(spec["key"]) == "variance")
    return _plot_subject_metric_comparison(
        subject_metrics_df=subject_metrics_df,
        projection_col=str(metric_spec["projection_col"]),
        behavior_col=str(metric_spec["behavior_col"]),
        metric_label=str(metric_spec["label"]),
        out_path=out_path,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        projection_plot_scale=float(metric_spec.get("projection_plot_scale", 1.0)),
        grid_points=grid_points,
        use_2x2_layout=True,
    )


def plot_sub_ses_run_cv_comparison(
    subject_metrics_df,
    out_path,
    grid_points=512,
    outlier_iqr_multiplier=3.0,
):
    metric_spec = next(spec for spec in METRIC_SPECS if str(spec["key"]) == "cv")
    return _plot_subject_metric_comparison(
        subject_metrics_df=subject_metrics_df,
        projection_col=str(metric_spec["projection_col"]),
        behavior_col=str(metric_spec["behavior_col"]),
        metric_label=str(metric_spec["label"]),
        out_path=out_path,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        projection_plot_scale=float(metric_spec.get("projection_plot_scale", 1.0)),
        grid_points=grid_points,
        use_2x2_layout=True,
    )


def analyze_subject_metric_difference(
    metric_df,
    projection_col,
    behavior_col,
    metric_label,
    out_path,
    outlier_iqr_multiplier=3.0,
    n_permutations=20000,
    n_bootstrap=10000,
    bootstrap_ci_percent=95.0,
    random_seed=0,
    grid_points=512,
):
    required_cols = {"sub_tag", "ses", "run", projection_col, behavior_col}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for subject metric difference analysis: {sorted(missing_cols)}"
        )

    metric_projection = metric_df[projection_col].to_numpy(dtype=np.float64)
    metric_behavior = metric_df[behavior_col].to_numpy(dtype=np.float64)
    subject_labels = _build_pair_labels(metric_df)

    paired = _paired_outlier_filtered(
        metric_projection,
        metric_behavior,
        labels=subject_labels,
        iqr_multiplier=outlier_iqr_multiplier,
    )
    if paired["n_pairs_kept"] == 0:
        raise RuntimeError(
            f"No paired subject values for {metric_label} after outlier filtering."
        )

    d_s = paired["paired_a_kept"] - paired["paired_b_kept"]
    d_mean = float(np.mean(d_s))
    d_median = float(np.median(d_s))

    perm_observed_mean, perm_p_two_sided = _paired_sign_flip_permutation_test(
        d_s,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    mean_ci_low, mean_ci_high = _bootstrap_ci(
        d_s,
        statistic="mean",
        n_bootstrap=n_bootstrap,
        ci_percent=bootstrap_ci_percent,
        random_seed=random_seed,
    )
    median_ci_low, median_ci_high = _bootstrap_ci(
        d_s,
        statistic="median",
        n_bootstrap=n_bootstrap,
        ci_percent=bootstrap_ci_percent,
        random_seed=int(random_seed) + 1,
    )

    paired_metric_df = pd.DataFrame(
        index=np.arange(paired["paired_a_all"].size)
    )
    base_cols = [col for col in ("sub_tag", "ses", "run") if col in metric_df.columns]
    if base_cols:
        paired_metric_df = metric_df.loc[paired["input_paired_mask"], base_cols].copy()
        paired_metric_df = paired_metric_df.reset_index(drop=True)
    paired_metric_df["pair_label"] = paired["paired_labels_all"]
    paired_metric_df[projection_col] = paired["paired_a_all"]
    paired_metric_df[behavior_col] = paired["paired_b_all"]
    paired_metric_df["removed_outlier"] = paired["paired_outlier_mask"]
    paired_metric_df["d_s_projection_minus_behavior"] = (
        paired_metric_df[projection_col] - paired_metric_df[behavior_col]
    )
    paired_metric_df.loc[paired_metric_df["removed_outlier"], "d_s_projection_minus_behavior"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ax1, ax2 = axes

    proj_kept = paired["paired_a_kept"]
    beh_kept = paired["paired_b_kept"]
    if proj_kept.size == 0 or beh_kept.size == 0:
        raise RuntimeError(
            f"No finite paired {metric_label} values after outlier filtering."
        )

    x_ds = _density_grid(d_s, grid_points=grid_points, fallback_pad=1e-6)
    ds_density = _evaluate_density(d_s, x_ds)
    ax1.plot(x_ds, ds_density, color="tab:green", linewidth=2.0)
    ax1.fill_between(x_ds, ds_density, alpha=0.2, color="tab:green")
    ax1.axvline(0.0, color="0.35", linewidth=1.0, linestyle=":")
    ax1.axvline(d_mean, color="black", linewidth=1.2, linestyle="-", label="mean")
    ax1.axvline(d_median, color="black", linewidth=1.2, linestyle="--", label="median")
    ax1.set_xlabel(f"d_s = projection {metric_label} - behavior {metric_label}")
    ax1.set_ylabel("Probability density")
    ax1.set_title("1) d_s distribution")
    ax1.legend(fontsize=8, loc="upper right")

    ax2.scatter(
        beh_kept,
        proj_kept,
        s=36,
        alpha=0.85,
        color="tab:blue",
        edgecolors="none",
    )
    scatter_x_kept = beh_kept
    scatter_y_kept = proj_kept
    xy_min = float(np.min(np.concatenate([scatter_x_kept, scatter_y_kept])))
    xy_max = float(np.max(np.concatenate([scatter_x_kept, scatter_y_kept])))
    xy_span = xy_max - xy_min
    pad = 0.06 * xy_span if xy_span > 0 else 0.05
    line_min = xy_min - pad
    line_max = xy_max + pad
    ax2.plot(
        [line_min, line_max],
        [line_min, line_max],
        color="0.25",
        linestyle="--",
        linewidth=1.2,
    )
    ax2.set_xlim(line_min, line_max)
    ax2.set_ylim(line_min, line_max)
    ax2.set_xlabel(f"Behavior {metric_label}")
    ax2.set_ylabel(f"Projection {metric_label}")
    ax2.set_title(f"2) Paired scatter ({metric_label})")

    removed_labels = paired_metric_df.loc[paired_metric_df["removed_outlier"], "pair_label"].astype(str).tolist()
    stats_text = (
        f"mean(d_s)={d_mean:.3g}, median(d_s)={d_median:.3g}\n"
        f"Sign-flip permutation p(two-sided)={perm_p_two_sided:.3g}\n"
        f"{bootstrap_ci_percent:.0f}% CI mean=[{mean_ci_low:.3g}, {mean_ci_high:.3g}]\n"
        f"{bootstrap_ci_percent:.0f}% CI median=[{median_ci_low:.3g}, {median_ci_high:.3g}]"
    )
    ax1.text(
        0.98,
        0.50,
        stats_text,
        transform=ax1.transAxes,
        va="center",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    stats_row = {
        "metric": str(metric_label),
        "projection_col": str(projection_col),
        "behavior_col": str(behavior_col),
        "n_pairs_total": int(paired["n_pairs_total"]),
        "n_pairs_removed_outlier": int(paired["n_pairs_removed"]),
        "n_pairs_used": int(paired["n_pairs_kept"]),
        "removed_subjects": ";".join(removed_labels),
        "mean_d_s_projection_minus_behavior": d_mean,
        "median_d_s_projection_minus_behavior": d_median,
        "perm_observed_mean": float(perm_observed_mean),
        "perm_p_two_sided": float(perm_p_two_sided),
        "bootstrap_ci_percent": float(bootstrap_ci_percent),
        "bootstrap_mean_ci_low": float(mean_ci_low),
        "bootstrap_mean_ci_high": float(mean_ci_high),
        "bootstrap_median_ci_low": float(median_ci_low),
        "bootstrap_median_ci_high": float(median_ci_high),
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
        "random_seed": int(random_seed),
    }
    return stats_row, paired_metric_df


def analyze_subject_cv_difference(
    metric_df,
    out_path,
    outlier_iqr_multiplier=3.0,
    n_permutations=20000,
    n_bootstrap=10000,
    bootstrap_ci_percent=95.0,
    random_seed=0,
    grid_points=512,
):
    metric_spec = next(spec for spec in METRIC_SPECS if str(spec["key"]) == "cv")
    return analyze_subject_metric_difference(
        metric_df=metric_df,
        projection_col=str(metric_spec["projection_col"]),
        behavior_col=str(metric_spec["behavior_col"]),
        metric_label=str(metric_spec["label"]),
        out_path=out_path,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        bootstrap_ci_percent=bootstrap_ci_percent,
        random_seed=random_seed,
        grid_points=grid_points,
    )


def _has_finite_metric_pairs(metric_df, projection_col, behavior_col):
    projection = np.asarray(metric_df[projection_col], dtype=np.float64)
    behavior = np.asarray(metric_df[behavior_col], dtype=np.float64)
    paired_mask = np.isfinite(projection) & np.isfinite(behavior)
    return bool(np.any(paired_mask))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-path", default="results/behave_vs_bold/projection_voxel_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_thr90.npy",
        help="Path to projection vector (.npy).")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH,
        help="Path to concat manifest TSV (default: group manifest).")
    parser.add_argument("--trial-keep-root", default=DEFAULT_TRIAL_KEEP_ROOT,
        help="Root containing trial_keep_run*.npy files.")
    parser.add_argument("--behavior-root", default=DEFAULT_BEHAVIOR_ROOT,
        help="Root containing PSPD*_ses_*_run_*.npy behavior files.")
    parser.add_argument("--behavior-column", type=int, default=1,
        help="Behavior column index (0-based). Use 1 for the second column.")
    parser.add_argument(
        "--scale-method",
        default="zscore",
        choices=["zscore", "minmax"],
        help="Scaling method used to compare projection and behavior variances.",
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=3.0,
        help="IQR multiplier used to remove paired outlier rows (sub/ses/run) before paired tests.",
    )
    parser.add_argument(
        "--permutation-iterations",
        type=int,
        default=20000,
        help="Number of sign-flip permutations used for d_s permutation test.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for d_s CI estimation.",
    )
    parser.add_argument(
        "--bootstrap-ci-percent",
        type=float,
        default=95.0,
        help="Bootstrap CI level in percent (e.g., 95).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for permutation and bootstrap procedures.",
    )
    parser.add_argument(
        "--whisker-near-fraction",
        type=float,
        default=0.10,
        help="Near-whisker band as a fraction of IQR for whisker outlier summary plots.",
    )
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    args = parser.parse_args()

    projection_path = os.path.abspath(os.path.expanduser(args.projection_path))
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest_path))
    trial_keep_root = os.path.abspath(os.path.expanduser(args.trial_keep_root))
    behavior_root = os.path.abspath(os.path.expanduser(args.behavior_root))

    projection = np.asarray(np.load(projection_path)).ravel()
    manifest_df = pd.read_csv(manifest_path, sep="\t")

    run_segments, _ = split_projection_by_run(projection, manifest_df, trial_keep_root)
    run_df, behavior_df, run_metric_df = compute_run_behavior_tables(
        run_segments,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
    )

    run_pair_outlier_tables = []
    excluded_run_keys = set()
    run_outlier_specs = _metric_specs_from_keys(RUN_OUTLIER_METRIC_KEYS)
    for metric_spec in run_outlier_specs:
        metric_projection_col = str(metric_spec["projection_col"])
        metric_behavior_col = str(metric_spec["behavior_col"])
        if not _has_finite_metric_pairs(
            run_metric_df,
            projection_col=metric_projection_col,
            behavior_col=metric_behavior_col,
        ):
            print(
                f"Skipping run outlier detection for {metric_spec['label']} (no finite paired values)."
            )
            continue
        metric_outlier_keys, metric_pair_outlier_df = identify_run_outliers(
            run_metric_df,
            projection_col=metric_projection_col,
            behavior_col=metric_behavior_col,
            outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
        )
        metric_pair_outlier_df = metric_pair_outlier_df.copy()
        metric_pair_outlier_df["metric_key"] = str(metric_spec["key"])
        metric_pair_outlier_df["metric_label"] = str(metric_spec["label"])
        run_pair_outlier_tables.append(metric_pair_outlier_df)
        excluded_run_keys.update(metric_outlier_keys)

    if run_pair_outlier_tables:
        run_pair_outlier_df = pd.concat(run_pair_outlier_tables, axis=0, ignore_index=True)
    else:
        run_pair_outlier_df = pd.DataFrame(
            columns=[
                "sub_tag",
                "ses",
                "run",
                "pair_label",
                "removed_outlier",
                "metric_key",
                "metric_label",
            ]
        )

    subject_metrics_df = compute_subject_projection_behavior_metrics(
        run_segments,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
        excluded_run_keys=excluded_run_keys,
    )
    comparison_df = build_projection_behavior_comparison(
        run_df, behavior_df, scale_method=args.scale_method
    )

    metric_merge_cols = ["sub_tag", "ses", "run", "n_trials_paired_finite"]
    for metric_spec in METRIC_SPECS:
        for col in (str(metric_spec["projection_col"]), str(metric_spec["behavior_col"])):
            if col not in {"variance_projection", "variance_behavior_col2"}:
                metric_merge_cols.append(col)
    seen_cols = set()
    unique_metric_merge_cols = []
    for col in metric_merge_cols:
        if col not in seen_cols:
            unique_metric_merge_cols.append(col)
            seen_cols.add(col)

    comparison_df = comparison_df.merge(
        run_metric_df[unique_metric_merge_cols],
        on=["sub_tag", "ses", "run"],
        how="left",
        validate="one_to_one",
    )
    comparison_df = add_subject_median_normalized_variance_features(comparison_df)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(projection_path) or os.getcwd()
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(projection_path))[0]
    output_paths = {
        "run_csv": os.path.join(out_dir, f"{stem}_sub_ses_run_variance.csv"),
        "behavior_csv": os.path.join(out_dir, f"{stem}_sub_ses_run_behavior_variance.csv"),
        "compare_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_variance.csv"
        ),
        "run_metric_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_metrics.csv"
        ),
        "subject_metrics_csv": os.path.join(
            out_dir, f"{stem}_subject_projection_behavior_metrics.csv"
        ),
        "run_density_plot": os.path.join(out_dir, f"{stem}_sub_ses_run_variance_density.png"),
        "metric_grid_plot": os.path.join(
            out_dir, f"{stem}_projection_behavior_all_metrics_raw_grid.png"
        ),
        "metric_grid_first_row_second_col_plot": os.path.join(
            out_dir, f"{stem}_projection_behavior_grid_row1_col2.png"
        ),
        "outside_box_counts_csv": os.path.join(
            out_dir,
            f"{stem}_all_metrics_outside_box_counts_behavior_vs_bold_all_categories.csv",
        ),
        "outside_box_counts_plot": os.path.join(
            out_dir,
            (
                f"{stem}_all_metrics_outside_box_counts_behavior_vs_bold_all_categories_"
                "subject_row_sorted_by_behavior.png"
            ),
        ),
        "whisker_outlier_counts_csv": os.path.join(
            out_dir,
            f"{stem}_all_metrics_whisker_outlier_counts_behavior_vs_bold_all_categories.csv",
        ),
        "whisker_outlier_thresholds_csv": os.path.join(
            out_dir,
            f"{stem}_all_metrics_whisker_outlier_thresholds.csv",
        ),
        "whisker_outlier_counts_plot": os.path.join(
            out_dir,
            (
                f"{stem}_all_metrics_whisker_outlier_counts_behavior_vs_bold_all_categories_"
                "subject_row_sorted_by_behavior.png"
            ),
        ),
        "raw_whisker_outlier_counts_csv": os.path.join(
            out_dir,
            f"{stem}_all_metrics_raw_whisker_outlier_counts_behavior_vs_bold_all_categories.csv",
        ),
        "raw_whisker_outlier_thresholds_csv": os.path.join(
            out_dir,
            f"{stem}_all_metrics_raw_whisker_outlier_thresholds.csv",
        ),
        "raw_whisker_outlier_counts_plot": os.path.join(
            out_dir,
            (
                f"{stem}_all_metrics_raw_whisker_outlier_counts_behavior_vs_bold_all_categories_"
                "subject_row_sorted_by_behavior.png"
            ),
        ),
        "paired_stats_csv": os.path.join(
            out_dir, f"{stem}_subject_projection_behavior_paired_stats.csv"
        ),
        "metric_ds_stats_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_metric_ds_stats.csv"
        ),
        "run_pairs_outlier_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_all_metric_pairs_outlier_flags.csv"
        ),
    }
    metric_path_map = {}
    for metric_spec in METRIC_SPECS:
        metric_key = str(metric_spec["key"])
        metric_stub = str(metric_spec["file_stub"])
        metric_path_map[metric_key] = {
            "comparison_plot": os.path.join(
                out_dir, f"{stem}_sub_ses_run_projection_behavior_{metric_stub}_density.png"
            ),
            "pairs_csv": os.path.join(
                out_dir, f"{stem}_sub_ses_run_projection_behavior_{metric_stub}_pairs.csv"
            ),
            "ds_plot": os.path.join(
                out_dir, f"{stem}_projection_behavior_{metric_stub}_ds_analysis.png"
            ),
            "ds_pairs_csv": os.path.join(
                out_dir, f"{stem}_sub_ses_run_projection_behavior_{metric_stub}_ds_pairs.csv"
            ),
        }

    run_df.to_csv(output_paths["run_csv"], index=False)
    behavior_df.to_csv(output_paths["behavior_csv"], index=False)
    comparison_df.to_csv(output_paths["compare_csv"], index=False)
    run_metric_df.to_csv(output_paths["run_metric_csv"], index=False)
    subject_metrics_df.to_csv(output_paths["subject_metrics_csv"], index=False)
    run_pair_outlier_df.to_csv(output_paths["run_pairs_outlier_csv"], index=False)

    plot_variance_cv_subject_session_run_3x2(
        comparison_df,
        output_paths["metric_grid_plot"],
    )
    plot_metric_grid_first_row_second_column(
        comparison_df,
        output_paths["metric_grid_first_row_second_col_plot"],
    )
    plot_outside_box_counts_behavior_vs_bold_3x2(
        comparison_df,
        output_paths["outside_box_counts_plot"],
        out_csv_path=output_paths["outside_box_counts_csv"],
        sort_subject_by_behavior=True,
    )
    plot_whisker_outlier_counts_behavior_vs_bold_3x2(
        comparison_df,
        output_paths["whisker_outlier_counts_plot"],
        out_csv_path=output_paths["whisker_outlier_counts_csv"],
        thresholds_csv_path=output_paths["whisker_outlier_thresholds_csv"],
        near_fraction=float(args.whisker_near_fraction),
        sort_subject_by_behavior=True,
    )
    plot_whisker_outlier_counts_behavior_vs_bold_cv_raw(
        run_metric_df,
        output_paths["raw_whisker_outlier_counts_plot"],
        out_csv_path=output_paths["raw_whisker_outlier_counts_csv"],
        thresholds_csv_path=output_paths["raw_whisker_outlier_thresholds_csv"],
        near_fraction=float(args.whisker_near_fraction),
        sort_subject_by_behavior=True,
    )
    plot_run_variance_density(run_df, output_paths["run_density_plot"])

    paired_stats_rows = []
    ds_stats_rows = []
    for metric_spec in METRIC_SPECS:
        metric_key = str(metric_spec["key"])
        metric_projection_col = str(metric_spec["projection_col"])
        metric_behavior_col = str(metric_spec["behavior_col"])
        if not _has_finite_metric_pairs(
            run_metric_df,
            projection_col=metric_projection_col,
            behavior_col=metric_behavior_col,
        ):
            print(
                f"Skipping metric {metric_spec['label']} (no finite paired values after preprocessing)."
            )
            continue
        metric_paths = metric_path_map[metric_key]

        summary_row, metric_pairs_df = _plot_subject_metric_comparison(
            subject_metrics_df=run_metric_df,
            projection_col=metric_projection_col,
            behavior_col=metric_behavior_col,
            metric_label=str(metric_spec["label"]),
            out_path=metric_paths["comparison_plot"],
            outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
            projection_plot_scale=float(metric_spec.get("projection_plot_scale", 1.0)),
            grid_points=512,
            use_2x2_layout=True,
        )
        summary_row["metric_key"] = metric_key
        paired_stats_rows.append(summary_row)
        metric_pairs_df.to_csv(metric_paths["pairs_csv"], index=False)

        ds_stats_row, ds_pairs_df = analyze_subject_metric_difference(
            metric_df=run_metric_df,
            projection_col=metric_projection_col,
            behavior_col=metric_behavior_col,
            metric_label=str(metric_spec["label"]),
            out_path=metric_paths["ds_plot"],
            outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
            n_permutations=int(args.permutation_iterations),
            n_bootstrap=int(args.bootstrap_iterations),
            bootstrap_ci_percent=float(args.bootstrap_ci_percent),
            random_seed=int(args.random_seed),
            grid_points=512,
        )
        ds_stats_row["metric_key"] = metric_key
        ds_stats_rows.append(ds_stats_row)
        ds_pairs_df.to_csv(metric_paths["ds_pairs_csv"], index=False)

    paired_stats_df = pd.DataFrame(paired_stats_rows)
    paired_stats_df.to_csv(output_paths["paired_stats_csv"], index=False)
    pd.DataFrame(ds_stats_rows).to_csv(output_paths["metric_ds_stats_csv"], index=False)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
