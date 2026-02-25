import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, ttest_rel, wilcoxon

DEFAULT_MANIFEST_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv"
DEFAULT_TRIAL_KEEP_ROOT = "/Data/zahra/results_glm"
DEFAULT_BEHAVIOR_ROOT = "/Data/zahra/behaviour"


def _resolve_trial_keep_path(row, trial_keep_root):
    from_manifest = str(getattr(row, "trial_keep_path", "") or "").strip()
    if from_manifest and os.path.exists(from_manifest):
        return from_manifest

    candidate = os.path.join(
        trial_keep_root,
        str(row.sub_tag),
        f"ses-{int(row.ses)}",
        "GLMOutputs-mni-std",
        f"trial_keep_run{int(row.run)}.npy",
    )
    if os.path.exists(candidate):
        return candidate

    raise FileNotFoundError(
        f"trial_keep file not found for {row.sub_tag}, ses-{row.ses}, run-{row.run}. "
        f"Checked: '{from_manifest}' and '{candidate}'"
    )


def _load_keep_mask(path):
    keep = np.load(path)
    keep = np.asarray(keep, dtype=bool)
    if keep.ndim != 1:
        raise ValueError(f"Expected 1D keep mask in {path}, got shape {keep.shape}.")
    return keep


def _extract_subject_digits(sub_tag):
    match = re.search(r"(\d+)$", str(sub_tag))
    if not match:
        raise ValueError(f"Could not parse subject digits from '{sub_tag}'.")
    return match.group(1)


def _resolve_behavior_path(sub_tag, ses, run, behavior_root):
    subject_digits = _extract_subject_digits(sub_tag)
    behavior_path = os.path.join(
        behavior_root,
        f"PSPD{subject_digits}_ses_{int(ses)}_run_{int(run)}.npy",
    )
    if not os.path.exists(behavior_path):
        raise FileNotFoundError(
            f"Missing behavior file for {sub_tag}, ses-{ses}, run-{run}: {behavior_path}"
        )
    return behavior_path


def _load_behavior_column(path, behavior_column):
    behavior = np.asarray(np.load(path), dtype=np.float64)
    if behavior.ndim == 1:
        if behavior_column != 0:
            raise IndexError(
                f"Behavior file {path} is 1D; requested column {behavior_column}."
            )
        return behavior
    if behavior.ndim != 2:
        raise ValueError(
            f"Behavior file {path} must be 1D or 2D, got shape {behavior.shape}."
        )
    if not (0 <= int(behavior_column) < int(behavior.shape[1])):
        raise IndexError(
            f"Behavior column {behavior_column} is out of bounds for {path} "
            f"(shape={behavior.shape})."
        )
    return behavior[:, int(behavior_column)]


def _load_run_kept_projection_behavior(segment, behavior_root, behavior_column):
    behavior_path = _resolve_behavior_path(
        segment["sub_tag"], segment["ses"], segment["run"], behavior_root
    )
    behavior_column_values = _load_behavior_column(behavior_path, behavior_column)

    n_trials_source = int(segment["n_trials_source"])
    if behavior_column_values.size < n_trials_source:
        raise ValueError(
            f"Behavior file {behavior_path} has {behavior_column_values.size} trials, "
            f"expected at least {n_trials_source}."
        )
    if behavior_column_values.size > n_trials_source:
        behavior_column_values = behavior_column_values[:n_trials_source]

    keep_mask = np.asarray(segment["keep_mask"], dtype=bool)
    if keep_mask.size != n_trials_source:
        raise ValueError(
            f"Keep mask length mismatch for {segment['sub_tag']} ses-{segment['ses']} "
            f"run-{segment['run']} ({keep_mask.size} vs {n_trials_source})."
        )

    kept_behavior = behavior_column_values[keep_mask]
    kept_projection = np.asarray(segment["values"], dtype=np.float64)
    if kept_projection.size != kept_behavior.size:
        raise ValueError(
            f"Kept projection/behavior length mismatch for {segment['sub_tag']} "
            f"ses-{segment['ses']} run-{segment['run']} "
            f"({kept_projection.size} vs {kept_behavior.size})."
        )
    return kept_projection, kept_behavior, behavior_path


def _prepare_run_entries(manifest_df, trial_keep_root):
    run_entries = []
    rows = manifest_df.sort_values("offset_start").itertuples(index=False)
    for row in rows:
        keep_path = _resolve_trial_keep_path(row, trial_keep_root)
        keep_mask = _load_keep_mask(keep_path)
        run_entries.append(
            {
                "sub_tag": str(row.sub_tag),
                "ses": int(row.ses),
                "run": int(row.run),
                "keep_path": keep_path,
                "keep_mask": keep_mask,
                "keep_count": int(np.count_nonzero(keep_mask)),
                "source_count": int(keep_mask.size),
            }
        )
    return run_entries


def _infer_projection_layout(projection_len, run_entries):
    total_kept = int(sum(entry["keep_count"] for entry in run_entries))
    total_source = int(sum(entry["source_count"] for entry in run_entries))

    if projection_len == total_kept:
        return "kept_only"
    if projection_len == total_source:
        return "source_all"

    raise ValueError(
        "Projection length does not match keep-trial bookkeeping. "
        f"projection_len={projection_len}, total_kept={total_kept}, total_source={total_source}."
    )


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

        run_segments.append(
            {
                "sub_tag": entry["sub_tag"],
                "ses": entry["ses"],
                "run": entry["run"],
                "n_trials_source": entry["source_count"],
                "n_trials_kept": int(run_values.size),
                "keep_mask": entry["keep_mask"].copy(),
                "values": np.asarray(run_values, dtype=projection.dtype),
            }
        )

    if cursor != projection.size:
        raise RuntimeError(f"Consumed {cursor} elements but projection has {projection.size}.")

    return run_segments, layout


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
        rows.append(
            {
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_source": int(segment["n_trials_source"]),
                "n_trials_kept": int(proj_finite.size),
                "mean_projection": proj_mean,
                "variance_projection": proj_var,
            }
        )
    return pd.DataFrame(rows)


def compute_run_behavior_tables(run_segments, behavior_root, behavior_column):
    run_rows = []
    behavior_rows = []
    cv_rows = []

    for segment in run_segments:
        kept_projection, kept_behavior, behavior_path = _load_run_kept_projection_behavior(
            segment, behavior_root, behavior_column
        )
        finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
        projection_values = np.asarray(kept_projection[finite_mask], dtype=np.float64)
        behavior_values = np.asarray(kept_behavior[finite_mask], dtype=np.float64)

        if projection_values.size == 0:
            proj_mean = np.nan
            proj_var = np.nan
        else:
            proj_mean = float(np.mean(projection_values))
            proj_var = float(np.var(projection_values))
        run_rows.append(
            {
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_source": int(segment["n_trials_source"]),
                "n_trials_kept": int(projection_values.size),
                "mean_projection": proj_mean,
                "variance_projection": proj_var,
            }
        )

        if behavior_values.size == 0:
            beh_mean = np.nan
            beh_var = np.nan
        else:
            beh_mean = float(np.mean(behavior_values))
            beh_var = float(np.var(behavior_values))
        behavior_rows.append(
            {
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_behavior_source": int(segment["n_trials_source"]),
                "n_trials_behavior_kept": int(behavior_values.size),
                "mean_behavior_col2": beh_mean,
                "variance_behavior_col2": beh_var,
                "behavior_path": behavior_path,
            }
        )

        proj_mean, proj_std, proj_cv = _coefficient_of_variation(projection_values)
        beh_mean, beh_std, beh_cv = _coefficient_of_variation(behavior_values)

        cv_rows.append(
            {
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_paired_finite": int(projection_values.size),
                "mean_projection": proj_mean,
                "std_projection": proj_std,
                "cv_projection": proj_cv,
                "mean_behavior_col2": beh_mean,
                "std_behavior_col2": beh_std,
                "cv_behavior_col2": beh_cv,
            }
        )

    return (
        pd.DataFrame(run_rows),
        pd.DataFrame(behavior_rows),
        pd.DataFrame(cv_rows),
    )


def _scale_values(values, method):
    values = np.asarray(values, dtype=np.float64)
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
    raise ValueError(f"Unsupported scaling method '{method}'.")


def build_projection_behavior_comparison(run_df, behavior_df, scale_method):
    comparison_df = run_df.merge(
        behavior_df,
        on=["sub_tag", "ses", "run"],
        how="inner",
        validate="one_to_one",
    )
    comparison_df["variance_projection_scaled"] = _scale_values(
        comparison_df["variance_projection"].to_numpy(dtype=np.float64),
        method=scale_method,
    )
    comparison_df["variance_behavior_scaled"] = _scale_values(
        comparison_df["variance_behavior_col2"].to_numpy(dtype=np.float64),
        method=scale_method,
    )
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

    df["subject_median_variance_projection"] = df.groupby("sub_tag")[
        "variance_projection"
    ].transform(lambda s: _finite_median(s.to_numpy(dtype=np.float64)))
    df["subject_median_variance_behavior_col2"] = df.groupby("sub_tag")[
        "variance_behavior_col2"
    ].transform(lambda s: _finite_median(s.to_numpy(dtype=np.float64)))

    proj = df["variance_projection"].to_numpy(dtype=np.float64)
    beh = df["variance_behavior_col2"].to_numpy(dtype=np.float64)
    proj_med = df["subject_median_variance_projection"].to_numpy(dtype=np.float64)
    beh_med = df["subject_median_variance_behavior_col2"].to_numpy(dtype=np.float64)

    proj_rel = np.full(proj.shape, np.nan, dtype=np.float64)
    beh_rel = np.full(beh.shape, np.nan, dtype=np.float64)

    proj_rel_mask = np.isfinite(proj) & np.isfinite(proj_med) & (proj_med > 0.0)
    beh_rel_mask = np.isfinite(beh) & np.isfinite(beh_med) & (beh_med > 0.0)
    proj_rel[proj_rel_mask] = proj[proj_rel_mask] / proj_med[proj_rel_mask]
    beh_rel[beh_rel_mask] = beh[beh_rel_mask] / beh_med[beh_rel_mask]

    proj_log_rel = np.full(proj.shape, np.nan, dtype=np.float64)
    beh_log_rel = np.full(beh.shape, np.nan, dtype=np.float64)
    proj_log_mask = np.isfinite(proj_rel) & (proj_rel > 0.0)
    beh_log_mask = np.isfinite(beh_rel) & (beh_rel > 0.0)
    proj_log_rel[proj_log_mask] = np.log(proj_rel[proj_log_mask])
    beh_log_rel[beh_log_mask] = np.log(beh_rel[beh_log_mask])

    log_rel_diff = np.full(proj.shape, np.nan, dtype=np.float64)
    paired_log_mask = np.isfinite(proj_log_rel) & np.isfinite(beh_log_rel)
    log_rel_diff[paired_log_mask] = proj_log_rel[paired_log_mask] - beh_log_rel[paired_log_mask]

    df["variance_projection_rel_subject_median"] = proj_rel
    df["variance_behavior_rel_subject_median"] = beh_rel
    df["log_variance_projection_rel_subject_median"] = proj_log_rel
    df["log_variance_behavior_rel_subject_median"] = beh_log_rel
    df["log_rel_diff_proj_minus_beh"] = log_rel_diff
    return df


def _one_sided_p_less_from_two_sided(two_sided_p, estimate):
    if not np.isfinite(two_sided_p) or not np.isfinite(estimate):
        return np.nan
    if estimate < 0:
        return float(two_sided_p / 2.0)
    return float(1.0 - (two_sided_p / 2.0))


def _iqr_outlier_mask(values, iqr_multiplier=3.0):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    q1, q3 = np.percentile(values, [25.0, 75.0])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0.0:
        return np.zeros(values.shape, dtype=bool)

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
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"Paired arrays shape mismatch: {values_a.shape} vs {values_b.shape}."
        )

    if labels is None:
        labels = np.arange(values_a.size).astype(str)
    else:
        labels = np.asarray(labels).astype(str)
        if labels.shape != values_a.shape:
            raise ValueError(
                f"Paired labels shape mismatch: {labels.shape} vs {values_a.shape}."
            )

    paired_mask = np.isfinite(values_a) & np.isfinite(values_b)
    paired_a = values_a[paired_mask]
    paired_b = values_b[paired_mask]
    paired_labels = labels[paired_mask]

    if iqr_multiplier is None:
        out_any = np.zeros(paired_a.shape, dtype=bool)
    else:
        out_a = _iqr_outlier_mask(paired_a, iqr_multiplier=iqr_multiplier)
        out_b = _iqr_outlier_mask(paired_b, iqr_multiplier=iqr_multiplier)
        out_any = out_a | out_b
    keep_mask = ~out_any

    return {
        "input_paired_mask": paired_mask,
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
        "n_pairs_kept": int(np.count_nonzero(keep_mask)),
    }


def _paired_tests_two_sided(values_a, values_b):
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    if values_a.shape != values_b.shape:
        raise ValueError(
            f"Paired arrays shape mismatch: {values_a.shape} vs {values_b.shape}."
        )

    row = {
        "n_pairs": int(values_a.size),
        "mean_diff_a_minus_b": np.nan,
        "median_diff_a_minus_b": np.nan,
        "ttest_stat": np.nan,
        "ttest_p_two_sided": np.nan,
        "wilcoxon_stat": np.nan,
        "wilcoxon_p_two_sided": np.nan,
    }
    if values_a.size == 0:
        return row

    diff = values_a - values_b
    row["mean_diff_a_minus_b"] = float(np.mean(diff))
    row["median_diff_a_minus_b"] = float(np.median(diff))

    if values_a.size >= 2:
        try:
            t_res = ttest_rel(values_a, values_b, nan_policy="omit")
        except TypeError:
            t_res = ttest_rel(values_a, values_b)
        row["ttest_stat"] = float(t_res.statistic)
        row["ttest_p_two_sided"] = float(t_res.pvalue)

    if np.allclose(diff, 0.0):
        row["wilcoxon_stat"] = 0.0
        row["wilcoxon_p_two_sided"] = 1.0
    else:
        try:
            w_res = wilcoxon(values_a, values_b, alternative="two-sided")
            row["wilcoxon_stat"] = float(w_res.statistic)
            row["wilcoxon_p_two_sided"] = float(w_res.pvalue)
        except ValueError:
            pass
    return row


def _bootstrap_ci(values, statistic="mean", n_bootstrap=10000, ci_percent=95.0, random_seed=0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan

    ci_percent = float(ci_percent)
    if not (0.0 < ci_percent < 100.0):
        raise ValueError(f"ci_percent must be in (0, 100), got {ci_percent}.")

    n_bootstrap = int(n_bootstrap)
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be > 0, got {n_bootstrap}.")

    rng = np.random.default_rng(int(random_seed))
    sample_idx = rng.integers(0, values.size, size=(n_bootstrap, values.size))
    sampled = values[sample_idx]

    statistic = str(statistic).lower()
    if statistic == "mean":
        boot_stats = np.mean(sampled, axis=1)
    elif statistic == "median":
        boot_stats = np.median(sampled, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic '{statistic}'.")

    alpha = 100.0 - ci_percent
    lower_q = alpha / 2.0
    upper_q = 100.0 - lower_q
    return (
        float(np.percentile(boot_stats, lower_q)),
        float(np.percentile(boot_stats, upper_q)),
    )


def _paired_sign_flip_permutation_test(diff_values, n_permutations=20000, random_seed=0):
    diff_values = np.asarray(diff_values, dtype=np.float64)
    diff_values = diff_values[np.isfinite(diff_values)]
    if diff_values.size == 0:
        return np.nan, np.nan

    observed = float(np.mean(diff_values))
    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        raise ValueError(f"n_permutations must be > 0, got {n_permutations}.")

    rng = np.random.default_rng(int(random_seed))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, diff_values.size))
    permuted_means = np.mean(signs * diff_values[None, :], axis=1)
    p_two_sided = (
        np.count_nonzero(np.abs(permuted_means) >= abs(observed)) + 1.0
    ) / (n_permutations + 1.0)
    return observed, float(p_two_sided)


def compute_subject_projection_behavior_metrics(
    run_segments,
    behavior_root,
    behavior_column,
    excluded_run_keys=None,
):
    excluded_keys = set(excluded_run_keys or [])
    rows = []
    subject_keys = sorted(
        {(str(segment["sub_tag"])) for segment in run_segments},
        key=lambda sub: int(_extract_subject_digits(sub)),
    )

    for sub_tag in subject_keys:
        subject_segments = [
            segment for segment in run_segments if str(segment["sub_tag"]) == str(sub_tag)
        ]
        projection_chunks = []
        behavior_chunks = []
        n_runs = 0
        n_trials_paired = 0
        for segment in subject_segments:
            run_key = (
                str(segment["sub_tag"]),
                int(segment["ses"]),
                int(segment["run"]),
            )
            if run_key in excluded_keys:
                continue

            kept_projection, kept_behavior, _ = _load_run_kept_projection_behavior(
                segment, behavior_root, behavior_column
            )
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

        if proj_values.size == 0:
            proj_mean = np.nan
            proj_var = np.nan
        else:
            proj_mean = float(np.mean(proj_values))
            proj_var = float(np.var(proj_values))
        if beh_values.size == 0:
            beh_mean = np.nan
            beh_var = np.nan
        else:
            beh_mean = float(np.mean(beh_values))
            beh_var = float(np.var(beh_values))

        proj_mean_cv, proj_std, proj_cv = _coefficient_of_variation(proj_values)
        beh_mean_cv, beh_std, beh_cv = _coefficient_of_variation(beh_values)

        rows.append(
            {
                "sub_tag": str(sub_tag),
                "n_runs_with_paired_trials": int(n_runs),
                "n_trials_paired_finite": int(n_trials_paired),
                "mean_projection": proj_mean,
                "variance_projection": proj_var,
                "std_projection": proj_std,
                "cv_projection": proj_cv,
                "mean_behavior_col2": beh_mean,
                "variance_behavior_col2": beh_var,
                "std_behavior_col2": beh_std,
                "cv_behavior_col2": beh_cv,
                "d_var_projection_minus_behavior": (
                    float(proj_var - beh_var)
                    if np.isfinite(proj_var) and np.isfinite(beh_var)
                    else np.nan
                ),
                "d_cv_projection_minus_behavior": (
                    float(proj_cv - beh_cv)
                    if np.isfinite(proj_cv) and np.isfinite(beh_cv)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def identify_run_outliers(metric_df, projection_col, behavior_col, outlier_iqr_multiplier=3.0):
    required_cols = {"sub_tag", "ses", "run", projection_col, behavior_col}
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for run outlier identification: {sorted(missing_cols)}"
        )

    labels = _build_pair_labels(metric_df)
    paired = _paired_outlier_filtered(
        metric_df[projection_col].to_numpy(dtype=np.float64),
        metric_df[behavior_col].to_numpy(dtype=np.float64),
        labels=labels,
        iqr_multiplier=outlier_iqr_multiplier,
    )

    paired_rows = metric_df.loc[paired["input_paired_mask"], ["sub_tag", "ses", "run"]].copy()
    paired_rows[projection_col] = paired["paired_a_all"]
    paired_rows[behavior_col] = paired["paired_b_all"]
    paired_rows["pair_label"] = paired["paired_labels_all"]
    paired_rows["removed_outlier"] = paired["paired_outlier_mask"]

    outlier_rows = paired_rows.loc[paired_rows["removed_outlier"]].copy()
    outlier_keys = {
        (str(row.sub_tag), int(row.ses), int(row.run))
        for row in outlier_rows.itertuples(index=False)
    }
    return outlier_keys, paired_rows


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


def _density_grid(values, grid_points=512, pad_fraction=0.1, fallback_pad=0.25):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("No finite values available for density-grid construction.")

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
    if values.size == 0:
        raise RuntimeError("No finite run/session variances available to plot.")

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    vrange = vmax - vmin

    if values.size < 2 or np.allclose(values, values[0]):
        width = max(abs(vmin) * 0.05, 1e-12)
        x = np.linspace(max(0.0, vmin - 4.0 * width), vmax + 4.0 * width, int(grid_points))
        density = (
            np.exp(-0.5 * ((x - vmin) / width) ** 2)
            / (width * np.sqrt(2.0 * np.pi))
        )
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
        try:
            return (0, int(_extract_subject_digits(value)))
        except ValueError:
            return (1, str(value))
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _build_category_color_map(values, category_name):
    unique_values = sorted(
        {value for value in values},
        key=lambda value: _category_sort_key(value, category_name),
    )
    n_values = len(unique_values)
    if n_values == 0:
        return {}, []

    if str(category_name) == "run":
        run_colors = [
            "#1f77b4",  # blue
            "#d62728",  # red
            "#2ca02c",  # green
            "#ff7f0e",  # orange
            "#9467bd",  # purple
            "#17becf",  # cyan
        ]
        color_map = {
            value: run_colors[idx % len(run_colors)]
            for idx, value in enumerate(unique_values)
        }
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
        color_map = {
            value: session_colors[idx % len(session_colors)]
            for idx, value in enumerate(unique_values)
        }
        return color_map, unique_values

    if str(category_name) == "sub_tag":
        # Build a larger qualitative palette to minimize subject-color collisions.
        tab20 = list(plt.get_cmap("tab20").colors)
        tab20b = list(plt.get_cmap("tab20b").colors)
        tab20c = list(plt.get_cmap("tab20c").colors)
        subject_palette = tab20 + tab20b + tab20c
        if n_values <= len(subject_palette):
            step = 11  # coprime with 60, spreads adjacent picks across the palette
            spread_palette = [
                subject_palette[(idx * step) % len(subject_palette)]
                for idx in range(len(subject_palette))
            ]
            color_map = {
                value: spread_palette[idx]
                for idx, value in enumerate(unique_values)
            }
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
            f"Missing required columns for paired z-score plot: {sorted(missing_cols)}"
        )

    projection_z = _scale_values(
        metric_df[projection_col].to_numpy(dtype=np.float64),
        method="zscore",
    )
    behavior_z = _scale_values(
        metric_df[behavior_col].to_numpy(dtype=np.float64),
        method="zscore",
    )
    paired_mask = np.isfinite(projection_z) & np.isfinite(behavior_z)
    if not np.any(paired_mask):
        raise RuntimeError(
            f"No paired finite values for {projection_col} and {behavior_col}."
        )

    paired_df = metric_df.loc[paired_mask, ["sub_tag", "ses", "run"]].copy()
    paired_df = paired_df.reset_index(drop=True)
    paired_df["projection_z"] = projection_z[paired_mask]
    paired_df["behavior_z"] = behavior_z[paired_mask]
    return paired_df


def _plot_paired_box_with_connections(
    ax,
    paired_df,
    group_col,
    color_map,
    jitter_seed=0,
    y_limits=None,
):
    behavior_values = paired_df["behavior_z"].to_numpy(dtype=np.float64)
    projection_values = paired_df["projection_z"].to_numpy(dtype=np.float64)
    group_values = paired_df[group_col].tolist()

    box_parts = ax.boxplot(
        [behavior_values, projection_values],
        positions=[0.0, 1.0],
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "black",
            "markeredgecolor": "black",
            "markersize": 4.0,
        },
        zorder=1,
    )
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

    for x0, x1, y0, y1, group_value in zip(
        x_behavior,
        x_projection,
        behavior_values,
        projection_values,
        group_values,
    ):
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

        ax.plot(
            [x0, x1],
            [y0_plot, y1_plot],
            color=color,
            linewidth=0.85,
            alpha=0.28,
            zorder=2,
        )
        ax.scatter(
            [x0],
            [y0_plot],
            s=34 if marker0 != "o" else 30,
            color=color,
            alpha=0.9,
            edgecolors="0.15",
            linewidths=0.3,
            marker=marker0,
            zorder=3,
        )
        ax.scatter(
            [x1],
            [y1_plot],
            s=34 if marker1 != "o" else 30,
            color=color,
            alpha=0.9,
            edgecolors="0.15",
            linewidths=0.3,
            marker=marker1,
            zorder=3,
        )

    ax.set_xlim(-0.45, 1.45)
    ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels(["Behavior variability", "BOLD variability"])
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)
    return int(clipped_count)


def _add_category_legend(ax, color_map, category_values, title):
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
    if not handles:
        return

    ncol = 1
    if len(handles) > 16:
        ncol = 2

    legend = ax.legend(
        handles=handles,
        title=str(title),
        loc="upper right",
        fontsize=7.5,
        title_fontsize=8.5,
        frameon=True,
        ncol=ncol,
        borderaxespad=0.4,
        handletextpad=0.35,
        columnspacing=0.8,
        labelspacing=0.25,
    )
    legend.get_frame().set_alpha(0.95)


def plot_variance_cv_subject_session_run_3x2(metric_df, out_path):
    required_cols = {
        "sub_tag",
        "ses",
        "run",
        "variance_projection",
        "variance_behavior_col2",
        "cv_projection",
        "cv_behavior_col2",
    }
    missing_cols = required_cols.difference(metric_df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for 3x2 variance/CV plot: {sorted(missing_cols)}"
        )

    variance_df = _build_paired_metric_zscore_df(
        metric_df=metric_df,
        projection_col="variance_projection",
        behavior_col="variance_behavior_col2",
    )
    cv_df = _build_paired_metric_zscore_df(
        metric_df=metric_df,
        projection_col="cv_projection",
        behavior_col="cv_behavior_col2",
    )

    column_defs = [
        ("Z-scored variance", variance_df),
        ("Z-scored CV", cv_df),
    ]
    row_defs = [
        ("sub_tag", "Subject colors", "Subject"),
        ("ses", "Session colors", "Session"),
        ("run", "Run colors", "Run"),
    ]

    column_limits = []
    for _, metric_col_df in column_defs:
        column_values = np.concatenate(
            [
                metric_col_df["behavior_z"].to_numpy(dtype=np.float64),
                metric_col_df["projection_z"].to_numpy(dtype=np.float64),
            ]
        )
        finite_values = column_values[np.isfinite(column_values)]
        if finite_values.size == 0:
            column_limits.append((-1.0, 1.0))
            continue
        q_low = float(np.percentile(finite_values, 2.0))
        q_high = float(np.percentile(finite_values, 98.0))
        q_span = q_high - q_low
        pad = 0.18 * q_span if q_span > 0 else 0.55
        y_low = min(q_low - pad, -2.8)
        y_high = max(q_high + pad, 2.8)
        column_limits.append((y_low, y_high))

    fig, axes = plt.subplots(3, 2, figsize=(17.8, 14.2))
    total_clipped = 0
    for row_idx, (group_col, row_title, legend_title) in enumerate(row_defs):
        for col_idx, (column_title, metric_col_df) in enumerate(column_defs):
            ax = axes[row_idx, col_idx]
            color_map, category_values = _build_category_color_map(
                metric_col_df[group_col].tolist(),
                category_name=group_col,
            )
            clipped_here = _plot_paired_box_with_connections(
                ax=ax,
                paired_df=metric_col_df,
                group_col=group_col,
                color_map=color_map,
                jitter_seed=111 + 17 * row_idx + 29 * col_idx,
                y_limits=column_limits[col_idx],
            )
            total_clipped += int(clipped_here)
            ax.set_ylim(column_limits[col_idx])
            ax.set_title(f"{row_title} | {column_title}", fontsize=11.0)
            if col_idx == 0:
                ax.set_ylabel("Variability")
            if col_idx == 1:
                _add_category_legend(
                    ax=ax,
                    color_map=color_map,
                    category_values=category_values,
                    title=legend_title,
                )

    fig.suptitle(
        "Behavior vs BOLD variability",
        fontsize=13.0,
    )
    tight_bottom = 0.0
    # if total_clipped > 0:
    #     fig.text(
    #         0.5,
    #         0.006,
    #         "Triangle markers denote values clipped by y-axis limits for readability.",
    #         ha="center",
    #         va="bottom",
    #         fontsize=9.0,
    #     )
        # tight_bottom = 0.02
    fig.tight_layout(rect=(0.0, tight_bottom, 0.995, 0.965))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


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

    proj_z = _scale_values(paired["paired_a_kept"], method="zscore")
    beh_z = _scale_values(paired["paired_b_kept"], method="zscore")
    z_all = np.concatenate([proj_z, beh_z])
    x_z = _density_grid(z_all, grid_points=grid_points, fallback_pad=1e-6)
    proj_z_density = _evaluate_density(proj_z, x_z)
    beh_z_density = _evaluate_density(beh_z, x_z)

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
        x_z,
        proj_z_density,
        color="tab:blue",
        linewidth=2.0,
        label=f"Projection {metric_label} (z)",
    )
    ax2.fill_between(x_z, proj_z_density, alpha=0.15, color="tab:blue")
    ax2.plot(
        x_z,
        beh_z_density,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label=f"Behavior {metric_label} (z)",
    )
    ax2.fill_between(x_z, beh_z_density, alpha=0.15, color="tab:orange")
    ax2.axvline(0.0, color="0.35", linewidth=1.0, linestyle=":")
    ax2.set_xlabel(f"Z-scored {metric_label} (within metric)")
    ax2.set_ylabel("Probability density")
    ax2.set_title(f"3) Z-scored {metric_label} + paired tests")
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
            beh_z,
            proj_z,
            s=36,
            alpha=0.85,
            color="tab:green",
            edgecolors="none",
        )
        z_min = float(np.min(np.concatenate([beh_z, proj_z])))
        z_max = float(np.max(np.concatenate([beh_z, proj_z])))
        z_span = z_max - z_min
        z_pad = 0.06 * z_span if z_span > 0 else 0.05
        line_min = z_min - z_pad
        line_max = z_max + z_pad
        ax3.plot(
            [line_min, line_max],
            [line_min, line_max],
            color="0.25",
            linestyle="--",
            linewidth=1.2,
        )
        ax3.set_xlim(line_min, line_max)
        ax3.set_ylim(line_min, line_max)
        ax3.set_xlabel(f"Z-scored behavior {metric_label}")
        ax3.set_ylabel(f"Z-scored projection {metric_label}")
        ax3.set_title(f"4) Z-scored {metric_label} scatter")

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
    return _plot_subject_metric_comparison(
        subject_metrics_df=subject_metrics_df,
        projection_col="variance_projection",
        behavior_col="variance_behavior_col2",
        metric_label="variance",
        out_path=out_path,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        projection_plot_scale=1e7,
        grid_points=grid_points,
        use_2x2_layout=True,
        z_text_projection_col="cv_projection",
        z_text_behavior_col="cv_behavior_col2",
    )


def plot_sub_ses_run_cv_comparison(
    subject_metrics_df,
    out_path,
    grid_points=512,
    outlier_iqr_multiplier=3.0,
):
    return _plot_subject_metric_comparison(
        subject_metrics_df=subject_metrics_df,
        projection_col="cv_projection",
        behavior_col="cv_behavior_col2",
        metric_label="CV",
        out_path=out_path,
        outlier_iqr_multiplier=outlier_iqr_multiplier,
        projection_plot_scale=1.0,
        grid_points=grid_points,
        use_2x2_layout=True,
    )


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
    cv_projection = metric_df["cv_projection"].to_numpy(dtype=np.float64)
    cv_behavior = metric_df["cv_behavior_col2"].to_numpy(dtype=np.float64)
    subject_labels = _build_pair_labels(metric_df)

    paired = _paired_outlier_filtered(
        cv_projection,
        cv_behavior,
        labels=subject_labels,
        iqr_multiplier=outlier_iqr_multiplier,
    )
    if paired["n_pairs_kept"] == 0:
        raise RuntimeError("No paired subject CV values after outlier filtering.")

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

    paired_cv_df = pd.DataFrame(
        index=np.arange(paired["paired_a_all"].size)
    )
    base_cols = [col for col in ("sub_tag", "ses", "run") if col in metric_df.columns]
    if base_cols:
        paired_cv_df = metric_df.loc[paired["input_paired_mask"], base_cols].copy()
        paired_cv_df = paired_cv_df.reset_index(drop=True)
    paired_cv_df["pair_label"] = paired["paired_labels_all"]
    paired_cv_df["cv_projection"] = paired["paired_a_all"]
    paired_cv_df["cv_behavior_col2"] = paired["paired_b_all"]
    paired_cv_df["removed_outlier"] = paired["paired_outlier_mask"]
    paired_cv_df["d_s_projection_minus_behavior"] = (
        paired_cv_df["cv_projection"] - paired_cv_df["cv_behavior_col2"]
    )
    paired_cv_df.loc[paired_cv_df["removed_outlier"], "d_s_projection_minus_behavior"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    ax1, ax2 = axes

    proj_cv_kept = paired["paired_a_kept"]
    beh_cv_kept = paired["paired_b_kept"]
    if proj_cv_kept.size == 0 or beh_cv_kept.size == 0:
        raise RuntimeError("No finite paired CV values after outlier filtering.")

    x_ds = _density_grid(d_s, grid_points=grid_points, fallback_pad=1e-6)
    ds_density = _evaluate_density(d_s, x_ds)
    ax1.plot(x_ds, ds_density, color="tab:green", linewidth=2.0)
    ax1.fill_between(x_ds, ds_density, alpha=0.2, color="tab:green")
    ax1.axvline(0.0, color="0.35", linewidth=1.0, linestyle=":")
    ax1.axvline(d_mean, color="black", linewidth=1.2, linestyle="-", label="mean")
    ax1.axvline(d_median, color="black", linewidth=1.2, linestyle="--", label="median")
    ax1.set_xlabel("d_s = CV projection - CV behavior")
    ax1.set_ylabel("Probability density")
    ax1.set_title("1) d_s distribution")
    ax1.legend(fontsize=8, loc="upper right")

    ax2.scatter(
        beh_cv_kept,
        proj_cv_kept,
        s=36,
        alpha=0.85,
        color="tab:blue",
        edgecolors="none",
    )
    scatter_x_kept = beh_cv_kept
    scatter_y_kept = proj_cv_kept
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
    ax2.set_xlabel("CV behavior")
    ax2.set_ylabel("CV projection")
    ax2.set_title("2) Paired scatter (raw CV)")

    removed_labels = paired_cv_df.loc[paired_cv_df["removed_outlier"], "pair_label"].astype(str).tolist()
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
    return stats_row, paired_cv_df


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
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    args = parser.parse_args()

    projection_path = os.path.abspath(os.path.expanduser(args.projection_path))
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest_path))
    trial_keep_root = os.path.abspath(os.path.expanduser(args.trial_keep_root))
    behavior_root = os.path.abspath(os.path.expanduser(args.behavior_root))

    projection = np.asarray(np.load(projection_path)).ravel()
    manifest_df = pd.read_csv(manifest_path, sep="\t")

    run_segments, _ = split_projection_by_run(projection, manifest_df, trial_keep_root)
    run_df, behavior_df, run_cv_df = compute_run_behavior_tables(
        run_segments,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
    )
    cv_run_outlier_keys, cv_run_pair_outlier_df = identify_run_outliers(
        run_cv_df,
        projection_col="cv_projection",
        behavior_col="cv_behavior_col2",
        outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
    )
    subject_metrics_df = compute_subject_projection_behavior_metrics(
        run_segments,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
        excluded_run_keys=cv_run_outlier_keys,
    )
    comparison_df = build_projection_behavior_comparison(
        run_df, behavior_df, scale_method=args.scale_method
    )
    comparison_df = comparison_df.merge(
        run_cv_df[
            [
                "sub_tag",
                "ses",
                "run",
                "cv_projection",
                "cv_behavior_col2",
                "n_trials_paired_finite",
            ]
        ],
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
        "subject_metrics_csv": os.path.join(
            out_dir, f"{stem}_subject_projection_behavior_metrics.csv"
        ),
        "run_density_plot": os.path.join(out_dir, f"{stem}_sub_ses_run_variance_density.png"),
        "compare_density_plot": os.path.join(
            out_dir, f"{stem}_projection_behavior_variance_scaled_density.png"
        ),
        "run_cv_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_cv.csv"
        ),
        "run_cv_plot": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_cv_density.png"
        ),
        "variance_cv_3x2_plot": os.path.join(
            out_dir, f"{stem}_projection_behavior_variance_cv_zscore_3x2.png"
        ),
        "paired_stats_csv": os.path.join(
            out_dir, f"{stem}_subject_projection_behavior_paired_stats.csv"
        ),
        "variance_pairs_csv": os.path.join(
            out_dir, f"{stem}_subject_projection_behavior_variance_pairs.csv"
        ),
        "cv_pairs_csv": os.path.join(
            out_dir, f"{stem}_subject_projection_behavior_cv_pairs.csv"
        ),
        "cv_ds_plot": os.path.join(
            out_dir, f"{stem}_projection_behavior_cv_ds_analysis.png"
        ),
        "cv_ds_pairs_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_cv_ds_pairs.csv"
        ),
        "cv_ds_stats_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_projection_behavior_cv_ds_stats.csv"
        ),
        "cv_run_pairs_outlier_csv": os.path.join(
            out_dir, f"{stem}_sub_ses_run_cv_pairs_outlier_flags.csv"
        ),
    }

    run_df.to_csv(output_paths["run_csv"], index=False)
    behavior_df.to_csv(output_paths["behavior_csv"], index=False)
    comparison_df.to_csv(output_paths["compare_csv"], index=False)
    run_cv_df.to_csv(output_paths["run_cv_csv"], index=False)
    subject_metrics_df.to_csv(output_paths["subject_metrics_csv"], index=False)
    cv_run_pair_outlier_df.to_csv(output_paths["cv_run_pairs_outlier_csv"], index=False)

    plot_variance_cv_subject_session_run_3x2(
        comparison_df,
        output_paths["variance_cv_3x2_plot"],
    )
    plot_run_variance_density(run_df, output_paths["run_density_plot"])
    variance_stats_row, variance_pairs_df = plot_scaled_variance_comparison_density(
        comparison_df,
        output_paths["compare_density_plot"],
        outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
    )
    cv_stats_row, cv_pairs_df = plot_sub_ses_run_cv_comparison(
        run_cv_df,
        output_paths["run_cv_plot"],
        outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
    )
    cv_ds_stats_row, cv_ds_pairs_df = analyze_subject_cv_difference(
        run_cv_df,
        output_paths["cv_ds_plot"],
        outlier_iqr_multiplier=float(args.outlier_iqr_multiplier),
        n_permutations=int(args.permutation_iterations),
        n_bootstrap=int(args.bootstrap_iterations),
        bootstrap_ci_percent=float(args.bootstrap_ci_percent),
        random_seed=int(args.random_seed),
    )

    paired_stats_df = pd.DataFrame([variance_stats_row, cv_stats_row])
    variance_pairs_df.to_csv(output_paths["variance_pairs_csv"], index=False)
    cv_pairs_df.to_csv(output_paths["cv_pairs_csv"], index=False)
    cv_ds_pairs_df.to_csv(output_paths["cv_ds_pairs_csv"], index=False)
    paired_stats_df.to_csv(output_paths["paired_stats_csv"], index=False)
    pd.DataFrame([cv_ds_stats_row]).to_csv(output_paths["cv_ds_stats_csv"], index=False)

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
