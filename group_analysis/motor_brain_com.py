#!/usr/bin/env python3
"""Compute run/session trial-variance summaries for projection and behavior.

This script:
1. Loads a projection vector saved by group_analysis/obj_param.py.
2. Uses concat_manifest + trial_keep_run*.npy files to split trials per run.
3. Computes projection variance over kept trials for each subject/session/run.
4. Loads behavior from /Data/zahra/behaviour, uses column 2 (index 1), and
   computes behavior variance over the same kept trials.
5. Scales both variance distributions (z-score or min-max) for comparison.
6. Saves per-run CSV summaries and distribution plots.
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, gaussian_kde, rankdata, ttest_rel, wilcoxon

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


def compute_run_variances(run_segments):
    rows = []
    for segment in run_segments:
        values = np.asarray(segment["values"], dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            mean_value = np.nan
            variance = np.nan
        else:
            mean_value = float(np.mean(finite_values))
            variance = float(np.var(finite_values))

        rows.append(
            {
                "sub_tag": segment["sub_tag"],
                "ses": segment["ses"],
                "run": segment["run"],
                "n_trials_source": int(segment["n_trials_source"]),
                "n_trials_kept": int(finite_values.size),
                "mean_projection": mean_value,
                "variance_projection": variance,
            }
        )

    return pd.DataFrame(rows)


def compute_behavior_variances(run_segments, behavior_root, behavior_column):
    rows = []
    for segment in run_segments:
        _, kept_behavior, behavior_path = _load_run_kept_projection_behavior(
            segment, behavior_root, behavior_column
        )
        n_trials_source = int(segment["n_trials_source"])
        finite_values = kept_behavior[np.isfinite(kept_behavior)]
        if finite_values.size == 0:
            mean_value = np.nan
            variance = np.nan
        else:
            mean_value = float(np.mean(finite_values))
            variance = float(np.var(finite_values))

        rows.append(
            {
                "sub_tag": segment["sub_tag"],
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_behavior_source": n_trials_source,
                "n_trials_behavior_kept": int(finite_values.size),
                "mean_behavior_col2": mean_value,
                "variance_behavior_col2": variance,
                "behavior_path": behavior_path,
            }
        )

    return pd.DataFrame(rows)


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


def compute_sub_ses_run_cvs(run_segments, behavior_root, behavior_column):
    rows = []
    for segment in run_segments:
        kept_projection, kept_behavior, _ = _load_run_kept_projection_behavior(
            segment, behavior_root, behavior_column
        )
        finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
        projection_values = kept_projection[finite_mask]
        behavior_values = kept_behavior[finite_mask]

        proj_mean, proj_std, proj_cv = _coefficient_of_variation(projection_values)
        beh_mean, beh_std, beh_cv = _coefficient_of_variation(behavior_values)

        rows.append(
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

    return pd.DataFrame(rows)


def fit_projection_behavior_unit_calibration(run_segments, behavior_root, behavior_column):
    projection_chunks = []
    behavior_chunks = []
    for segment in run_segments:
        kept_projection, kept_behavior, _ = _load_run_kept_projection_behavior(
            segment, behavior_root, behavior_column
        )
        finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
        if np.any(finite_mask):
            projection_chunks.append(kept_projection[finite_mask])
            behavior_chunks.append(kept_behavior[finite_mask])

    if not projection_chunks:
        raise RuntimeError("No finite projection/behavior trial pairs for unit calibration.")
    projection_all = np.concatenate(projection_chunks)
    behavior_all = np.concatenate(behavior_chunks)
    if projection_all.size < 2:
        raise RuntimeError("Need at least 2 finite paired trials for unit calibration.")

    design = np.column_stack([projection_all, np.ones_like(projection_all)])
    slope, intercept = np.linalg.lstsq(design, behavior_all, rcond=None)[0]
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise RuntimeError("Non-finite slope/intercept from projection-to-behavior calibration.")

    corr = np.corrcoef(projection_all, behavior_all)[0, 1]
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "n_pairs": int(projection_all.size),
        "trial_corr": float(corr) if np.isfinite(corr) else np.nan,
    }


def compute_unit_matched_variances(
    run_segments, behavior_root, behavior_column, slope, intercept
):
    rows = []
    for segment in run_segments:
        kept_projection, kept_behavior, behavior_path = _load_run_kept_projection_behavior(
            segment, behavior_root, behavior_column
        )
        finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
        projection_finite = kept_projection[finite_mask]
        behavior_finite = kept_behavior[finite_mask]

        if projection_finite.size == 0:
            variance_projection = np.nan
            variance_behavior = np.nan
        else:
            projection_behavior_units = (slope * projection_finite) + intercept
            variance_projection = float(np.var(projection_behavior_units))
            variance_behavior = float(np.var(behavior_finite))

        rows.append(
            {
                "sub_tag": segment["sub_tag"],
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_paired_finite": int(projection_finite.size),
                "variance_projection_unit_matched": variance_projection,
                "variance_behavior_unit_matched": variance_behavior,
                "projection_to_behavior_slope": float(slope),
                "projection_to_behavior_intercept": float(intercept),
                "behavior_path": behavior_path,
            }
        )

    return pd.DataFrame(rows)


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


def _paired_log_variance_space(projection_values, behavior_values, eps_scale=0.5):
    """Map both modalities into a shared log-variance space without regression calibration."""
    proj = np.asarray(projection_values, dtype=np.float64)
    beh = np.asarray(behavior_values, dtype=np.float64)

    paired_finite = np.isfinite(proj) & np.isfinite(beh)
    positive = np.concatenate([proj[paired_finite], beh[paired_finite]])
    positive = positive[positive > 0]
    if positive.size == 0:
        epsilon = 1e-12
    else:
        epsilon = max(float(np.min(positive)) * float(eps_scale), 1e-12)

    proj_log = np.full(proj.shape, np.nan, dtype=np.float64)
    beh_log = np.full(beh.shape, np.nan, dtype=np.float64)
    proj_finite = np.isfinite(proj)
    beh_finite = np.isfinite(beh)
    proj_log[proj_finite] = np.log10(np.clip(proj[proj_finite], 0.0, None) + epsilon)
    beh_log[beh_finite] = np.log10(np.clip(beh[beh_finite], 0.0, None) + epsilon)
    return proj_log, beh_log, float(epsilon)


def _to_percentile_scores(values):
    values = np.asarray(values, dtype=np.float64)
    scores = np.full(values.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    if finite_values.size == 0:
        return scores
    if finite_values.size == 1:
        scores[finite_mask] = 0.5
        return scores
    ranks = rankdata(finite_values, method="average")
    scores[finite_mask] = (ranks - 1.0) / (finite_values.size - 1.0)
    return scores


def _paired_quantile_space(projection_values, behavior_values):
    """Scale-free paired space based on within-modality percentile ranks [0, 1]."""
    proj = np.asarray(projection_values, dtype=np.float64)
    beh = np.asarray(behavior_values, dtype=np.float64)
    return _to_percentile_scores(proj), _to_percentile_scores(beh)


def _paired_hypothesis_projection_less(projection_values, behavior_values, alpha=0.05):
    """Test H1: projection < behavior on paired run/session values in H1 space."""
    proj = np.asarray(projection_values, dtype=np.float64)
    beh = np.asarray(behavior_values, dtype=np.float64)
    paired_mask = np.isfinite(proj) & np.isfinite(beh)
    proj = proj[paired_mask]
    beh = beh[paired_mask]

    result = {
        "n_pairs": int(proj.size),
        "alpha": float(alpha),
        "mean_diff_proj_minus_beh": np.nan,
        "median_diff_proj_minus_beh": np.nan,
        "wilcoxon_p_less": np.nan,
        "wilcoxon_stat": np.nan,
        "ttest_p_less": np.nan,
        "ttest_stat": np.nan,
        "sign_test_p_less": np.nan,
        "n_non_tied_pairs": 0,
        "n_proj_less": 0,
        "supports_hypothesis": False,
        "decision_label": "insufficient_data",
    }
    if proj.size < 2:
        return result

    diff = proj - beh
    result["mean_diff_proj_minus_beh"] = float(np.mean(diff))
    result["median_diff_proj_minus_beh"] = float(np.median(diff))
    non_tie_mask = diff != 0
    n_non_tied = int(np.count_nonzero(non_tie_mask))
    n_proj_less = int(np.count_nonzero(diff[non_tie_mask] < 0))
    result["n_non_tied_pairs"] = n_non_tied
    result["n_proj_less"] = n_proj_less
    if n_non_tied > 0:
        sign_res = binomtest(k=n_proj_less, n=n_non_tied, p=0.5, alternative="greater")
        result["sign_test_p_less"] = float(sign_res.pvalue)

    try:
        w_stat, w_p = wilcoxon(proj, beh, alternative="less")
        result["wilcoxon_stat"] = float(w_stat)
        result["wilcoxon_p_less"] = float(w_p)
    except ValueError:
        pass

    try:
        t_res = ttest_rel(proj, beh, alternative="less", nan_policy="omit")
        result["ttest_stat"] = float(t_res.statistic)
        result["ttest_p_less"] = float(t_res.pvalue)
    except TypeError:
        t_res = ttest_rel(proj, beh, nan_policy="omit")
        if np.isfinite(t_res.statistic) and np.isfinite(t_res.pvalue):
            if t_res.statistic < 0:
                result["ttest_p_less"] = float(t_res.pvalue / 2.0)
            else:
                result["ttest_p_less"] = float(1.0 - (t_res.pvalue / 2.0))
            result["ttest_stat"] = float(t_res.statistic)

    # Primary decision uses the one-sided Wilcoxon test.
    if np.isfinite(result["wilcoxon_p_less"]):
        supports = bool(result["wilcoxon_p_less"] < alpha)
        result["supports_hypothesis"] = supports
        result["decision_label"] = "support_H1_projection_less" if supports else "reject_H1_projection_less"

    return result


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


def plot_scaled_variance_comparison_density(
    comparison_df,
    out_path,
    scale_method,
    alpha=0.05,
    grid_points=512,
    hypothesis_projection_values=None,
    hypothesis_behavior_values=None,
    hypothesis_label=None,
    tail_annotate_threshold=80.0,
    third_exclude_behavior_var_above=80.0,
):
    projection_raw = comparison_df["variance_projection"].to_numpy(dtype=np.float64)
    behavior_raw = comparison_df["variance_behavior_col2"].to_numpy(dtype=np.float64)
    labeled_rows_to_exclude = []
    tail_text = None
    x_tail = None
    y_tail = None
    if tail_annotate_threshold is not None:
        tail_threshold = float(tail_annotate_threshold)
        outlier_mask = np.isfinite(behavior_raw) & (behavior_raw > tail_threshold)
        if np.any(outlier_mask):
            outlier_rows = comparison_df.loc[
                outlier_mask, ["sub_tag", "ses", "run", "variance_behavior_col2"]
            ].sort_values("variance_behavior_col2", ascending=False)
            shown_rows = outlier_rows.head(3)
            labels = []
            for row in shown_rows.itertuples(index=False):
                labels.append(
                    f"{row.sub_tag} s{int(row.ses)} r{int(row.run)} beh={float(row.variance_behavior_col2):.6f}"
                )
                labeled_rows_to_exclude.append((str(row.sub_tag), int(row.ses), int(row.run)))
            tail_text = "\n".join(labels)
            if len(outlier_rows) > len(shown_rows):
                tail_text += "\n..."
            x_tail = float(np.max(behavior_raw[outlier_mask]))

    finite_pair_mask = np.isfinite(behavior_raw) & np.isfinite(projection_raw)
    paired_mask = finite_pair_mask.copy()
    exclusion_parts = []
    if third_exclude_behavior_var_above is not None:
        third_threshold = float(third_exclude_behavior_var_above)
        paired_mask &= behavior_raw <= third_threshold
        exclusion_parts.append(f"behavior<= {third_threshold:g}")

    if labeled_rows_to_exclude:
        labeled_exclude_mask = np.zeros(len(comparison_df), dtype=bool)
        for sub_tag, ses, run in labeled_rows_to_exclude:
            labeled_exclude_mask |= (
                (comparison_df["sub_tag"].astype(str).to_numpy() == str(sub_tag))
                & (comparison_df["ses"].astype(int).to_numpy() == int(ses))
                & (comparison_df["run"].astype(int).to_numpy() == int(run))
            )
        paired_mask &= ~labeled_exclude_mask
        exclusion_parts.append("labeled outlier rows removed")

    excluded_count = int(np.count_nonzero(finite_pair_mask) - np.count_nonzero(paired_mask))
    exclusion_suffix = f", {', '.join(exclusion_parts)}" if exclusion_parts else ""

    paired_behavior_raw = behavior_raw[paired_mask]
    paired_projection_raw = projection_raw[paired_mask]
    if paired_behavior_raw.size == 0:
        raise RuntimeError("No finite paired raw variances available for subplot 3.")

    paired_projection_z = _scale_values(paired_projection_raw, method="zscore")
    paired_behavior_z = _scale_values(paired_behavior_raw, method="zscore")
    paired_z_mask = np.isfinite(paired_projection_z) & np.isfinite(paired_behavior_z)
    paired_projection_z = paired_projection_z[paired_z_mask]
    paired_behavior_z = paired_behavior_z[paired_z_mask]
    if paired_projection_z.size == 0:
        raise RuntimeError("No finite paired z-scored variances available for subplot 3.")

    projection_scale_factor = 1e7
    projection_raw_scaled = projection_raw * projection_scale_factor
    x_proj = _density_grid(projection_raw_scaled, grid_points=grid_points, fallback_pad=1e-6)
    proj_raw_density = _evaluate_density(projection_raw_scaled, x_proj)

    x_beh = _density_grid(behavior_raw, grid_points=grid_points, fallback_pad=1e-6)
    beh_raw_density = _evaluate_density(behavior_raw, x_beh)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax0, ax1, ax2 = axes

    ax0.plot(x_proj, proj_raw_density, color="tab:blue", linewidth=2.0)
    ax0.fill_between(x_proj, proj_raw_density, alpha=0.2, color="tab:blue")
    ax0.set_xlabel("Projection variance")
    ax0.set_ylabel("Probability density")
    ax0.set_title("1) Projection var")

    ax1.plot(x_beh, beh_raw_density, color="tab:orange", linewidth=2.0)
    ax1.fill_between(x_beh, beh_raw_density, alpha=0.2, color="tab:orange")
    ax1.set_xlabel("Behavior variance")
    ax1.set_ylabel("Probability density")
    ax1.set_title("2) Behavior var")

    if tail_text is not None and x_tail is not None:
        y_tail = float(np.interp(x_tail, x_beh, beh_raw_density))
        ax1.annotate(
            tail_text,
            xy=(x_tail, y_tail),
            xytext=(0, 22),
            textcoords="offset points",
            va="bottom",
            ha="center",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.6"},
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.35"},
        )

    z_all = np.concatenate([paired_projection_z, paired_behavior_z])
    x_z = _density_grid(z_all, grid_points=grid_points, fallback_pad=1e-6)
    proj_z_density = _evaluate_density(paired_projection_z, x_z)
    beh_z_density = _evaluate_density(paired_behavior_z, x_z)

    ax2.plot(
        x_z,
        proj_z_density,
        color="tab:blue",
        linewidth=2.0,
        label="zscore(projection var)",
    )
    ax2.fill_between(x_z, proj_z_density, alpha=0.18, color="tab:blue")
    ax2.plot(
        x_z,
        beh_z_density,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label="zscore(behavior var)",
    )
    ax2.fill_between(x_z, beh_z_density, alpha=0.18, color="tab:orange")
    ax2.set_xlabel("Z-scored variance")
    ax2.set_ylabel("Probability density")
    ax2.set_title("comparision")

    mean_diff = float(np.mean(paired_projection_z - paired_behavior_z))

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    try:
        wilcoxon_stat, wilcoxon_p = wilcoxon(
            paired_projection_z, paired_behavior_z, alternative="two-sided"
        )
        wilcoxon_stat = float(wilcoxon_stat)
        wilcoxon_p = float(wilcoxon_p)
    except ValueError:
        pass

    t_stat = np.nan
    t_p = np.nan
    try:
        t_res = ttest_rel(paired_projection_z, paired_behavior_z, nan_policy="omit")
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    except TypeError:
        t_res = ttest_rel(paired_projection_z, paired_behavior_z)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

    stats = {
        "n_pairs": int(paired_projection_z.size),
        "excluded_pairs": int(excluded_count),
        "mean_diff_proj_minus_beh": float(mean_diff),
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p_two_sided": wilcoxon_p,
        "ttest_stat": t_stat,
        "ttest_p_two_sided": t_p,
        "decision_label": "paired_zscore_tests",
    }

    stat_line = f"n={paired_projection_z.size}"
    if excluded_count > 0:
        stat_line += f", excluded={excluded_count}"
    stat_line += f"\nmean(zproj-zbeh)={mean_diff:.3g}"
    if np.isfinite(wilcoxon_p):
        stat_line += f"\nWilcoxon p={wilcoxon_p:.3g}, W={wilcoxon_stat:.3g}"
    else:
        stat_line += "\nWilcoxon unavailable"
    if np.isfinite(t_p):
        stat_line += f"\npaired t p={t_p:.3g}, t={t_stat:.3g}"
    else:
        stat_line += "\npaired t unavailable"
    ax2.text(
        0.98,
        0.98,
        stat_line,
        transform=ax2.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return stats


def plot_sub_ses_run_cv_comparison(run_cv_df, out_path, alpha=0.05, grid_points=512):
    projection_cv = run_cv_df["cv_projection"].to_numpy(dtype=np.float64)
    behavior_cv = run_cv_df["cv_behavior_col2"].to_numpy(dtype=np.float64)
    projection_cv_z = _scale_values(projection_cv, method="zscore")
    behavior_cv_z = _scale_values(behavior_cv, method="zscore")

    finite_proj = projection_cv[np.isfinite(projection_cv)]
    finite_beh = behavior_cv[np.isfinite(behavior_cv)]
    finite_proj_z = projection_cv_z[np.isfinite(projection_cv_z)]
    finite_beh_z = behavior_cv_z[np.isfinite(behavior_cv_z)]
    paired_mask_raw = np.isfinite(projection_cv) & np.isfinite(behavior_cv)
    paired_proj_raw = projection_cv[paired_mask_raw]
    paired_beh_raw = behavior_cv[paired_mask_raw]
    paired_mask_z = np.isfinite(projection_cv_z) & np.isfinite(behavior_cv_z)
    paired_proj_z = projection_cv_z[paired_mask_z]
    paired_beh_z = behavior_cv_z[paired_mask_z]

    if finite_proj.size == 0:
        raise RuntimeError("No finite sub/ses/run projection CV values available to plot.")
    if finite_beh.size == 0:
        raise RuntimeError("No finite sub/ses/run behavior CV values available to plot.")
    if paired_proj_raw.size == 0:
        raise RuntimeError("No finite paired sub/ses/run raw CV values available to plot.")
    if paired_proj_z.size == 0:
        raise RuntimeError("No finite paired sub/ses/run z-scored CV values available to plot.")

    cv_hypothesis_stats = _paired_hypothesis_projection_less(
        paired_proj_raw, paired_beh_raw, alpha=float(alpha)
    )

    x_proj = _density_grid(finite_proj, grid_points=grid_points, fallback_pad=1e-6)
    proj_density = _evaluate_density(finite_proj, x_proj)
    x_beh = _density_grid(finite_beh, grid_points=grid_points, fallback_pad=1e-6)
    beh_density = _evaluate_density(finite_beh, x_beh)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    ax0, ax1 = axes[0]
    ax2, ax3 = axes[1]

    ax0.plot(x_proj, proj_density, color="tab:blue", linewidth=2.0)
    ax0.fill_between(x_proj, proj_density, alpha=0.2, color="tab:blue")
    ax0.set_xlabel("CV projection")
    ax0.set_ylabel("Probability density")
    ax0.set_title("CV projection")

    ax1.plot(x_beh, beh_density, color="tab:orange", linewidth=2.0)
    ax1.fill_between(x_beh, beh_density, alpha=0.2, color="tab:orange")
    ax1.set_xlabel("CV behavior")
    ax1.set_ylabel("Probability density")
    ax1.set_title("CV behaviour")

    combined = np.concatenate([finite_proj_z, finite_beh_z])
    x_combined = _density_grid(combined, grid_points=grid_points, fallback_pad=1e-6)
    proj_density_combined = _evaluate_density(finite_proj_z, x_combined)
    beh_density_combined = _evaluate_density(finite_beh_z, x_combined)
    ax2.plot(
        x_combined,
        proj_density_combined,
        color="tab:blue",
        linewidth=2.0,
        label="zscore(CV projection)",
    )
    ax2.fill_between(x_combined, proj_density_combined, alpha=0.15, color="tab:blue")
    ax2.plot(
        x_combined,
        beh_density_combined,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label="zscore(CV behavior)",
    )
    ax2.fill_between(x_combined, beh_density_combined, alpha=0.15, color="tab:orange")
    ax2.set_xlabel("Z-scored CV")
    ax2.set_ylabel("Probability density")
    ax2.set_title("CV comparsion")
    ax2.legend(fontsize=8, loc="upper right")

    proj_cv_mean = float(np.mean(finite_proj_z))
    proj_cv_std = float(np.std(finite_proj_z))
    beh_cv_mean = float(np.mean(finite_beh_z))
    beh_cv_std = float(np.std(finite_beh_z))
    diff_z = paired_proj_z - paired_beh_z
    mean_diff_z = float(np.mean(diff_z))

    wilcoxon_stat = np.nan
    wilcoxon_p = np.nan
    try:
        wilcoxon_stat, wilcoxon_p = wilcoxon(
            paired_proj_z, paired_beh_z, alternative="two-sided"
        )
        wilcoxon_stat = float(wilcoxon_stat)
        wilcoxon_p = float(wilcoxon_p)
    except ValueError:
        pass

    t_stat = np.nan
    t_p = np.nan
    try:
        t_res = ttest_rel(paired_proj_z, paired_beh_z, nan_policy="omit")
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    except TypeError:
        # SciPy fallback without nan_policy in very old versions
        t_res = ttest_rel(paired_proj_z, paired_beh_z)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)

    wilcoxon_line = (
        f"Wilcoxon p={wilcoxon_p:.3g}, W={wilcoxon_stat:.3g}"
        if np.isfinite(wilcoxon_p)
        else "Wilcoxon: unavailable"
    )
    ttest_line = (
        f"paired t p={t_p:.3g}, t={t_stat:.3g}"
        if np.isfinite(t_p)
        else "paired t: unavailable"
    )
    cv_h1_wilcoxon_line = (
        f"H1 raw CV (proj<beh): p={cv_hypothesis_stats['wilcoxon_p_less']:.3g}"
        if np.isfinite(cv_hypothesis_stats["wilcoxon_p_less"])
        else "H1 raw CV (proj<beh): Wilcoxon unavailable"
    )
    cv_h1_sign_line = (
        f"sign p={cv_hypothesis_stats['sign_test_p_less']:.3g}"
        if np.isfinite(cv_hypothesis_stats["sign_test_p_less"])
        else "sign p=nan"
    )
    stats_text = (
        f"n paired = {paired_proj_z.size}\n"
        f"mean(raw proj-beh)={cv_hypothesis_stats['mean_diff_proj_minus_beh']:.3g}\n"
        f"mean(zproj-zbeh)={mean_diff_z:.3g}\n"
        f"proj mean={proj_cv_mean:.3g}, std={proj_cv_std:.3g}\n"
        f"beh mean={beh_cv_mean:.3g}, std={beh_cv_std:.3g}\n"
        f"{cv_h1_wilcoxon_line}, {cv_h1_sign_line}\n"
        f"decision={cv_hypothesis_stats['decision_label']}\n"
        f"{wilcoxon_line}\n"
        f"{ttest_line}"
    )
    ax2.text(
        0.98,
        0.50,
        stats_text,
        transform=ax2.transAxes,
        va="center",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    ax3.scatter(
        paired_beh_z,
        paired_proj_z,
        s=30,
        alpha=0.8,
        color="tab:blue",
        edgecolors="none",
    )
    ax3.set_xlabel("Z-scored CV behavior")
    ax3.set_ylabel("Z-scored CV projection")
    ax3.set_title("4) Sub/ses/run z-scored CV scatter")

    r2 = np.nan
    if paired_proj_z.size >= 2:
        corr = float(np.corrcoef(paired_beh_z, paired_proj_z)[0, 1])
        if np.unique(paired_beh_z).size >= 2:
            slope, intercept = np.polyfit(paired_beh_z, paired_proj_z, 1)
            x_line = np.linspace(
                float(np.min(paired_beh_z)),
                float(np.max(paired_beh_z)),
                100,
            )
            y_line = (slope * x_line) + intercept
            ax3.plot(
                x_line,
                y_line,
                color="crimson",
                linewidth=1.6,
                linestyle="--",
            )
            fitted = (slope * paired_beh_z) + intercept
            ss_res = float(np.sum((paired_proj_z - fitted) ** 2))
            ss_tot = float(np.sum((paired_proj_z - np.mean(paired_proj_z)) ** 2))
            if ss_tot > 0:
                r2 = 1.0 - (ss_res / ss_tot)
        if not np.isfinite(r2) and np.isfinite(corr):
            r2 = float(corr**2)
    else:
        corr = np.nan

    stats_items = [f"n={paired_proj_z.size}"]
    stats_items.append(f"r={corr:.3f}" if np.isfinite(corr) else "r=nan")
    stats_items.append(f"R^2={r2:.3f}" if np.isfinite(r2) else "R^2=nan")
    corr_text = ", ".join(stats_items)
    ax3.text(
        0.98,
        0.98,
        corr_text,
        transform=ax3.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"},
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return cv_hypothesis_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection-path",
        default="/home/zkavian/Thesis_code_Glm_Opt/results/projection_voxel_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.6_gamma1.npy",
        help="Path to projection vector (.npy).",
    )
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help="Path to concat manifest TSV (default: group manifest).",
    )
    parser.add_argument(
        "--trial-keep-root",
        default=DEFAULT_TRIAL_KEEP_ROOT,
        help="Root containing trial_keep_run*.npy files.",
    )
    parser.add_argument(
        "--behavior-root",
        default=DEFAULT_BEHAVIOR_ROOT,
        help="Root containing PSPD*_ses_*_run_*.npy behavior files.",
    )
    parser.add_argument(
        "--behavior-column",
        type=int,
        default=1,
        help="Behavior column index (0-based). Use 1 for the second column.",
    )
    parser.add_argument(
        "--scale-method",
        default="zscore",
        choices=["zscore", "minmax"],
        help="Scaling method used to compare projection and behavior variances.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for H1: projection variance < behavior variance.",
    )
    parser.add_argument(
        "--hypothesis-space",
        default="paired_quantile",
        choices=["scaled", "raw", "paired_quantile", "log_raw", "unit_matched"],
        help=(
            "Values used for H1 test. "
            "'scaled' uses separate modality scaling; "
            "'raw' uses raw run variances; "
            "'paired_quantile' uses within-modality percentile ranks (recommended for scale mismatch); "
            "'log_raw' uses a shared log10(var + eps) transform; "
            "'unit_matched' keeps legacy trial-level linear calibration."
        ),
    )
    parser.add_argument(
        "--tail-annotate-threshold",
        type=float,
        default=80.0,
        help="Annotate behavior-variance tail in subplot 2 for runs with variance > threshold.",
    )
    parser.add_argument(
        "--third-exclude-behavior-var-above",
        type=float,
        default=80.0,
        help=(
            "Exclude runs with raw behavior variance above this threshold from subplot 3 "
            "and H1 statistics. Set a negative value to disable exclusion."
        ),
    )
    parser.add_argument("--out-dir", default=None, help="Output directory.")
    args = parser.parse_args()

    projection_path = os.path.abspath(os.path.expanduser(args.projection_path))
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest_path))
    trial_keep_root = os.path.abspath(os.path.expanduser(args.trial_keep_root))
    behavior_root = os.path.abspath(os.path.expanduser(args.behavior_root))

    projection = np.asarray(np.load(projection_path)).ravel()
    manifest_df = pd.read_csv(manifest_path, sep="\t")

    run_segments, layout = split_projection_by_run(projection, manifest_df, trial_keep_root)
    run_df = compute_run_variances(run_segments)
    behavior_df = compute_behavior_variances(
        run_segments,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
    )
    comparison_df = build_projection_behavior_comparison(
        run_df, behavior_df, scale_method=args.scale_method
    )
    unit_matched_df = None
    calibration = None
    hypothesis_projection_values = None
    hypothesis_behavior_values = None
    hypothesis_label = None
    log_eps = None
    if args.hypothesis_space == "scaled":
        hypothesis_projection_values = comparison_df["variance_projection_scaled"].to_numpy(
            dtype=np.float64
        )
        hypothesis_behavior_values = comparison_df["variance_behavior_scaled"].to_numpy(
            dtype=np.float64
        )
        hypothesis_label = f"scaled ({args.scale_method}), paired"
    elif args.hypothesis_space == "raw":
        hypothesis_projection_values = comparison_df["variance_projection"].to_numpy(
            dtype=np.float64
        )
        hypothesis_behavior_values = comparison_df["variance_behavior_col2"].to_numpy(
            dtype=np.float64
        )
        hypothesis_label = "raw paired variances"
    elif args.hypothesis_space == "paired_quantile":
        hypothesis_projection_values, hypothesis_behavior_values = _paired_quantile_space(
            comparison_df["variance_projection"].to_numpy(dtype=np.float64),
            comparison_df["variance_behavior_col2"].to_numpy(dtype=np.float64),
        )
        hypothesis_label = "paired quantile ranks [0,1]"
    elif args.hypothesis_space == "log_raw":
        hypothesis_projection_values, hypothesis_behavior_values, log_eps = _paired_log_variance_space(
            comparison_df["variance_projection"].to_numpy(dtype=np.float64),
            comparison_df["variance_behavior_col2"].to_numpy(dtype=np.float64),
        )
        hypothesis_label = f"log10(raw var + eps={log_eps:.2g}), paired"
    else:
        calibration = fit_projection_behavior_unit_calibration(
            run_segments, behavior_root=behavior_root, behavior_column=int(args.behavior_column)
        )
        unit_matched_df = compute_unit_matched_variances(
            run_segments,
            behavior_root=behavior_root,
            behavior_column=int(args.behavior_column),
            slope=calibration["slope"],
            intercept=calibration["intercept"],
        )
        merged_unit = comparison_df.merge(
            unit_matched_df[
                [
                    "sub_tag",
                    "ses",
                    "run",
                    "n_trials_paired_finite",
                    "variance_projection_unit_matched",
                    "variance_behavior_unit_matched",
                ]
            ],
            on=["sub_tag", "ses", "run"],
            how="inner",
            validate="one_to_one",
        )
        comparison_df = merged_unit
        hypothesis_projection_values = comparison_df[
            "variance_projection_unit_matched"
        ].to_numpy(dtype=np.float64)
        hypothesis_behavior_values = comparison_df[
            "variance_behavior_unit_matched"
        ].to_numpy(dtype=np.float64)
        hypothesis_label = "unit-matched paired variances"

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(projection_path) or os.getcwd()
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(projection_path))[0]
    run_csv_path = os.path.join(out_dir, f"{stem}_sub_ses_run_variance.csv")
    behavior_csv_path = os.path.join(out_dir, f"{stem}_sub_ses_run_behavior_variance.csv")
    compare_csv_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_behavior_variance.csv"
    )
    dist_plot_path = os.path.join(out_dir, f"{stem}_sub_ses_run_variance_density.png")
    compare_plot_path = os.path.join(
        out_dir, f"{stem}_projection_behavior_variance_scaled_density.png"
    )
    run_cv_csv_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_behavior_cv.csv"
    )
    run_cv_plot_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_behavior_cv_density.png"
    )
    unit_matched_csv_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_behavior_variance_unit_matched.csv"
    )

    run_cv_df = compute_sub_ses_run_cvs(
        run_segments, behavior_root=behavior_root, behavior_column=int(args.behavior_column)
    )

    run_df.to_csv(run_csv_path, index=False)
    behavior_df.to_csv(behavior_csv_path, index=False)
    comparison_df.to_csv(compare_csv_path, index=False)
    run_cv_df.to_csv(run_cv_csv_path, index=False)
    if unit_matched_df is not None:
        unit_matched_df.to_csv(unit_matched_csv_path, index=False)
    plot_run_variance_density(run_df, dist_plot_path)
    hypothesis_stats = plot_scaled_variance_comparison_density(
        comparison_df,
        compare_plot_path,
        scale_method=args.scale_method,
        alpha=float(args.alpha),
        hypothesis_projection_values=hypothesis_projection_values,
        hypothesis_behavior_values=hypothesis_behavior_values,
        hypothesis_label=hypothesis_label,
        tail_annotate_threshold=float(args.tail_annotate_threshold),
        third_exclude_behavior_var_above=(
            None
            if float(args.third_exclude_behavior_var_above) < 0
            else float(args.third_exclude_behavior_var_above)
        ),
    )
    cv_hypothesis_stats = plot_sub_ses_run_cv_comparison(
        run_cv_df, run_cv_plot_path, alpha=float(args.alpha)
    )

    paired_mask = np.isfinite(comparison_df["variance_projection"]) & np.isfinite(
        comparison_df["variance_behavior_col2"]
    )
    if np.any(paired_mask):
        paired_proj = comparison_df.loc[paired_mask, "variance_projection"].to_numpy(
            dtype=np.float64
        )
        paired_beh = comparison_df.loc[paired_mask, "variance_behavior_col2"].to_numpy(
            dtype=np.float64
        )
        if paired_proj.size >= 2:
            corr_value = float(np.corrcoef(paired_proj, paired_beh)[0, 1])
        else:
            corr_value = np.nan
    else:
        corr_value = np.nan

    print(f"Projection length: {projection.size}")
    print(f"Projection layout: {layout}")
    print(f"Run/session rows: {len(run_df)}")
    print(f"Behavior root: {behavior_root}")
    print(f"Behavior column: {args.behavior_column} (second column is index 1)")
    print(f"Scale method: {args.scale_method}")
    print(f"Hypothesis space: {args.hypothesis_space}")
    print(f"Tail annotate threshold (subplot 2): {float(args.tail_annotate_threshold):.3f}")
    if float(args.third_exclude_behavior_var_above) < 0:
        print("Third subplot exclusion: disabled")
    else:
        print(
            "Third subplot exclusion (raw behavior var > threshold): "
            f"{float(args.third_exclude_behavior_var_above):.3f}"
        )
    if calibration is not None:
        print(
            "Unit calibration (behavior ~= slope*projection + intercept): "
            f"slope={calibration['slope']}, intercept={calibration['intercept']}, "
            f"paired_trials={calibration['n_pairs']}, corr={calibration['trial_corr']}"
        )
    if log_eps is not None:
        print(f"Log-raw shared transform epsilon: {log_eps}")
    print(f"Hypothesis alpha: {float(args.alpha):.3f}")
    print(f"Projection-vs-behavior variance correlation: {corr_value}")
    print(
        "Variance subplot tests (paired z-scored variances): "
        f"n={hypothesis_stats['n_pairs']}, excluded={hypothesis_stats['excluded_pairs']}, "
        f"Wilcoxon p(two-sided)={hypothesis_stats['wilcoxon_p_two_sided']}, "
        f"W={hypothesis_stats['wilcoxon_stat']}, "
        f"paired t p(two-sided)={hypothesis_stats['ttest_p_two_sided']}, "
        f"t={hypothesis_stats['ttest_stat']}"
    )
    print(
        "CV hypothesis (raw paired CV, H1: projection < behavior): "
        f"Wilcoxon p(one-sided)={cv_hypothesis_stats['wilcoxon_p_less']}, "
        f"sign p(one-sided)={cv_hypothesis_stats['sign_test_p_less']}, "
        f"decision={cv_hypothesis_stats['decision_label']}"
    )
    print("\nPer run/session summary:")
    summary_cols = [
        "sub_tag",
        "ses",
        "run",
        "n_trials_kept",
        "variance_projection",
        "n_trials_behavior_kept",
        "variance_behavior_col2",
        "variance_projection_scaled",
        "variance_behavior_scaled",
    ]
    print(comparison_df.loc[:, summary_cols].to_string(index=False))
    print("\nPer sub/session/run CV summary:")
    run_cv_summary_cols = [
        "sub_tag",
        "ses",
        "run",
        "n_trials_paired_finite",
        "cv_projection",
        "cv_behavior_col2",
    ]
    print(run_cv_df.loc[:, run_cv_summary_cols].to_string(index=False))
    print(f"\nSaved run/session CSV: {run_csv_path}")
    print(f"Saved behavior CSV:    {behavior_csv_path}")
    print(f"Saved compare CSV:     {compare_csv_path}")
    print(f"Saved sub/ses/run CV CSV:  {run_cv_csv_path}")
    if unit_matched_df is not None:
        print(f"Saved unit-matched CSV:{unit_matched_csv_path}")
    print(f"Saved density plot:    {dist_plot_path}")
    print(f"Saved compare plot:    {compare_plot_path}")
    print(f"Saved sub/ses/run CV plot: {run_cv_plot_path}")


if __name__ == "__main__":
    main()
