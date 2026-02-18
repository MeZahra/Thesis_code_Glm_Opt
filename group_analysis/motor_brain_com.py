import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import gaussian_kde, ttest_1samp, ttest_rel, wilcoxon

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


def compute_run_behavior_tables(run_segments, behavior_root, behavior_column):
    run_rows = []
    behavior_rows = []
    cv_rows = []

    for segment in run_segments:
        proj_values = np.asarray(segment["values"], dtype=np.float64)
        proj_finite = proj_values[np.isfinite(proj_values)]
        if proj_finite.size == 0:
            proj_mean = np.nan
            proj_var = np.nan
        else:
            proj_mean = float(np.mean(proj_finite))
            proj_var = float(np.var(proj_finite))
        run_rows.append(
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

        kept_projection, kept_behavior, behavior_path = _load_run_kept_projection_behavior(
            segment, behavior_root, behavior_column
        )
        beh_finite = kept_behavior[np.isfinite(kept_behavior)]
        if beh_finite.size == 0:
            beh_mean = np.nan
            beh_var = np.nan
        else:
            beh_mean = float(np.mean(beh_finite))
            beh_var = float(np.var(beh_finite))
        behavior_rows.append(
            {
                "sub_tag": str(segment["sub_tag"]),
                "ses": int(segment["ses"]),
                "run": int(segment["run"]),
                "n_trials_behavior_source": int(segment["n_trials_source"]),
                "n_trials_behavior_kept": int(beh_finite.size),
                "mean_behavior_col2": beh_mean,
                "variance_behavior_col2": beh_var,
                "behavior_path": behavior_path,
            }
        )

        finite_mask = np.isfinite(kept_projection) & np.isfinite(kept_behavior)
        projection_values = kept_projection[finite_mask]
        behavior_values = kept_behavior[finite_mask]

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


def _fallback_subject_aggregate_log_ratio(
    fit_df, alpha=0.05, fit_error_chain=None, n_rows_total=None
):
    subject_df = (
        fit_df.groupby("sub_tag", as_index=False)["log_rel_diff_proj_minus_beh"].mean()
    )
    diff = subject_df["log_rel_diff_proj_minus_beh"].to_numpy(dtype=np.float64)
    diff = diff[np.isfinite(diff)]

    result = {
        "analysis_method": "subject_aggregate_fallback",
        "model_formula": "subject_mean(log_rel_diff_proj_minus_beh) ~ 1",
        "optimizer_used": "fallback",
        "converged": False,
        "mixedlm_error_chain": str(fit_error_chain or ""),
        "n_rows_total": int(len(fit_df) if n_rows_total is None else n_rows_total),
        "n_rows_used": int(len(fit_df)),
        "n_subjects_used": int(subject_df["sub_tag"].nunique()),
        "n_rows": int(len(fit_df)),
        "n_subjects": int(subject_df["sub_tag"].nunique()),
        "alpha": float(alpha),
        "intercept_estimate": np.nan,
        "intercept_p_two_sided": np.nan,
        "intercept_p_one_sided_less": np.nan,
        "random_intercept_var": np.nan,
        "subject_mean_diff": np.nan,
        "fallback_wilcoxon_stat": np.nan,
        "fallback_wilcoxon_p_less": np.nan,
        "fallback_ttest_stat": np.nan,
        "fallback_ttest_p_less": np.nan,
        "supports_hypothesis": False,
        "decision_label": "insufficient_data_fallback",
    }

    if diff.size < 2:
        return result

    result["subject_mean_diff"] = float(np.mean(diff))

    try:
        w_stat, w_p = wilcoxon(diff, alternative="less")
        result["fallback_wilcoxon_stat"] = float(w_stat)
        result["fallback_wilcoxon_p_less"] = float(w_p)
    except ValueError:
        pass

    try:
        t_res = ttest_1samp(diff, popmean=0.0, nan_policy="omit")
        result["fallback_ttest_stat"] = float(t_res.statistic)
        result["fallback_ttest_p_less"] = _one_sided_p_less_from_two_sided(
            float(t_res.pvalue), float(t_res.statistic)
        )
    except TypeError:
        t_res = ttest_1samp(diff, popmean=0.0)
        result["fallback_ttest_stat"] = float(t_res.statistic)
        result["fallback_ttest_p_less"] = _one_sided_p_less_from_two_sided(
            float(t_res.pvalue), float(t_res.statistic)
        )

    primary_p = result["fallback_wilcoxon_p_less"]
    if not np.isfinite(primary_p):
        primary_p = result["fallback_ttest_p_less"]
    supports = bool(np.isfinite(primary_p) and primary_p < float(alpha))
    result["supports_hypothesis"] = supports
    result["decision_label"] = (
        "support_H1_projection_less_subject_aggregate"
        if supports
        else "reject_H1_projection_less_subject_aggregate"
    )
    return result


def fit_subject_median_log_ratio_mixedlm(comparison_df, alpha=0.05):
    formula = "log_rel_diff_proj_minus_beh ~ C(ses) + C(run)"
    fit_df = comparison_df[
        ["sub_tag", "ses", "run", "log_rel_diff_proj_minus_beh"]
    ].copy()
    fit_df["sub_tag"] = fit_df["sub_tag"].astype(str)
    fit_df["ses"] = pd.to_numeric(fit_df["ses"], errors="coerce")
    fit_df["run"] = pd.to_numeric(fit_df["run"], errors="coerce")
    fit_df["log_rel_diff_proj_minus_beh"] = pd.to_numeric(
        fit_df["log_rel_diff_proj_minus_beh"], errors="coerce"
    )

    finite_mask = (
        np.isfinite(fit_df["ses"].to_numpy(dtype=np.float64))
        & np.isfinite(fit_df["run"].to_numpy(dtype=np.float64))
        & np.isfinite(fit_df["log_rel_diff_proj_minus_beh"].to_numpy(dtype=np.float64))
    )
    fit_df = fit_df.loc[finite_mask].copy()

    if fit_df.empty or fit_df["sub_tag"].nunique() < 2:
        return _fallback_subject_aggregate_log_ratio(
            fit_df,
            alpha=alpha,
            fit_error_chain="insufficient rows or subjects for mixed model",
            n_rows_total=len(comparison_df),
        )

    optimizer_order = ["lbfgs", "powell", "cg", "nm"]
    fit_errors = []
    for optimizer in optimizer_order:
        try:
            model = smf.mixedlm(formula, data=fit_df, groups=fit_df["sub_tag"])
            fit_result = model.fit(reml=False, method=optimizer, disp=False)
            intercept = float(fit_result.params.get("Intercept", np.nan))
            p_two_sided = float(fit_result.pvalues.get("Intercept", np.nan))
            p_less = _one_sided_p_less_from_two_sided(p_two_sided, intercept)
            supports = bool(np.isfinite(p_less) and intercept < 0.0 and p_less < float(alpha))

            random_var = np.nan
            if getattr(fit_result, "cov_re", None) is not None and fit_result.cov_re.size:
                random_var = float(np.asarray(fit_result.cov_re)[0, 0])

            return {
                "analysis_method": "mixedlm",
                "model_formula": formula,
                "optimizer_used": optimizer,
                "converged": bool(getattr(fit_result, "converged", False)),
                "mixedlm_error_chain": "",
                "n_rows_total": int(len(comparison_df)),
                "n_rows_used": int(len(fit_df)),
                "n_subjects_used": int(fit_df["sub_tag"].nunique()),
                "n_rows": int(len(fit_df)),
                "n_subjects": int(fit_df["sub_tag"].nunique()),
                "alpha": float(alpha),
                "intercept_estimate": intercept,
                "intercept_p_two_sided": p_two_sided,
                "intercept_p_one_sided_less": p_less,
                "random_intercept_var": random_var,
                "subject_mean_diff": np.nan,
                "fallback_wilcoxon_stat": np.nan,
                "fallback_wilcoxon_p_less": np.nan,
                "fallback_ttest_stat": np.nan,
                "fallback_ttest_p_less": np.nan,
                "supports_hypothesis": supports,
                "decision_label": (
                    "support_H1_projection_less_mixedlm"
                    if supports
                    else "reject_H1_projection_less_mixedlm"
                ),
            }
        except Exception as exc:  # noqa: BLE001
            fit_errors.append(f"{optimizer}:{exc.__class__.__name__}:{exc}")

    return _fallback_subject_aggregate_log_ratio(
        fit_df,
        alpha=alpha,
        fit_error_chain=" | ".join(fit_errors),
        n_rows_total=len(comparison_df),
    )


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


def plot_scaled_variance_comparison_density(
    comparison_df,
    out_path,
    grid_points=512,
    tail_annotate_threshold=80.0,
    mixedlm_stats=None,
):
    projection_raw = comparison_df["variance_projection"].to_numpy(dtype=np.float64)
    behavior_raw = comparison_df["variance_behavior_col2"].to_numpy(dtype=np.float64)
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
            tail_text = "\n".join(labels)
            if len(outlier_rows) > len(shown_rows):
                tail_text += "\n..."
            x_tail = float(np.max(behavior_raw[outlier_mask]))

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

    paired_projection_log = comparison_df[
        "log_variance_projection_rel_subject_median"
    ].to_numpy(dtype=np.float64)
    paired_behavior_log = comparison_df[
        "log_variance_behavior_rel_subject_median"
    ].to_numpy(dtype=np.float64)
    paired_mask = np.isfinite(paired_projection_log) & np.isfinite(paired_behavior_log)
    paired_projection_log = paired_projection_log[paired_mask]
    paired_behavior_log = paired_behavior_log[paired_mask]
    if paired_projection_log.size == 0:
        raise RuntimeError(
            "No finite subject-median normalized log-variance pairs available for subplot 3."
        )

    log_all = np.concatenate([paired_projection_log, paired_behavior_log])
    x_log = _density_grid(log_all, grid_points=grid_points, fallback_pad=1e-6)
    proj_log_density = _evaluate_density(paired_projection_log, x_log)
    beh_log_density = _evaluate_density(paired_behavior_log, x_log)

    ax2.plot(
        x_log,
        proj_log_density,
        color="tab:blue",
        linewidth=2.0,
    )
    ax2.fill_between(x_log, proj_log_density, alpha=0.18, color="tab:blue")
    ax2.plot(
        x_log,
        beh_log_density,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
    )
    ax2.fill_between(x_log, beh_log_density, alpha=0.18, color="tab:orange")
    ax2.set_xlabel("log(variance / subject median)")
    ax2.set_ylabel("Probability density")
    ax2.set_title("3) Subject-median log-rel comparison")

    stat_line = ""
    if mixedlm_stats is not None and mixedlm_stats.get("analysis_method") == "mixedlm":
        pval = float(mixedlm_stats.get("intercept_p_one_sided_less", np.nan))
        p_text = f"{pval:.3g}" if np.isfinite(pval) else "nan"
        stat_line = f"p(one-sided,<0)={p_text}"

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


def plot_sub_ses_run_cv_comparison(run_cv_df, out_path, grid_points=512):
    projection_cv = run_cv_df["cv_projection"].to_numpy(dtype=np.float64)
    behavior_cv = run_cv_df["cv_behavior_col2"].to_numpy(dtype=np.float64)

    finite_proj = projection_cv[np.isfinite(projection_cv)]
    finite_beh = behavior_cv[np.isfinite(behavior_cv)]
    paired_mask_raw = np.isfinite(projection_cv) & np.isfinite(behavior_cv)
    paired_proj_raw = projection_cv[paired_mask_raw]
    paired_beh_raw = behavior_cv[paired_mask_raw]

    if finite_proj.size == 0:
        raise RuntimeError("No finite sub/ses/run projection CV values available to plot.")
    if finite_beh.size == 0:
        raise RuntimeError("No finite sub/ses/run behavior CV values available to plot.")
    if paired_proj_raw.size == 0:
        raise RuntimeError("No finite paired sub/ses/run raw CV values available to plot.")

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

    combined = np.concatenate([finite_proj, finite_beh])
    x_combined = _density_grid(combined, grid_points=grid_points, fallback_pad=1e-6)
    proj_density_combined = _evaluate_density(finite_proj, x_combined)
    beh_density_combined = _evaluate_density(finite_beh, x_combined)
    ax2.plot(
        x_combined,
        proj_density_combined,
        color="tab:blue",
        linewidth=2.0,
        label="CV projection",
    )
    ax2.fill_between(x_combined, proj_density_combined, alpha=0.15, color="tab:blue")
    ax2.plot(
        x_combined,
        beh_density_combined,
        color="tab:orange",
        linewidth=2.0,
        linestyle="--",
        label="CV behavior",
    )
    ax2.fill_between(x_combined, beh_density_combined, alpha=0.15, color="tab:orange")
    ax2.set_xlabel("CV")
    ax2.set_ylabel("Probability density")
    ax2.set_title("CV comparsion")
    ax2.legend(fontsize=8, loc="upper right")

    t_stat = np.nan
    t_p = np.nan
    try:
        t_res = ttest_rel(paired_proj_raw, paired_beh_raw, nan_policy="omit")
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    except TypeError:
        # SciPy fallback without nan_policy in very old versions
        t_res = ttest_rel(paired_proj_raw, paired_beh_raw)
        t_stat = float(t_res.statistic)
        t_p = float(t_res.pvalue)
    stats_text = (
        f"paired t p(two-sided)={t_p:.3g}, t={t_stat:.3g}"
        if np.isfinite(t_p)
        else "paired t-test unavailable"
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
        paired_beh_raw,
        paired_proj_raw,
        s=30,
        alpha=0.8,
        color="tab:blue",
        edgecolors="none",
    )
    ax3.set_xlabel("CV behavior")
    ax3.set_ylabel("CV projection")
    ax3.set_title("4) Sub/ses/run CV scatter")

    r2 = np.nan
    if paired_proj_raw.size >= 2:
        corr = float(np.corrcoef(paired_beh_raw, paired_proj_raw)[0, 1])
        if np.unique(paired_beh_raw).size >= 2:
            slope, intercept = np.polyfit(paired_beh_raw, paired_proj_raw, 1)
            x_line = np.linspace(
                float(np.min(paired_beh_raw)),
                float(np.max(paired_beh_raw)),
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
            fitted = (slope * paired_beh_raw) + intercept
            ss_res = float(np.sum((paired_proj_raw - fitted) ** 2))
            ss_tot = float(np.sum((paired_proj_raw - np.mean(paired_proj_raw)) ** 2))
            if ss_tot > 0:
                r2 = 1.0 - (ss_res / ss_tot)
        if not np.isfinite(r2) and np.isfinite(corr):
            r2 = float(corr**2)
    else:
        corr = np.nan

    stats_items = [f"n={paired_proj_raw.size}"]
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
        "--tail-annotate-threshold",
        type=float,
        default=80.0,
        help="Annotate behavior-variance tail in subplot 2 for runs with variance > threshold.",
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
    comparison_df = build_projection_behavior_comparison(
        run_df, behavior_df, scale_method=args.scale_method
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
        "mixedlm_csv": os.path.join(
            out_dir, f"{stem}_projection_behavior_mixedlm_stats.csv"
        ),
    }

    mixedlm_stats = fit_subject_median_log_ratio_mixedlm(
        comparison_df, alpha=float(args.alpha)
    )

    run_df.to_csv(output_paths["run_csv"], index=False)
    behavior_df.to_csv(output_paths["behavior_csv"], index=False)
    comparison_df.to_csv(output_paths["compare_csv"], index=False)
    run_cv_df.to_csv(output_paths["run_cv_csv"], index=False)
    pd.DataFrame([mixedlm_stats]).to_csv(output_paths["mixedlm_csv"], index=False)

    plot_run_variance_density(run_df, output_paths["run_density_plot"])
    plot_scaled_variance_comparison_density(
        comparison_df,
        output_paths["compare_density_plot"],
        tail_annotate_threshold=float(args.tail_annotate_threshold),
        mixedlm_stats=mixedlm_stats,
    )
    plot_sub_ses_run_cv_comparison(run_cv_df, output_paths["run_cv_plot"])

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
