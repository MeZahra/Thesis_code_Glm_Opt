#!/usr/bin/env python3
"""Subject-level comparison plots for projection variance and std/rms by ses/run.

This script mirrors the projection/run bookkeeping from `motor_brain_com.py`,
but focuses only on projection metrics (no behavior comparison).

Outputs:
1) Per-run/session variance CSV.
2) Per-run/session std/rms CSV.
3) Variability figure (2x2): ses1-run1, ses1-run2, ses2-run1, ses2-run2.
4) std/rms figure (2x2): ses1-run1, ses1-run2, ses2-run1, ses2-run2.
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from scipy.stats import ttest_rel, wilcoxon

from motor_brain_com import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_TRIAL_KEEP_ROOT,
    compute_run_variances,
    split_projection_by_run,
)

VARIANCE_SCALE = 1e7
PAIRWISE_ALPHA = 0.05


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection-path",
        default=(
            "/home/zkavian/Thesis_code_Glm_Opt/results/"
            "projection_voxel_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.6_gamma1.npy"
        ),
        help="Path to projection vector (.npy).",
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

    projection_path = os.path.abspath(os.path.expanduser(args.projection_path))
    manifest_path = os.path.abspath(os.path.expanduser(args.manifest_path))
    trial_keep_root = os.path.abspath(os.path.expanduser(args.trial_keep_root))

    projection = np.asarray(np.load(projection_path)).ravel()
    manifest_df = pd.read_csv(manifest_path, sep="\t")

    run_segments, layout = split_projection_by_run(projection, manifest_df, trial_keep_root)
    run_variance_df = compute_run_variances(run_segments)
    run_variance_df["variance_projection"] = (
        run_variance_df["variance_projection"].astype(np.float64) * float(VARIANCE_SCALE)
    )
    run_std_rms_df = compute_run_projection_std_over_rms(run_segments)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.dirname(projection_path) or os.getcwd()
    out_dir = os.path.abspath(os.path.expanduser(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(projection_path))[0]
    variance_csv_path = os.path.join(out_dir, f"{stem}_sub_ses_run_projection_variability.csv")
    std_rms_csv_path = os.path.join(out_dir, f"{stem}_sub_ses_run_projection_std_over_rms.csv")
    stats_csv_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_variability_pairwise_stats.csv"
    )
    variance_plot_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_variability_subject_comp.png"
    )
    std_rms_plot_path = os.path.join(
        out_dir, f"{stem}_sub_ses_run_projection_std_over_rms_subject_comp.png"
    )

    pairwise_stats_df = build_projection_variability_stats_table(run_variance_df)

    run_variance_df.to_csv(variance_csv_path, index=False)
    run_std_rms_df.to_csv(std_rms_csv_path, index=False)
    pairwise_stats_df.to_csv(stats_csv_path, index=False)

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

    print(f"Projection length: {projection.size}")
    print(f"Projection layout: {layout}")
    print(f"Rows (variance): {len(run_variance_df)}")
    print(f"Rows (std/rms): {len(run_std_rms_df)}")
    print(f"Saved variance CSV: {variance_csv_path}")
    print(f"Saved std/rms CSV:  {std_rms_csv_path}")
    print(f"Saved stats CSV:    {stats_csv_path}")
    print(f"Saved variability figure: {variance_plot_path}")
    print(f"Saved std/rms figure:    {std_rms_plot_path}")
    print("\nProjection variability pairwise statistics:")
    print(pairwise_stats_df.to_string(index=False))


if __name__ == "__main__":
    main()
