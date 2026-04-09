#!/usr/bin/env python3
"""Create sham-comparison summary tables for Method 1 GVS outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METHOD1_DIR = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "GVS_effects"
    / "four_methods_v2"
    / "method1_sham_referenced_raw_distance"
)

METRIC_TITLES = {
    "laplacian_spectral_distance": "Laplacian spectral distance",
    "frobenius_norm": "Frobenius norm",
    "correlation_distance": "Correlation distance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Method 1 sham-reference outputs and create compact summary "
            "tables showing which GVS conditions differ statistically from sham."
        )
    )
    parser.add_argument(
        "--method1-dir",
        type=Path,
        default=DEFAULT_METHOD1_DIR,
        help="Path to the method1_sham_referenced_raw_distance result directory.",
    )
    return parser.parse_args()


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in str(value).lower()).strip("_")


def _format_interval(lower: float, upper: float) -> str:
    if not (np.isfinite(lower) and np.isfinite(upper)):
        return "NA"
    return f"[{lower:.3f}, {upper:.3f}]"


def load_method1_tables(method1_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    tables_dir = method1_dir / "tables"
    plots_dir = method1_dir / "plots"
    stats_path = tables_dir / "method1_group_statistics.csv"
    observed_path = tables_dir / "method1_subject_observed_distances.csv"

    if not stats_path.exists():
        raise FileNotFoundError(f"Missing statistics table: {stats_path}")
    if not observed_path.exists():
        raise FileNotFoundError(f"Missing observed-distance table: {observed_path}")

    stats_df = pd.read_csv(stats_path)
    observed_df = pd.read_csv(observed_path)
    return stats_df, observed_df, tables_dir, plots_dir


def build_summary_table(stats_df: pd.DataFrame, observed_df: pd.DataFrame) -> pd.DataFrame:
    observed_summary = (
        observed_df.groupby(
            ["medication", "condition_code", "condition_name", "condition_factor", "distance_metric"],
            dropna=False,
            observed=False,
        )["distance_value"]
        .agg(
            observed_mean="mean",
            observed_sd=lambda values: float(np.std(values, ddof=1)) if len(values) >= 2 else float("nan"),
            observed_sem=lambda values: float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) >= 2 else float("nan"),
        )
        .reset_index()
    )

    summary_df = stats_df.merge(
        observed_summary,
        on=["medication", "condition_code", "condition_name", "condition_factor", "distance_metric"],
        how="left",
    )
    summary_df["delta_vs_null_mean"] = summary_df["t_obs_mean_distance"] - summary_df["null_mean"]
    summary_df["different_from_sham_uncorrected"] = summary_df["empirical_p_value"] < 0.05
    summary_df["different_from_sham_fdr"] = summary_df["significant_fdr"].fillna(False).astype(bool)
    summary_df["null_95pct_interval"] = [
        _format_interval(lower, upper)
        for lower, upper in zip(summary_df["null_q025"], summary_df["null_q975"], strict=False)
    ]
    summary_df["distance_metric_title"] = summary_df["distance_metric"].map(METRIC_TITLES).fillna(summary_df["distance_metric"])

    summary_df = summary_df.sort_values(
        ["distance_metric", "medication", "condition_factor", "condition_code"]
    ).reset_index(drop=True)
    return summary_df


def write_csv_tables(summary_df: pd.DataFrame, tables_dir: Path) -> None:
    combined_columns = [
        "distance_metric",
        "distance_metric_title",
        "medication",
        "condition_code",
        "condition_name",
        "condition_factor",
        "n_subjects_observed",
        "t_obs_mean_distance",
        "observed_sd",
        "observed_sem",
        "null_mean",
        "delta_vs_null_mean",
        "null_95pct_interval",
        "empirical_p_value",
        "q_value_fdr",
        "different_from_sham_uncorrected",
        "different_from_sham_fdr",
    ]
    summary_df.loc[:, combined_columns].to_csv(
        tables_dir / "method1_statistics_vs_sham_summary.csv",
        index=False,
    )

    significance_rows: list[dict[str, str]] = []
    for (metric_name, medication), group_df in summary_df.groupby(["distance_metric", "medication"], sort=False):
        significant_conditions = group_df.loc[
            group_df["different_from_sham_fdr"], "condition_factor"
        ].astype(str).tolist()
        significance_rows.append(
            {
                "distance_metric": metric_name,
                "distance_metric_title": METRIC_TITLES.get(metric_name, metric_name),
                "medication": str(medication),
                "gvs_conditions_significantly_different_from_sham_fdr": ", ".join(significant_conditions)
                if significant_conditions
                else "None",
            }
        )
    pd.DataFrame(significance_rows).to_csv(
        tables_dir / "method1_significant_gvs_vs_sham_fdr.csv",
        index=False,
    )

    for metric_name, metric_df in summary_df.groupby("distance_metric", sort=False):
        metric_columns = [
            "medication",
            "condition_factor",
            "n_subjects_observed",
            "t_obs_mean_distance",
            "observed_sd",
            "observed_sem",
            "null_mean",
            "delta_vs_null_mean",
            "null_95pct_interval",
            "empirical_p_value",
            "q_value_fdr",
            "different_from_sham_fdr",
        ]
        metric_df.loc[:, metric_columns].to_csv(
            tables_dir / f"method1_{_slugify(metric_name)}_statistics_vs_sham.csv",
            index=False,
        )


def render_metric_table(metric_df: pd.DataFrame, metric_name: str, out_path: Path) -> None:
    display_df = metric_df.loc[
        :,
        [
            "medication",
            "condition_factor",
            "n_subjects_observed",
            "t_obs_mean_distance",
            "observed_sem",
            "null_mean",
            "empirical_p_value",
            "q_value_fdr",
            "different_from_sham_fdr",
        ],
    ].copy()
    display_df.columns = [
        "Medication",
        "GVS",
        "N",
        "MeanDist",
        "SEM",
        "NullMean",
        "p",
        "q(FDR)",
        "Diff vs sham",
    ]
    display_df["MeanDist"] = display_df["MeanDist"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "NA")
    display_df["SEM"] = display_df["SEM"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "NA")
    display_df["NullMean"] = display_df["NullMean"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "NA")
    display_df["p"] = display_df["p"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "NA")
    display_df["q(FDR)"] = display_df["q(FDR)"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "NA")
    display_df["Diff vs sham"] = display_df["Diff vs sham"].map(lambda value: "Yes" if bool(value) else "No")

    n_rows = int(display_df.shape[0])
    fig_height = max(4.8, 0.42 * (n_rows + 3))
    fig, ax = plt.subplots(figsize=(11.5, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.to_numpy(),
        colLabels=display_df.columns.tolist(),
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_facecolor("#e9eef5")
            cell.set_text_props(weight="bold")
        elif display_df.iloc[row - 1]["Diff vs sham"] == "Yes":
            cell.set_facecolor("#fde2e1")
        else:
            cell.set_facecolor("#ffffff" if row % 2 else "#f7f9fb")

    sig_conditions = metric_df.loc[metric_df["different_from_sham_fdr"], ["medication", "condition_factor"]]
    if sig_conditions.empty:
        subtitle = "FDR-significant vs sham: None"
    else:
        parts = [f"{row.medication}-{row.condition_factor}" for row in sig_conditions.itertuples(index=False)]
        subtitle = f"FDR-significant vs sham: {', '.join(parts)}"

    ax.set_title(
        f"Method 1 sham-reference statistics\n{METRIC_TITLES.get(metric_name, metric_name)}\n{subtitle}",
        fontsize=12,
        pad=18,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_plot_tables(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    for metric_name, metric_df in summary_df.groupby("distance_metric", sort=False):
        render_metric_table(
            metric_df=metric_df,
            metric_name=metric_name,
            out_path=plots_dir / f"method1_{_slugify(metric_name)}_statistics_vs_sham_table.png",
        )


def main() -> None:
    args = parse_args()
    method1_dir = args.method1_dir.expanduser().resolve()
    stats_df, observed_df, tables_dir, plots_dir = load_method1_tables(method1_dir)
    summary_df = build_summary_table(stats_df=stats_df, observed_df=observed_df)
    write_csv_tables(summary_df=summary_df, tables_dir=tables_dir)
    write_plot_tables(summary_df=summary_df, plots_dir=plots_dir)
    print(f"Saved summary CSV tables to: {tables_dir}", flush=True)
    print(f"Saved table figures to: {plots_dir}", flush=True)


if __name__ == "__main__":
    main()
