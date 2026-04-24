#!/usr/bin/env python3
"""Summarize per-session behavior normality across subjects.

This script reads the kept trial-wise behavior values used to build the
`*_behavior_trial_combined_run_histogram_panel.png` figures and computes
normality tests on the pooled within-session values (combined across runs).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRIAL_VALUES_CSV = (
    REPO_ROOT / "results" / "behavior_trial_lineplot_panels" / "behavior_trial_values.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "results"
    / "behavior_trial_lineplot_panels"
    / "behavior_trial_normality_summary.csv"
)
DEFAULT_OUTPUT_PNG = (
    REPO_ROOT
    / "results"
    / "behavior_trial_lineplot_panels"
    / "behavior_trial_normality_summary_table.png"
)
DEFAULT_PANEL_DIR = REPO_ROOT / "results" / "behavior_trial_lineplot_panels"
DEFAULT_EXCLUDED_SUB_TAGS = {"sub-pd017"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-subject/session normality tests on pooled behavior "
            "values used in the combined-run histogram panels."
        )
    )
    parser.add_argument(
        "--trial-values-csv",
        default=str(DEFAULT_TRIAL_VALUES_CSV),
        help="CSV containing kept trial-wise behavior values.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Output CSV path for the normality summary table.",
    )
    parser.add_argument(
        "--out-png",
        default=str(DEFAULT_OUTPUT_PNG),
        help="Output PNG path for the rendered normality summary table.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold used for Gaussian/non-Gaussian decisions.",
    )
    return parser.parse_args()


def _finite_values(values: pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _format_decision(pvalue: float, alpha: float) -> str:
    if not np.isfinite(pvalue):
        return "Undetermined"
    return "Gaussian" if float(pvalue) > float(alpha) else "Not Gaussian"


def _run_shapiro(values: np.ndarray) -> tuple[float, float]:
    if values.size < 3:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.shapiro(values)
    return (float(result.statistic), float(result.pvalue))


def _run_dagostino_pearson(values: np.ndarray) -> tuple[float, float]:
    if values.size < 8:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.normaltest(values)
    return (float(result.statistic), float(result.pvalue))


def _run_jarque_bera(values: np.ndarray) -> tuple[float, float]:
    if values.size < 2:
        return (np.nan, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = stats.jarque_bera(values)
    return (float(result.statistic), float(result.pvalue))


def _overall_decision(
    shapiro_p: float,
    dagostino_p: float,
    jarque_bera_p: float,
    alpha: float,
) -> str:
    available = [
        float(pvalue)
        for pvalue in (shapiro_p, dagostino_p, jarque_bera_p)
        if np.isfinite(pvalue)
    ]
    if len(available) == 0:
        return "Undetermined"
    if all(pvalue > float(alpha) for pvalue in available):
        return "Gaussian"
    return "Not Gaussian"


def build_normality_summary(trial_values_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    required_columns = {"sub_tag", "ses", "run", "behavior_value"}
    missing_columns = required_columns.difference(trial_values_df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns in trial-values CSV: {missing_text}")

    rows: list[dict[str, object]] = []
    grouped = trial_values_df.groupby(["sub_tag", "ses"], sort=True)

    for (sub_tag, ses), session_df in grouped:
        values = _finite_values(session_df["behavior_value"])
        shapiro_stat, shapiro_p = _run_shapiro(values)
        dagostino_stat, dagostino_p = _run_dagostino_pearson(values)
        jarque_bera_stat, jarque_bera_p = _run_jarque_bera(values)
        combined_panel_path = (
            DEFAULT_PANEL_DIR / f"{sub_tag}_behavior_trial_combined_run_histogram_panel.png"
        )

        rows.append(
            {
                "sub_tag": str(sub_tag),
                "ses": int(ses),
                "n_runs": int(session_df["run"].nunique()),
                "n_trials": int(values.size),
                "shapiro_wilk_stat": shapiro_stat,
                "shapiro_wilk_pvalue": shapiro_p,
                "shapiro_wilk_decision": _format_decision(shapiro_p, alpha),
                "dagostino_pearson_stat": dagostino_stat,
                "dagostino_pearson_pvalue": dagostino_p,
                "dagostino_pearson_decision": _format_decision(dagostino_p, alpha),
                "jarque_bera_stat": jarque_bera_stat,
                "jarque_bera_pvalue": jarque_bera_p,
                "jarque_bera_decision": _format_decision(jarque_bera_p, alpha),
                "overall_gaussian_decision": _overall_decision(
                    shapiro_p=shapiro_p,
                    dagostino_p=dagostino_p,
                    jarque_bera_p=jarque_bera_p,
                    alpha=alpha,
                ),
                "combined_run_histogram_panel_path": str(combined_panel_path),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    return summary_df.sort_values(["sub_tag", "ses"]).reset_index(drop=True)


def _format_numeric(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}g}"


def _format_test_cell(
    stat: float,
    pvalue: float,
    stat_label: str,
) -> str:
    return f"{stat_label}={_format_numeric(stat)}\n" f"p={_format_numeric(pvalue)}"


def _decision_color(decision: str) -> str:
    if decision == "Gaussian":
        return "#d9ead3"
    if decision == "Not Gaussian":
        return "#f4cccc"
    return "#f3f3f3"


def render_normality_table_png(
    summary_df: pd.DataFrame,
    out_png: Path,
    alpha: float,
) -> None:
    if summary_df.empty:
        raise ValueError("Summary table is empty; cannot render PNG.")

    display_df = pd.DataFrame(
        {
            "Subject-Ses": summary_df["sub_tag"].astype(str) + "-ses" + summary_df["ses"].astype(str),
            "Shapiro-Wilk": [
                _format_test_cell(stat, pvalue, "W")
                for stat, pvalue in zip(
                    summary_df["shapiro_wilk_stat"],
                    summary_df["shapiro_wilk_pvalue"],
                )
            ],
            "D'Agostino-Pearson": [
                _format_test_cell(stat, pvalue, "K2")
                for stat, pvalue in zip(
                    summary_df["dagostino_pearson_stat"],
                    summary_df["dagostino_pearson_pvalue"],
                )
            ],
            "Jarque-Bera": [
                _format_test_cell(stat, pvalue, "JB")
                for stat, pvalue in zip(
                    summary_df["jarque_bera_stat"],
                    summary_df["jarque_bera_pvalue"],
                )
            ],
            "Final Decision": summary_df["overall_gaussian_decision"].astype(str),
        }
    )

    n_rows = len(display_df)
    fig_height = max(8.0, 0.5 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.0, 0.0, 1.0, 0.95],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 1.45)

    col_widths = {
        0: 0.18,
        1: 0.24,
        2: 0.24,
        3: 0.20,
        4: 0.14,
    }
    n_cols = len(display_df.columns)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#5b5b5b")
        cell.set_linewidth(0.65)
        if col in col_widths:
            cell.set_width(col_widths[col])
        if row == 0:
            cell.set_facecolor("#dbe7f3")
            cell.set_text_props(weight="bold", color="black")
            cell.set_height(0.06)
            continue

        cell.set_height(0.052)
        row_idx = row - 1
        if col == 0:
            cell.set_facecolor("#f7f7f7")
            cell.set_text_props(weight="bold")
        elif col == 1:
            cell.set_facecolor(_decision_color(summary_df.iloc[row_idx]["shapiro_wilk_decision"]))
        elif col == 2:
            cell.set_facecolor(
                _decision_color(summary_df.iloc[row_idx]["dagostino_pearson_decision"])
            )
        elif col == 3:
            cell.set_facecolor(_decision_color(summary_df.iloc[row_idx]["jarque_bera_decision"]))
        elif col == 4:
            cell.set_facecolor(
                _decision_color(summary_df.iloc[row_idx]["overall_gaussian_decision"])
            )

    fig.suptitle(
        f"Behavior Normality Summary by Subject-Session (alpha={alpha:.2f})",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.01,
        0.015,
        "Green: Gaussian, Red: Not Gaussian, Gray: Undetermined",
        ha="left",
        va="bottom",
        fontsize=10,
    )
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    trial_values_csv = Path(args.trial_values_csv).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_png = Path(args.out_png).expanduser().resolve()
    alpha = float(args.alpha)

    if not trial_values_csv.exists():
        raise FileNotFoundError(f"Trial-values CSV not found: {trial_values_csv}")

    trial_values_df = pd.read_csv(trial_values_csv)
    if "sub_tag" not in trial_values_df.columns:
        raise ValueError("Trial-values CSV must contain a 'sub_tag' column.")
    trial_values_df = trial_values_df.loc[
        ~trial_values_df["sub_tag"].astype(str).isin(DEFAULT_EXCLUDED_SUB_TAGS)
    ].copy()
    summary_df = build_normality_summary(trial_values_df=trial_values_df, alpha=alpha)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    render_normality_table_png(summary_df=summary_df, out_png=out_png, alpha=alpha)

    gaussian_count = int((summary_df["overall_gaussian_decision"] == "Gaussian").sum())
    not_gaussian_count = int(
        (summary_df["overall_gaussian_decision"] == "Not Gaussian").sum()
    )
    undetermined_count = int(
        (summary_df["overall_gaussian_decision"] == "Undetermined").sum()
    )

    print(f"Wrote {len(summary_df)} subject-session rows to {out_csv}")
    print(f"Wrote PNG table to {out_png}")
    print(
        "Excluded subjects: "
        + ", ".join(sorted(DEFAULT_EXCLUDED_SUB_TAGS))
    )
    print(
        "Overall decisions: "
        f"Gaussian={gaussian_count}, "
        f"Not Gaussian={not_gaussian_count}, "
        f"Undetermined={undetermined_count}"
    )


if __name__ == "__main__":
    main()
