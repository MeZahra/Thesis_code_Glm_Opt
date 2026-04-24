#!/usr/bin/env python3
"""Plot kept behavior summaries for each subject as session-run subplots.

The script uses the same behavior-file and trial_keep convention as
motor_brain_com.py, but instead of plotting only (trial i, trial i+1) pairs,
it summarizes each kept run in a 2x2 session-run panel. It saves:

1. trial-by-trial line plots
2. lag-1 scatter plots ((trial i), (trial i+1))
3. within-run histograms
4. lagged self-prediction cross-validation curves
5. within-session run-pair histograms
6. within-session run-pair cross-correlation curves
7. within-session combined-run histograms
8. within-session run-separated autocorrelation curves

Session colors are shared across runs from the same session.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from motor_brain_com import (
        DEFAULT_BEHAVIOR_ROOT,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_TRIAL_KEEP_ROOT,
        _demean_finite,
        _load_behavior_column,
        _prepare_run_entries,
        _resolve_behavior_path,
    )
except ImportError:
    from group_analysis.main.motor_brain_com import (
        DEFAULT_BEHAVIOR_ROOT,
        DEFAULT_MANIFEST_PATH,
        DEFAULT_TRIAL_KEEP_ROOT,
        _demean_finite,
        _load_behavior_column,
        _prepare_run_entries,
        _resolve_behavior_path,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "behavior_trial_lineplot_panels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each subject, save 2x2 session-run panels including trial-by-trial "
            "line plots and lag-1 scatter plots using the same behavior-file and "
            "trial_keep convention as motor_brain_com.py."
        )
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
    parser.add_argument(
        "--behavior-root",
        default=DEFAULT_BEHAVIOR_ROOT,
        help="Root containing PSPD*_ses_*_run_*.npy behavior files.",
    )
    parser.add_argument(
        "--behavior-column",
        type=int,
        default=1,
        help="Behavior column index (0-based). Default uses column 1.",
    )
    parser.add_argument(
        "--reciprocal-behavior",
        action="store_true",
        help="Replace the selected behavior values with 1/value before plotting.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory where per-subject figures will be saved.",
    )
    parser.add_argument(
        "--demean-within-run",
        action="store_true",
        help="Demean kept behavior values within each run before plotting.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=20,
        help="Number of histogram bins for each subject-level histogram panel.",
    )
    parser.add_argument(
        "--cv-max-lag",
        type=int,
        default=10,
        help="Maximum lag to evaluate in the lagged self-prediction cross-validation panel.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Requested number of forward-chaining cross-validation folds per lag.",
    )
    parser.add_argument(
        "--cv-min-train-size",
        type=int,
        default=12,
        help="Minimum number of training pairs required per cross-validation fold.",
    )
    parser.add_argument(
        "--cv-min-test-size",
        type=int,
        default=8,
        help="Minimum number of held-out pairs required per cross-validation fold.",
    )
    parser.add_argument(
        "--runpair-max-lag",
        type=int,
        default=20,
        help="Maximum positive/negative lag for within-session run-pair cross-correlation.",
    )
    parser.add_argument(
        "--runpair-min-overlap",
        type=int,
        default=10,
        help="Minimum overlap required to report a run-pair correlation at a given lag.",
    )
    parser.add_argument(
        "--combined-autocorr-max-lag",
        type=int,
        default=20,
        help="Maximum positive lag for within-session run-separated autocorrelation.",
    )
    parser.add_argument(
        "--combined-autocorr-min-overlap",
        type=int,
        default=10,
        help="Minimum lagged-pair count required to report per-run autocorrelation.",
    )
    return parser.parse_args()


def _build_subject_trial_table(
    manifest_df: pd.DataFrame,
    trial_keep_root: Path,
    behavior_root: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    run_entries = _prepare_run_entries(manifest_df, str(trial_keep_root))

    for entry in run_entries:
        sub_tag = str(entry["sub_tag"])
        ses = int(entry["ses"])
        run = int(entry["run"])
        keep_mask = np.asarray(entry["keep_mask"], dtype=bool)
        n_trials_source = int(entry["source_count"])

        behavior_path = _resolve_behavior_path(sub_tag, ses, run, str(behavior_root))
        behavior_values = np.asarray(
            _load_behavior_column(behavior_path, int(behavior_column)),
            dtype=np.float64,
        )

        if behavior_values.size < n_trials_source:
            raise ValueError(
                f"Behavior file has fewer trials than expected for {sub_tag} ses-{ses} run-{run}: "
                f"{behavior_values.size} < {n_trials_source}"
            )
        if behavior_values.size > n_trials_source:
            behavior_values = behavior_values[:n_trials_source]
        if reciprocal_behavior:
            behavior_values = _safe_reciprocal(behavior_values)

        kept_behavior = np.asarray(behavior_values[keep_mask], dtype=np.float64)
        if demean_within_run:
            kept_behavior = _demean_finite(kept_behavior)

        kept_trial_numbers = np.flatnonzero(keep_mask) + 1
        finite_mask = np.isfinite(kept_behavior)
        if not np.any(finite_mask):
            continue

        kept_trial_numbers = kept_trial_numbers[finite_mask]
        finite_behavior = kept_behavior[finite_mask]

        for trial_run, value in zip(kept_trial_numbers, finite_behavior):
            rows.append(
                {
                    "sub_tag": sub_tag,
                    "ses": ses,
                    "run": run,
                    "trial_index_run": int(trial_run),
                    "behavior_value": float(value),
                }
            )

    return pd.DataFrame(rows)


def _compute_y_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return (-1.0, 1.0)

    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    vrange = vmax - vmin
    pad = 0.05 * vrange if vrange > 0 else max(0.05 * abs(vmax), 1.0)
    return (vmin - pad, vmax + pad)


def _safe_reciprocal(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reciprocal = np.full(values.shape, np.nan, dtype=np.float64)
    valid_mask = np.isfinite(values) & (values != 0.0)
    reciprocal[valid_mask] = 1.0 / values[valid_mask]
    return reciprocal


def _extract_consecutive_trial_pairs(run_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    trial_indices = run_df["trial_index_run"].to_numpy(dtype=np.int64)
    behavior_values = run_df["behavior_value"].to_numpy(dtype=np.float64)
    if behavior_values.size < 2:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    consecutive_mask = trial_indices[1:] == (trial_indices[:-1] + 1)
    if not np.any(consecutive_mask):
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    x_values = behavior_values[:-1][consecutive_mask]
    y_values = behavior_values[1:][consecutive_mask]
    finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[finite_mask], y_values[finite_mask]


def _value_label(
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool = False,
) -> str:
    if reciprocal_behavior:
        label = f"1 / Behaviour column {int(behavior_column)}"
    else:
        label = f"Behaviour column {int(behavior_column)}"
    if demean_within_run:
        label += " (demeaned within run)"
    return label


def _extract_run_keys(subject_df: pd.DataFrame) -> list[tuple[int, int]]:
    run_keys = (
        subject_df.loc[:, ["ses", "run"]]
        .drop_duplicates()
        .sort_values(["ses", "run"])
        .itertuples(index=False)
    )
    run_keys = [(int(row.ses), int(row.run)) for row in run_keys]
    return run_keys


def _extract_session_keys(subject_df: pd.DataFrame) -> list[int]:
    return (
        subject_df["ses"]
        .dropna()
        .astype(int)
        .sort_values()
        .drop_duplicates()
        .tolist()
    )


def _build_session_color_map(subject_df: pd.DataFrame) -> dict[int, tuple[float, ...]]:
    session_keys = sorted(subject_df["ses"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10")
    return {int(ses): cmap(color_idx % 10) for color_idx, ses in enumerate(session_keys)}


def _build_run_color_map(subject_df: pd.DataFrame) -> dict[int, tuple[float, ...]]:
    run_keys = sorted(subject_df["run"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10")
    return {int(run): cmap(color_idx % 10) for color_idx, run in enumerate(run_keys)}


def _compute_histogram_bin_edges(values: np.ndarray, bins: int) -> np.ndarray:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return np.array([-1.0, 1.0], dtype=np.float64)
    if finite_values.size == 1 or np.isclose(np.ptp(finite_values), 0.0):
        center = float(finite_values[0])
        half_width = max(0.5, 0.05 * abs(center), 1e-3)
        return np.array([center - half_width, center + half_width], dtype=np.float64)
    return np.asarray(
        np.histogram_bin_edges(finite_values, bins=max(1, int(bins))),
        dtype=np.float64,
    )


def _plot_subject_consecutive_scatter(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=False, sharey=False)
    axes_flat = axes.ravel()

    run_keys = _extract_run_keys(subject_df)
    if len(run_keys) > len(axes_flat):
        raise ValueError(
            f"Expected at most {len(axes_flat)} session-run panels per subject, found {len(run_keys)}."
        )
    session_color_map = _build_session_color_map(subject_df)

    run_pair_values: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for ses, run in run_keys:
        run_df = subject_df.loc[
            (subject_df["ses"] == ses) & (subject_df["run"] == run)
        ].sort_values("trial_index_run")
        x_values, y_values = _extract_consecutive_trial_pairs(run_df)
        run_pair_values[(int(ses), int(run))] = (x_values, y_values)

    for ax, (ses, run) in zip(axes_flat, run_keys):
        x_values, y_values = run_pair_values[(int(ses), int(run))]
        color = session_color_map[int(ses)]
        if x_values.size > 0:
            axis_min, axis_max = _compute_y_limits(np.concatenate([x_values, y_values]))
        else:
            axis_min, axis_max = (-1.0, 1.0)

        if x_values.size == 0:
            ax.text(
                0.5,
                0.5,
                "No consecutive kept\ntrial pairs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
        else:
            ax.scatter(
                x_values,
                y_values,
                s=28,
                alpha=0.86,
                color=color,
                edgecolors="none",
            )

        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="black",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_title(f"ses-{ses} run-{run}", fontsize=11)

    for ax in axes_flat[len(run_keys) :]:
        ax.set_visible(False)

    for ax in axes_flat[: len(run_keys)]:
        if ax.get_visible():
            ax.set_xlabel("trial i")
            ax.set_ylabel("trial i+1")

    fig.tight_layout()
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _build_time_series_cv_splits(
    n_samples: int,
    n_splits: int,
    min_train_size: int,
    min_test_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_samples = int(max(0, n_samples))
    requested_splits = int(max(1, n_splits))
    min_train_size = int(max(2, min_train_size))
    min_test_size = int(max(2, min_test_size))

    if n_samples < (min_train_size + min_test_size):
        return []

    test_starts = np.linspace(
        min_train_size,
        n_samples - min_test_size,
        num=requested_splits,
        dtype=int,
    )
    test_starts = np.unique(test_starts)
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    for split_idx, test_start in enumerate(test_starts):
        test_end = (
            int(test_starts[split_idx + 1])
            if split_idx + 1 < len(test_starts)
            else n_samples
        )
        if (test_end - test_start) < min_test_size:
            test_end = min(n_samples, int(test_start + min_test_size))
        if test_start < min_train_size or (test_end - test_start) < min_test_size:
            continue
        train_idx = np.arange(int(test_start), dtype=int)
        test_idx = np.arange(int(test_start), int(test_end), dtype=int)
        if train_idx.size < min_train_size or test_idx.size < min_test_size:
            continue
        splits.append((train_idx, test_idx))

    return splits


def _fit_simple_linear_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[float, float]:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    x_mean = float(np.mean(x_train))
    y_mean = float(np.mean(y_train))
    x_centered = x_train - x_mean
    denom = float(np.sum(x_centered ** 2))
    if not np.isfinite(denom) or np.isclose(denom, 0.0):
        return y_mean, 0.0
    slope = float(np.sum(x_centered * (y_train - y_mean)) / denom)
    intercept = float(y_mean - (slope * x_mean))
    return intercept, slope


def _compute_lag_crossvalidation_table(
    values: np.ndarray,
    max_lag: int,
    cv_folds: int,
    min_train_size: int,
    min_test_size: int,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    rows: list[dict[str, float | int]] = []

    if values.size < 2:
        return pd.DataFrame(
            columns=[
                "lag",
                "n_pairs",
                "n_folds",
                "mean_cv_corr",
                "mean_cv_r2",
                "mean_cv_rmse",
            ]
        )

    for lag in range(1, int(max(1, max_lag)) + 1):
        if values.size <= lag:
            break

        x_source = np.asarray(values[:-lag], dtype=np.float64)
        y_target = np.asarray(values[lag:], dtype=np.float64)
        pair_mask = np.isfinite(x_source) & np.isfinite(y_target)
        x_source = x_source[pair_mask]
        y_target = y_target[pair_mask]
        n_pairs = int(y_target.size)
        splits = _build_time_series_cv_splits(
            n_samples=n_pairs,
            n_splits=int(cv_folds),
            min_train_size=int(min_train_size),
            min_test_size=int(min_test_size),
        )

        fold_corrs: list[float] = []
        fold_r2: list[float] = []
        fold_rmses: list[float] = []

        for train_idx, test_idx in splits:
            x_train = x_source[train_idx]
            y_train = y_target[train_idx]
            x_test = x_source[test_idx]
            y_test = y_target[test_idx]

            finite_train = np.isfinite(x_train) & np.isfinite(y_train)
            finite_test = np.isfinite(x_test) & np.isfinite(y_test)
            x_train = x_train[finite_train]
            y_train = y_train[finite_train]
            x_test = x_test[finite_test]
            y_test = y_test[finite_test]

            if x_train.size < 2 or y_test.size == 0:
                continue

            intercept, slope = _fit_simple_linear_regression(x_train, y_train)
            y_pred = intercept + (slope * x_test)
            residual = y_test - y_pred
            fold_rmses.append(float(np.sqrt(np.mean(residual ** 2))))

            if (
                y_test.size >= 2
                and not np.isclose(np.std(y_test), 0.0)
                and not np.isclose(np.std(y_pred), 0.0)
            ):
                fold_corrs.append(float(np.corrcoef(y_test, y_pred)[0, 1]))

            ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
            if y_test.size >= 2 and not np.isclose(ss_tot, 0.0):
                ss_res = float(np.sum(residual ** 2))
                fold_r2.append(float(1.0 - (ss_res / ss_tot)))

        rows.append(
            {
                "lag": int(lag),
                "n_pairs": n_pairs,
                "n_folds": int(len(fold_rmses)),
                "mean_cv_corr": (
                    float(np.mean(fold_corrs))
                    if len(fold_corrs) > 0
                    else np.nan
                ),
                "mean_cv_r2": float(np.mean(fold_r2)) if len(fold_r2) > 0 else np.nan,
                "mean_cv_rmse": (
                    float(np.mean(fold_rmses))
                    if len(fold_rmses) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def _compute_corr_limits(values: np.ndarray) -> tuple[float, float]:
    finite_values = np.asarray(values, dtype=np.float64)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return (-1.0, 1.0)
    lower = max(-1.0, float(np.min(finite_values)) - 0.08)
    upper = min(1.0, float(np.max(finite_values)) + 0.08)
    if np.isclose(lower, upper):
        lower = max(-1.0, lower - 0.1)
        upper = min(1.0, upper + 0.1)
    return lower, upper


def _make_session_pair_figure(
    n_sessions: int,
    *,
    sharex: bool,
    sharey: bool,
) -> tuple[plt.Figure, np.ndarray]:
    n_sessions = int(max(1, n_sessions))
    n_cols = 2 if n_sessions > 1 else 1
    n_rows = int(np.ceil(n_sessions / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.25 * n_cols, 4.4 * n_rows),
        sharex=sharex,
        sharey=sharey,
    )
    return fig, np.atleast_1d(axes).ravel()


def _extract_session_run_pair(
    subject_df: pd.DataFrame,
    ses: int,
) -> tuple[list[int], list[pd.DataFrame]]:
    session_df = subject_df.loc[subject_df["ses"] == int(ses)].copy()
    run_keys = sorted(session_df["run"].dropna().astype(int).unique().tolist())
    run_dfs = [
        session_df.loc[session_df["run"] == int(run)].sort_values("trial_index_run").copy()
        for run in run_keys
    ]
    return run_keys, run_dfs


def _align_pair_for_lag(
    values_a: np.ndarray,
    values_b: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    lag = int(lag)

    start_a = max(0, -lag)
    start_b = max(0, lag)
    overlap = min(values_a.size - start_a, values_b.size - start_b)
    if overlap <= 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    end_a = int(start_a + overlap)
    end_b = int(start_b + overlap)
    return values_a[start_a:end_a], values_b[start_b:end_b]


def _compute_runpair_crosscorrelation_table(
    values_a: np.ndarray,
    values_b: np.ndarray,
    max_lag: int,
    min_overlap: int,
) -> pd.DataFrame:
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    min_overlap = int(max(2, min_overlap))
    rows: list[dict[str, float | int]] = []

    if values_a.size == 0 or values_b.size == 0:
        return pd.DataFrame(columns=["lag", "n_overlap", "corr"])

    for lag in range(-int(max_lag), int(max_lag) + 1):
        aligned_a, aligned_b = _align_pair_for_lag(values_a, values_b, lag)
        pair_mask = np.isfinite(aligned_a) & np.isfinite(aligned_b)
        aligned_a = aligned_a[pair_mask]
        aligned_b = aligned_b[pair_mask]
        n_overlap = int(aligned_a.size)
        corr_value = np.nan
        if (
            n_overlap >= min_overlap
            and not np.isclose(np.std(aligned_a), 0.0)
            and not np.isclose(np.std(aligned_b), 0.0)
        ):
            corr_value = float(np.corrcoef(aligned_a, aligned_b)[0, 1])
        rows.append(
            {
                "lag": int(lag),
                "n_overlap": n_overlap,
                "corr": corr_value,
            }
        )

    return pd.DataFrame(rows)


def _compute_autocorrelation_table(
    values: np.ndarray,
    max_lag: int,
    min_overlap: int,
) -> pd.DataFrame:
    values = np.asarray(values, dtype=np.float64)
    min_overlap = int(max(2, min_overlap))
    rows: list[dict[str, float | int]] = []

    if values.size == 0:
        return pd.DataFrame(columns=["lag", "n_pairs", "autocorr"])

    for lag in range(1, int(max(1, max_lag)) + 1):
        if values.size <= lag:
            break
        x_values = np.asarray(values[:-lag], dtype=np.float64)
        y_values = np.asarray(values[lag:], dtype=np.float64)
        pair_mask = np.isfinite(x_values) & np.isfinite(y_values)
        x_values = x_values[pair_mask]
        y_values = y_values[pair_mask]
        n_pairs = int(x_values.size)
        autocorr_value = np.nan
        if n_pairs >= min_overlap:
            if (
                x_values.size >= 2
                and not np.isclose(np.std(x_values), 0.0)
                and not np.isclose(np.std(y_values), 0.0)
            ):
                autocorr_value = float(np.corrcoef(x_values, y_values)[0, 1])

        rows.append(
            {
                "lag": int(lag),
                "n_pairs": int(n_pairs),
                "autocorr": autocorr_value,
            }
        )

    return pd.DataFrame(rows)


def _plot_subject_trials(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    plot_kind: str,
) -> None:
    if plot_kind == "scatter":
        _plot_subject_consecutive_scatter(
            subject_df=subject_df,
            out_path=out_path,
            behavior_column=int(behavior_column),
            demean_within_run=bool(demean_within_run),
            reciprocal_behavior=bool(reciprocal_behavior),
            dpi=int(dpi),
        )
        return
    if plot_kind != "line":
        raise ValueError(f"Unsupported plot kind: {plot_kind}")

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=False)
    axes_flat = axes.ravel()

    x_max = int(subject_df["trial_index_run"].max())
    run_keys = _extract_run_keys(subject_df)
    if len(run_keys) > len(axes_flat):
        raise ValueError(
            f"Expected at most {len(axes_flat)} session-run panels per subject, found {len(run_keys)}."
        )
    session_color_map = _build_session_color_map(subject_df)

    for ax, (ses, run) in zip(axes_flat, run_keys):
        run_df = subject_df.loc[
            (subject_df["ses"] == ses) & (subject_df["run"] == run)
        ].sort_values("trial_index_run")
        if run_df.empty:
            ax.set_visible(False)
            continue

        x_values = run_df["trial_index_run"].to_numpy(dtype=np.float64)
        y_values = run_df["behavior_value"].to_numpy(dtype=np.float64)
        y_min, y_max = _compute_y_limits(y_values)
        color = session_color_map[int(ses)]

        ax.plot(
            x_values,
            y_values,
            linestyle="-",
            linewidth=1.45,
            alpha=0.86,
            color=color,
        )

        ax.set_xlim(1, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_title(f"ses-{ses} run-{run} (n={len(run_df)})", fontsize=11)

    for ax in axes_flat[len(run_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    value_label = _value_label(
        behavior_column=int(behavior_column),
        demean_within_run=bool(demean_within_run),
        reciprocal_behavior=bool(reciprocal_behavior),
    )
    for ax in axes[1, :]:
        if ax.get_visible():
            ax.set_xlabel("Trial number")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(value_label)

    fig.suptitle(
        f"{sub_tag}: behavior across kept trials by session-run (line plot)",
        fontsize=15,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_subject_combined_run_histograms(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    hist_bins: int,
) -> None:
    session_keys = _extract_session_keys(subject_df)
    fig, axes_flat = _make_session_pair_figure(
        len(session_keys),
        sharex=False,
        sharey=False,
    )
    session_color_map = _build_session_color_map(subject_df)

    for ax, ses in zip(axes_flat, session_keys):
        run_keys, run_dfs = _extract_session_run_pair(subject_df, ses)
        nonempty_run_dfs = [run_df for run_df in run_dfs if not run_df.empty]
        if len(nonempty_run_dfs) == 0:
            ax.text(
                0.5,
                0.5,
                "No finite trials",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(f"ses-{ses}", fontsize=11)
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.7)
            continue

        combined_values = np.concatenate(
            [
                run_df["behavior_value"].to_numpy(dtype=np.float64)
                for run_df in nonempty_run_dfs
            ],
            axis=0,
        )
        bin_edges = _compute_histogram_bin_edges(combined_values, bins=int(hist_bins))
        color = session_color_map[int(ses)]
        run_label = " + ".join(f"run-{run}" for run in run_keys)

        ax.hist(
            combined_values,
            bins=bin_edges,
            density=True,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.axvline(
            float(np.mean(combined_values)),
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.85,
        )
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.7)
        ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
        ax.set_title(f"ses-{ses}: {run_label} (n={combined_values.size})", fontsize=11)
        ax.set_xlabel(
            _value_label(
                behavior_column=int(behavior_column),
                demean_within_run=bool(demean_within_run),
                reciprocal_behavior=bool(reciprocal_behavior),
            )
        )
        ax.set_ylabel("PDF")

    for ax in axes_flat[len(session_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    fig.suptitle(
        f"{sub_tag}: within-session combined-run behavior histograms",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_subject_runpair_histograms(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    hist_bins: int,
) -> None:
    session_keys = _extract_session_keys(subject_df)
    fig, axes_flat = _make_session_pair_figure(
        len(session_keys),
        sharex=False,
        sharey=False,
    )
    run_color_map = _build_run_color_map(subject_df)

    for ax, ses in zip(axes_flat, session_keys):
        run_keys, run_dfs = _extract_session_run_pair(subject_df, ses)
        if len(run_keys) < 2:
            ax.text(
                0.5,
                0.5,
                "Need >=2 runs\nfor comparison",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(f"ses-{ses}", fontsize=11)
            ax.grid(True, axis="y", alpha=0.18, linewidth=0.7)
            continue

        paired_runs = run_keys[:2]
        paired_dfs = run_dfs[:2]
        combined_values = np.concatenate(
            [
                run_df["behavior_value"].to_numpy(dtype=np.float64)
                for run_df in paired_dfs
            ]
        )
        bin_edges = _compute_histogram_bin_edges(combined_values, bins=int(hist_bins))

        for run, run_df in zip(paired_runs, paired_dfs):
            run_values = run_df["behavior_value"].to_numpy(dtype=np.float64)
            color = run_color_map[int(run)]
            ax.hist(
                run_values,
                bins=bin_edges,
                density=True,
                alpha=0.42,
                color=color,
                edgecolor="white",
                linewidth=0.6,
                label=f"run-{run} (n={len(run_df)})",
            )
            ax.axvline(
                float(np.mean(run_values)),
                color=color,
                linestyle="--",
                linewidth=1.15,
                alpha=0.9,
            )

        ax.grid(True, axis="y", alpha=0.18, linewidth=0.7)
        ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))
        ax.set_title(
            f"ses-{ses}: run-{paired_runs[0]} vs run-{paired_runs[1]}",
            fontsize=11,
        )
        ax.legend(frameon=False, fontsize=9, loc="best")
        ax.set_xlabel(
            _value_label(
                behavior_column=int(behavior_column),
                demean_within_run=bool(demean_within_run),
                reciprocal_behavior=bool(reciprocal_behavior),
            )
        )
        ax.set_ylabel("PDF")

    for ax in axes_flat[len(session_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    fig.suptitle(
        f"{sub_tag}: within-session run-pair behavior histograms",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_subject_histograms(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    hist_bins: int,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=False, sharey=False)
    axes_flat = axes.ravel()

    run_keys = _extract_run_keys(subject_df)
    if len(run_keys) > len(axes_flat):
        raise ValueError(
            f"Expected at most {len(axes_flat)} session-run panels per subject, found {len(run_keys)}."
        )
    session_color_map = _build_session_color_map(subject_df)

    for ax, (ses, run) in zip(axes_flat, run_keys):
        run_df = subject_df.loc[
            (subject_df["ses"] == ses) & (subject_df["run"] == run)
        ].sort_values("trial_index_run")
        if run_df.empty:
            ax.set_visible(False)
            continue

        y_values = run_df["behavior_value"].to_numpy(dtype=np.float64)
        color = session_color_map[int(ses)]
        bin_edges = _compute_histogram_bin_edges(y_values, bins=int(hist_bins))

        ax.hist(
            y_values,
            bins=bin_edges,
            density=True,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.6,
        )
        run_mean = float(np.mean(y_values))
        ax.axvline(run_mean, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.grid(True, axis="y", alpha=0.18, linewidth=0.7)
        ax.set_title(f"ses-{ses} run-{run} (n={len(run_df)})", fontsize=11)
        ax.set_xlim(float(bin_edges[0]), float(bin_edges[-1]))

    for ax in axes_flat[len(run_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    value_label = _value_label(
        behavior_column=int(behavior_column),
        demean_within_run=bool(demean_within_run),
        reciprocal_behavior=bool(reciprocal_behavior),
    )
    for ax in axes[1, :]:
        if ax.get_visible():
            ax.set_xlabel(value_label)
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("PDF")

    fig.suptitle(
        f"{sub_tag}: behavior PDF histogram across kept trials by session-run",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _plot_subject_lag_crossvalidation(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    cv_max_lag: int,
    cv_folds: int,
    cv_min_train_size: int,
    cv_min_test_size: int,
) -> pd.DataFrame:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    run_keys = _extract_run_keys(subject_df)
    if len(run_keys) > len(axes_flat):
        raise ValueError(
            f"Expected at most {len(axes_flat)} session-run panels per subject, found {len(run_keys)}."
        )
    session_color_map = _build_session_color_map(subject_df)

    lag_tables: list[pd.DataFrame] = []
    for ses, run in run_keys:
        run_df = subject_df.loc[
            (subject_df["ses"] == ses) & (subject_df["run"] == run)
        ].sort_values("trial_index_run")
        lag_table = _compute_lag_crossvalidation_table(
            values=run_df["behavior_value"].to_numpy(dtype=np.float64),
            max_lag=int(cv_max_lag),
            cv_folds=int(cv_folds),
            min_train_size=int(cv_min_train_size),
            min_test_size=int(cv_min_test_size),
        )
        if not lag_table.empty:
            lag_table = lag_table.copy()
            lag_table["sub_tag"] = str(subject_df["sub_tag"].iloc[0])
            lag_table["ses"] = int(ses)
            lag_table["run"] = int(run)
        lag_tables.append(lag_table)

    finite_scores = []
    for lag_table in lag_tables:
        if lag_table.empty:
            continue
        finite_scores.extend(
            lag_table["mean_cv_corr"].to_numpy(dtype=np.float64).tolist()
        )
    y_min, y_max = _compute_corr_limits(np.asarray(finite_scores, dtype=np.float64))

    for ax, (ses, run), lag_table in zip(axes_flat, run_keys, lag_tables):
        run_df = subject_df.loc[
            (subject_df["ses"] == ses) & (subject_df["run"] == run)
        ].sort_values("trial_index_run")
        color = session_color_map[int(ses)]

        if lag_table.empty or not np.any(np.isfinite(lag_table["mean_cv_corr"].to_numpy(dtype=np.float64))):
            ax.text(
                0.5,
                0.5,
                "Insufficient trials\nfor lag CV",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.set_title(f"ses-{ses} run-{run} (n={len(run_df)})", fontsize=11)
            ax.set_xlim(1, max(1, int(cv_max_lag)))
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.18, linewidth=0.7)
            continue

        lag_values = lag_table["lag"].to_numpy(dtype=np.int64)
        corr_values = lag_table["mean_cv_corr"].to_numpy(dtype=np.float64)
        ax.plot(
            lag_values,
            corr_values,
            marker="o",
            markersize=4.2,
            linewidth=1.4,
            alpha=0.9,
            color=color,
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_xlim(1, max(1, int(cv_max_lag)))
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_title(f"ses-{ses} run-{run} (n={len(run_df)})", fontsize=11)

    for ax in axes_flat[len(run_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    value_label = _value_label(
        behavior_column=int(behavior_column),
        demean_within_run=bool(demean_within_run),
        reciprocal_behavior=bool(reciprocal_behavior),
    )
    for ax in axes[1, :]:
        if ax.get_visible():
            ax.set_xlabel("Lag")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel("Mean held-out corr")

    fig.suptitle(
        f"{sub_tag}: lagged self-prediction cross-validation for {value_label}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    if len(lag_tables) == 0:
        return pd.DataFrame()
    lag_tables = [lag_table for lag_table in lag_tables if not lag_table.empty]
    if len(lag_tables) == 0:
        return pd.DataFrame()
    return pd.concat(lag_tables, ignore_index=True)


def _plot_subject_combined_run_autocorrelation(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    max_lag: int,
    min_overlap: int,
) -> pd.DataFrame:
    session_keys = _extract_session_keys(subject_df)
    fig, axes_flat = _make_session_pair_figure(
        len(session_keys),
        sharex=True,
        sharey=True,
    )
    run_color_map = _build_run_color_map(subject_df)

    session_autocorr_tables: list[list[tuple[int, int, pd.DataFrame]]] = []
    finite_scores: list[float] = []
    for ses in session_keys:
        run_keys, run_dfs = _extract_session_run_pair(subject_df, ses)
        run_tables: list[tuple[int, int, pd.DataFrame]] = []
        for run, run_df in zip(run_keys, run_dfs):
            autocorr_table = _compute_autocorrelation_table(
                values=run_df["behavior_value"].to_numpy(dtype=np.float64),
                max_lag=int(max_lag),
                min_overlap=int(min_overlap),
            )
            if not autocorr_table.empty:
                autocorr_table = autocorr_table.copy()
                autocorr_table["sub_tag"] = str(subject_df["sub_tag"].iloc[0])
                autocorr_table["ses"] = int(ses)
                autocorr_table["run"] = int(run)
                finite_scores.extend(
                    autocorr_table.loc[np.isfinite(autocorr_table["autocorr"]), "autocorr"]
                    .to_numpy(dtype=np.float64)
                    .tolist()
                )
            run_tables.append((int(run), int(len(run_df)), autocorr_table))
        session_autocorr_tables.append(run_tables)

    y_min, y_max = _compute_corr_limits(np.asarray(finite_scores, dtype=np.float64))

    autocorr_tables: list[pd.DataFrame] = []
    for ax, ses, run_tables in zip(axes_flat, session_keys, session_autocorr_tables):
        run_keys, run_dfs = _extract_session_run_pair(subject_df, ses)
        total_points = int(sum(len(run_df) for run_df in run_dfs))
        run_label = (
            " + ".join(f"run-{run}" for run in run_keys) if len(run_keys) > 0 else "no runs"
        )
        has_finite_curve = any(
            not table.empty
            and np.any(np.isfinite(table["autocorr"].to_numpy(dtype=np.float64)))
            for _, _, table in run_tables
        )

        if not has_finite_curve:
            ax.text(
                0.5,
                0.5,
                "Insufficient trials\nfor autocorrelation",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.set_title(f"ses-{ses}: {run_label} (n={total_points})", fontsize=11)
            ax.set_xlim(1, max(1, int(max_lag)))
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.18, linewidth=0.7)
            continue

        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.set_xlim(1, max(1, int(max_lag)))
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_title(f"ses-{ses}: {run_label} (n={total_points})", fontsize=11)
        annotation_lines: list[str] = []
        for run, run_n, autocorr_table in run_tables:
            if autocorr_table.empty:
                continue
            autocorr_tables.append(autocorr_table)
            lag_values = autocorr_table["lag"].to_numpy(dtype=np.int64)
            corr_values = autocorr_table["autocorr"].to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(corr_values)
            if not np.any(finite_mask):
                continue
            ax.plot(
                lag_values,
                corr_values,
                marker="o",
                markersize=4.0,
                linewidth=1.3,
                alpha=0.9,
                color=run_color_map[int(run)],
                label=f"run-{run} (n={run_n})",
            )
            finite_lags = lag_values[finite_mask]
            finite_corrs = corr_values[finite_mask]
            max_idx = int(np.argmax(np.abs(finite_corrs)))
            peak_lag = int(finite_lags[max_idx])
            peak_corr = float(finite_corrs[max_idx])
            annotation_lines.append(f"run-{run}: max |r|={peak_corr:.2f} @ lag {peak_lag}")
        if len(annotation_lines) > 0:
            ax.text(
                0.02,
                0.96,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.8,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 1.8,
                },
            )
            ax.legend(frameon=False, fontsize=8.8, loc="lower left")

    for ax in axes_flat[len(session_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    value_label = _value_label(
        behavior_column=int(behavior_column),
        demean_within_run=bool(demean_within_run),
        reciprocal_behavior=bool(reciprocal_behavior),
    )
    for ax in axes_flat[: len(session_keys)]:
        if ax.get_visible():
            ax.set_xlabel("Lag")
            ax.set_ylabel("Autocorr")

    fig.suptitle(
        f"{sub_tag}: within-session run-separated autocorrelation for {value_label}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    if len(autocorr_tables) == 0:
        return pd.DataFrame()
    return pd.concat(autocorr_tables, ignore_index=True)


def _plot_subject_runpair_crosscorrelation(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    reciprocal_behavior: bool,
    dpi: int,
    runpair_max_lag: int,
    runpair_min_overlap: int,
) -> pd.DataFrame:
    session_keys = _extract_session_keys(subject_df)
    fig, axes_flat = _make_session_pair_figure(
        len(session_keys),
        sharex=True,
        sharey=True,
    )
    session_color_map = _build_session_color_map(subject_df)

    pair_tables: list[pd.DataFrame] = []
    finite_scores: list[float] = []
    for ses in session_keys:
        run_keys, run_dfs = _extract_session_run_pair(subject_df, ses)
        if len(run_keys) < 2:
            pair_tables.append(pd.DataFrame())
            continue

        paired_runs = run_keys[:2]
        paired_dfs = run_dfs[:2]
        pair_table = _compute_runpair_crosscorrelation_table(
            values_a=paired_dfs[0]["behavior_value"].to_numpy(dtype=np.float64),
            values_b=paired_dfs[1]["behavior_value"].to_numpy(dtype=np.float64),
            max_lag=int(runpair_max_lag),
            min_overlap=int(runpair_min_overlap),
        )
        if not pair_table.empty:
            pair_table = pair_table.copy()
            pair_table["sub_tag"] = str(subject_df["sub_tag"].iloc[0])
            pair_table["ses"] = int(ses)
            pair_table["run_a"] = int(paired_runs[0])
            pair_table["run_b"] = int(paired_runs[1])
            finite_scores.extend(
                pair_table.loc[np.isfinite(pair_table["corr"]), "corr"]
                .to_numpy(dtype=np.float64)
                .tolist()
            )
        pair_tables.append(pair_table)

    y_min, y_max = _compute_corr_limits(np.asarray(finite_scores, dtype=np.float64))

    for ax, ses, pair_table in zip(axes_flat, session_keys, pair_tables):
        run_keys, run_dfs = _extract_session_run_pair(subject_df, ses)
        if len(run_keys) < 2:
            ax.text(
                0.5,
                0.5,
                "Need >=2 runs\nfor comparison",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(f"ses-{ses}", fontsize=11)
            ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.set_xlim(-int(runpair_max_lag), int(runpair_max_lag))
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.18, linewidth=0.7)
            continue

        paired_runs = run_keys[:2]
        if pair_table.empty or not np.any(np.isfinite(pair_table["corr"].to_numpy(dtype=np.float64))):
            ax.text(
                0.5,
                0.5,
                "Insufficient overlap\nfor correlation",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
            ax.axvline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.45)
            ax.set_title(
                f"ses-{ses}: run-{paired_runs[0]} vs run-{paired_runs[1]}",
                fontsize=11,
            )
            ax.set_xlim(-int(runpair_max_lag), int(runpair_max_lag))
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.18, linewidth=0.7)
            continue

        lag_values = pair_table["lag"].to_numpy(dtype=np.int64)
        corr_values = pair_table["corr"].to_numpy(dtype=np.float64)
        ax.plot(
            lag_values,
            corr_values,
            marker="o",
            markersize=4.2,
            linewidth=1.4,
            alpha=0.9,
            color=session_color_map[int(ses)],
        )
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.6)
        ax.axvline(0.0, color="black", linestyle=":", linewidth=0.9, alpha=0.45)
        ax.set_xlim(-int(runpair_max_lag), int(runpair_max_lag))
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_title(
            f"ses-{ses}: run-{paired_runs[0]} vs run-{paired_runs[1]}",
            fontsize=11,
        )
        max_idx = int(np.nanargmax(np.abs(corr_values)))
        max_lag = int(lag_values[max_idx])
        max_corr = float(corr_values[max_idx])
        ax.text(
            0.02,
            0.96,
            f"max |r|={max_corr:.2f} @ lag {max_lag}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 1.8,
            },
        )

    for ax in axes_flat[len(session_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    value_label = _value_label(
        behavior_column=int(behavior_column),
        demean_within_run=bool(demean_within_run),
        reciprocal_behavior=bool(reciprocal_behavior),
    )
    for ax in axes_flat[: len(session_keys)]:
        if ax.get_visible():
            ax.set_xlabel("Lag (positive: run-B later than run-A)")
            ax.set_ylabel("Pearson corr")

    fig.suptitle(
        f"{sub_tag}: within-session run-pair cross-correlation for {value_label}",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)

    pair_tables = [pair_table for pair_table in pair_tables if not pair_table.empty]
    if len(pair_tables) == 0:
        return pd.DataFrame()
    return pd.concat(pair_tables, ignore_index=True)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    trial_keep_root = Path(args.trial_keep_root).expanduser().resolve()
    behavior_root = Path(args.behavior_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    default_out_dir = DEFAULT_OUT_DIR.expanduser().resolve()
    if bool(args.reciprocal_behavior) and out_dir == default_out_dir:
        out_dir = out_dir.with_name(f"{out_dir.name}_reciprocal")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path, sep="\t")
    trial_df = _build_subject_trial_table(
        manifest_df=manifest_df,
        trial_keep_root=trial_keep_root,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
        demean_within_run=bool(args.demean_within_run),
        reciprocal_behavior=bool(args.reciprocal_behavior),
    )

    if trial_df.empty:
        raise ValueError("No finite kept behavior trials were found.")

    lag_metric_tables: list[pd.DataFrame] = []
    runpair_corr_tables: list[pd.DataFrame] = []
    combined_run_autocorr_tables: list[pd.DataFrame] = []
    n_subjects = 0
    for sub_tag, subject_df in trial_df.groupby("sub_tag", sort=False):
        subject_df = subject_df.sort_values(["ses", "run", "trial_index_run"]).copy()
        line_out_path = out_dir / f"{sub_tag}_behavior_trial_lineplot_panel.png"
        scatter_out_path = out_dir / f"{sub_tag}_behavior_trial_scatter_panel.png"
        hist_out_path = out_dir / f"{sub_tag}_behavior_trial_histogram_panel.png"
        lag_cv_out_path = out_dir / f"{sub_tag}_behavior_trial_lag_crossvalidation_panel.png"
        runpair_hist_out_path = out_dir / f"{sub_tag}_behavior_trial_runpair_histogram_panel.png"
        runpair_corr_out_path = out_dir / f"{sub_tag}_behavior_trial_runpair_crosscorrelation_panel.png"
        combined_hist_out_path = out_dir / f"{sub_tag}_behavior_trial_combined_run_histogram_panel.png"
        combined_autocorr_out_path = out_dir / f"{sub_tag}_behavior_trial_combined_run_autocorrelation_panel.png"
        _plot_subject_trials(
            subject_df=subject_df,
            out_path=line_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            plot_kind="line",
        )
        _plot_subject_trials(
            subject_df=subject_df,
            out_path=scatter_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            plot_kind="scatter",
        )
        _plot_subject_histograms(
            subject_df=subject_df,
            out_path=hist_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            hist_bins=int(args.hist_bins),
        )
        _plot_subject_combined_run_histograms(
            subject_df=subject_df,
            out_path=combined_hist_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            hist_bins=int(args.hist_bins),
        )
        _plot_subject_runpair_histograms(
            subject_df=subject_df,
            out_path=runpair_hist_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            hist_bins=int(args.hist_bins),
        )
        lag_metric_table = _plot_subject_lag_crossvalidation(
            subject_df=subject_df,
            out_path=lag_cv_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            cv_max_lag=int(args.cv_max_lag),
            cv_folds=int(args.cv_folds),
            cv_min_train_size=int(args.cv_min_train_size),
            cv_min_test_size=int(args.cv_min_test_size),
        )
        if not lag_metric_table.empty:
            lag_metric_tables.append(lag_metric_table)
        combined_autocorr_table = _plot_subject_combined_run_autocorrelation(
            subject_df=subject_df,
            out_path=combined_autocorr_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            max_lag=int(args.combined_autocorr_max_lag),
            min_overlap=int(args.combined_autocorr_min_overlap),
        )
        if not combined_autocorr_table.empty:
            combined_run_autocorr_tables.append(combined_autocorr_table)
        runpair_corr_table = _plot_subject_runpair_crosscorrelation(
            subject_df=subject_df,
            out_path=runpair_corr_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            reciprocal_behavior=bool(args.reciprocal_behavior),
            dpi=int(args.dpi),
            runpair_max_lag=int(args.runpair_max_lag),
            runpair_min_overlap=int(args.runpair_min_overlap),
        )
        if not runpair_corr_table.empty:
            runpair_corr_tables.append(runpair_corr_table)
        n_subjects += 1

    if len(lag_metric_tables) > 0:
        lag_metrics_df = pd.concat(lag_metric_tables, ignore_index=True)
        lag_metrics_path = out_dir / "behavior_trial_lag_crossvalidation.csv"
        lag_metrics_df.to_csv(lag_metrics_path, index=False)
    if len(runpair_corr_tables) > 0:
        runpair_corr_df = pd.concat(runpair_corr_tables, ignore_index=True)
        runpair_corr_path = out_dir / "behavior_trial_runpair_crosscorrelation.csv"
        runpair_corr_df.to_csv(runpair_corr_path, index=False)
    if len(combined_run_autocorr_tables) > 0:
        combined_run_autocorr_df = pd.concat(combined_run_autocorr_tables, ignore_index=True)
        combined_run_autocorr_path = out_dir / "behavior_trial_combined_run_autocorrelation.csv"
        combined_run_autocorr_df.to_csv(combined_run_autocorr_path, index=False)

    print(
        f"Saved {n_subjects} line figures, {n_subjects} lag-1 scatter figures, "
        f"{n_subjects} histogram figures, {n_subjects} lag cross-validation figures, "
        f"{n_subjects} run-pair histogram figures, {n_subjects} run-pair cross-correlation figures, "
        f"{n_subjects} combined-run histogram figures, and {n_subjects} run-separated autocorrelation figures to: {out_dir}"
    )


if __name__ == "__main__":
    main()
