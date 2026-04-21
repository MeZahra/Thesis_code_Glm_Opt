#!/usr/bin/env python3
"""Plot kept behavior trial values for each subject as session-run subplots.

The script uses the same behavior-file and trial_keep convention as
motor_brain_com.py, but instead of plotting (trial i, trial i+1) pairs, it
plots each kept trial value against trial number. It saves both line and
scatter versions in a 2x2 panel layout with one subplot per session-run.
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
            "For each subject, plot kept behavior trial values against trial "
            "number in a 2x2 session-run panel using the same behavior-file "
            "and trial_keep convention as motor_brain_com.py."
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
    return parser.parse_args()


def _build_subject_trial_table(
    manifest_df: pd.DataFrame,
    trial_keep_root: Path,
    behavior_root: Path,
    behavior_column: int,
    demean_within_run: bool,
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


def _plot_subject_trials(
    subject_df: pd.DataFrame,
    out_path: Path,
    behavior_column: int,
    demean_within_run: bool,
    dpi: int,
    plot_kind: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    session_keys = sorted(subject_df["ses"].dropna().astype(int).unique().tolist())
    cmap = plt.get_cmap("tab10")

    y_all = subject_df["behavior_value"].to_numpy(dtype=np.float64)
    y_min, y_max = _compute_y_limits(y_all)
    x_max = int(subject_df["trial_index_run"].max())

    run_keys = (
        subject_df.loc[:, ["ses", "run"]]
        .drop_duplicates()
        .sort_values(["ses", "run"])
        .itertuples(index=False)
    )
    run_keys = [(int(row.ses), int(row.run)) for row in run_keys]
    if len(run_keys) > len(axes_flat):
        raise ValueError(
            f"Expected at most {len(axes_flat)} session-run panels per subject, found {len(run_keys)}."
        )
    session_color_map = {
        int(ses): cmap(color_idx % 10) for color_idx, ses in enumerate(session_keys)
    }

    for ax, (ses, run) in zip(axes_flat, run_keys):
        run_df = subject_df.loc[
            (subject_df["ses"] == ses) & (subject_df["run"] == run)
        ].sort_values("trial_index_run")
        if run_df.empty:
            ax.set_visible(False)
            continue

        x_values = run_df["trial_index_run"].to_numpy(dtype=np.float64)
        y_values = run_df["behavior_value"].to_numpy(dtype=np.float64)
        color = session_color_map[int(ses)]

        if plot_kind == "line":
            ax.plot(
                x_values,
                y_values,
                linestyle="-",
                linewidth=1.45,
                alpha=0.86,
                color=color,
            )
        elif plot_kind == "scatter":
            ax.scatter(
                x_values,
                y_values,
                s=28,
                alpha=0.86,
                color=color,
                edgecolors="none",
            )
        else:
            raise ValueError(f"Unsupported plot kind: {plot_kind}")

        ax.set_xlim(1, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.18, linewidth=0.7)
        ax.set_title(f"ses-{ses} run-{run} (n={len(run_df)})", fontsize=11)

    for ax in axes_flat[len(run_keys) :]:
        ax.set_visible(False)

    sub_tag = str(subject_df["sub_tag"].iloc[0])
    value_label = f"Behaviour column {int(behavior_column)}"
    if demean_within_run:
        value_label += " (demeaned within run)"
    for ax in axes[1, :]:
        if ax.get_visible():
            ax.set_xlabel("Trial number")
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(value_label)

    figure_label = "line plot" if plot_kind == "line" else "scatter plot"
    fig.suptitle(
        f"{sub_tag}: behavior across kept trials by session-run ({figure_label})",
        fontsize=15,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    manifest_path = Path(args.manifest_path).expanduser().resolve()
    trial_keep_root = Path(args.trial_keep_root).expanduser().resolve()
    behavior_root = Path(args.behavior_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path, sep="\t")
    trial_df = _build_subject_trial_table(
        manifest_df=manifest_df,
        trial_keep_root=trial_keep_root,
        behavior_root=behavior_root,
        behavior_column=int(args.behavior_column),
        demean_within_run=bool(args.demean_within_run),
    )

    if trial_df.empty:
        raise ValueError("No finite kept behavior trials were found.")

    n_figures = 0
    for sub_tag, subject_df in trial_df.groupby("sub_tag", sort=False):
        subject_df = subject_df.sort_values(["ses", "run", "trial_index_run"]).copy()
        line_out_path = out_dir / f"{sub_tag}_behavior_trial_lineplot_panel.png"
        scatter_out_path = out_dir / f"{sub_tag}_behavior_trial_scatter_panel.png"
        _plot_subject_trials(
            subject_df=subject_df,
            out_path=line_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            dpi=int(args.dpi),
            plot_kind="line",
        )
        _plot_subject_trials(
            subject_df=subject_df,
            out_path=scatter_out_path,
            behavior_column=int(args.behavior_column),
            demean_within_run=bool(args.demean_within_run),
            dpi=int(args.dpi),
            plot_kind="scatter",
        )
        n_figures += 1

    print(f"Saved {n_figures} line figures and {n_figures} scatter figures to: {out_dir}")


if __name__ == "__main__":
    main()
