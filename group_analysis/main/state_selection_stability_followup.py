#!/usr/bin/env python3
"""State-stratified selected-vs-control voxel-stability follow-up.

This script reuses the voxel-level stability definitions from prove_hypo.py and
tests whether the selected voxel set is preferentially stable in the ON state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from group_analysis.main import prove_hypo as ph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute OFF/ON selected-vs-control voxel stability summaries using the "
            "same metrics as prove_hypo.py, then test the state-by-selection interaction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--selected-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ containing the selected voxel indices.",
    )
    parser.add_argument(
        "--anat-path",
        type=Path,
        default=Path("results/connectivity/tmp/data/MNI152_T1_2mm_brain.nii.gz"),
        help="MNI anatomy used to define the brain and motor voxel pools.",
    )
    parser.add_argument(
        "--motor-mask-path",
        type=Path,
        default=None,
        help="Optional NIfTI mask for motor areas. Non-zero values define the control pool.",
    )
    parser.add_argument(
        "--motor-label-patterns",
        type=str,
        default=",".join(ph.DEFAULT_MOTOR_LABEL_PATTERNS),
        help="Comma-separated motor-region label substrings used when building atlas-based motor mask.",
    )
    parser.add_argument(
        "--motor-atlas-cache-dir",
        type=Path,
        default=Path("results/connectivity/atlas_cache"),
        help="Cache directory for Harvard-Oxford atlas downloads.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv"),
        help="Group-concat manifest with cleaned_beta paths and trial_keep paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prove_hypothesis/state_selection_stability_followup"),
        help="Directory for CSV/JSON/figure outputs.",
    )
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=20000,
        help="Number of flat voxels to process at once from each 4D beta volume.",
    )
    return parser.parse_args()


def _one_sided_p_from_t(two_sided_p: float, t_stat: float, direction: str) -> float | None:
    if two_sided_p is None or t_stat is None or not math.isfinite(two_sided_p) or not math.isfinite(t_stat):
        return None
    half = float(two_sided_p) / 2.0
    if direction == "less":
        return half if float(t_stat) < 0 else 1.0 - half
    if direction == "greater":
        return half if float(t_stat) > 0 else 1.0 - half
    raise ValueError(f"Unknown direction: {direction}")


def _safe_float(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _fmt_signed(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):+.{digits}f}"


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "nan"
    return f"{float(value):.{digits}f}"


def _sem(values: pd.Series) -> float:
    arr = values.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 1:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def _fit_mixedlm(long_df: pd.DataFrame) -> dict[str, object]:
    data = long_df.copy()
    data["state"] = pd.Categorical(data["state"], categories=["off", "on"], ordered=True)
    data["selection"] = pd.Categorical(data["selection"], categories=["control", "selected"], ordered=True)

    formula = "value ~ C(state, Treatment(reference='off')) * C(selection, Treatment(reference='control'))"
    term_state = "C(state, Treatment(reference='off'))[T.on]"
    term_selection = "C(selection, Treatment(reference='control'))[T.selected]"
    term_interaction = (
        "C(state, Treatment(reference='off'))[T.on]:"
        "C(selection, Treatment(reference='control'))[T.selected]"
    )

    fit_errors: list[str] = []
    for method in ("lbfgs", "powell"):
        try:
            model = smf.mixedlm(formula, data=data, groups=data["subject"])
            result = model.fit(reml=False, method=method, disp=False)
            params = result.params
            pvalues = result.pvalues
            bse = result.bse
            return {
                "fit_method": method,
                "converged": bool(getattr(result, "converged", True)),
                "n_obs": int(data.shape[0]),
                "n_subjects": int(data["subject"].nunique()),
                "intercept": _safe_float(params.get("Intercept")),
                "coef_state_on": _safe_float(params.get(term_state)),
                "coef_selection_selected": _safe_float(params.get(term_selection)),
                "coef_interaction": _safe_float(params.get(term_interaction)),
                "se_state_on": _safe_float(bse.get(term_state)),
                "se_selection_selected": _safe_float(bse.get(term_selection)),
                "se_interaction": _safe_float(bse.get(term_interaction)),
                "p_state_on_two_sided": _safe_float(pvalues.get(term_state)),
                "p_selection_selected_two_sided": _safe_float(pvalues.get(term_selection)),
                "p_interaction_two_sided": _safe_float(pvalues.get(term_interaction)),
                "p_interaction_one_sided_less": _one_sided_p_from_t(
                    _safe_float(pvalues.get(term_interaction)),
                    _safe_float(params.get(term_interaction)),
                    "less",
                ),
            }
        except Exception as exc:  # pragma: no cover - fallback path depends on numeric data.
            fit_errors.append(f"{method}: {exc}")

    return {
        "fit_method": None,
        "converged": False,
        "n_obs": int(data.shape[0]),
        "n_subjects": int(data["subject"].nunique()),
        "fit_errors": fit_errors,
    }


def _paired_state_summary(wide_df: pd.DataFrame, metric_name: str) -> dict[str, object]:
    sel_off = wide_df[f"selected_{metric_name}_off"]
    sel_on = wide_df[f"selected_{metric_name}_on"]
    ctl_off = wide_df[f"control_{metric_name}_off"]
    ctl_on = wide_df[f"control_{metric_name}_on"]

    d_sel = sel_on - sel_off
    d_ctl = ctl_on - ctl_off
    d_interaction = d_sel - d_ctl

    t_sel = stats.ttest_rel(sel_on, sel_off, nan_policy="omit")
    t_ctl = stats.ttest_rel(ctl_on, ctl_off, nan_policy="omit")
    t_interaction = stats.ttest_rel(d_sel, d_ctl, nan_policy="omit")

    try:
        w_sel = stats.wilcoxon(d_sel)
        wilcoxon_sel_stat = float(w_sel.statistic)
        wilcoxon_sel_p = float(w_sel.pvalue)
    except Exception:
        wilcoxon_sel_stat = np.nan
        wilcoxon_sel_p = np.nan

    try:
        w_ctl = stats.wilcoxon(d_ctl)
        wilcoxon_ctl_stat = float(w_ctl.statistic)
        wilcoxon_ctl_p = float(w_ctl.pvalue)
    except Exception:
        wilcoxon_ctl_stat = np.nan
        wilcoxon_ctl_p = np.nan

    try:
        w_interaction = stats.wilcoxon(d_interaction)
        wilcoxon_interaction_stat = float(w_interaction.statistic)
        wilcoxon_interaction_p = float(w_interaction.pvalue)
    except Exception:
        wilcoxon_interaction_stat = np.nan
        wilcoxon_interaction_p = np.nan

    return {
        "n_subjects_paired": int(wide_df.shape[0]),
        "selected_off_mean": float(sel_off.mean()),
        "selected_on_mean": float(sel_on.mean()),
        "control_off_mean": float(ctl_off.mean()),
        "control_on_mean": float(ctl_on.mean()),
        "D_sel_mean_on_minus_off": float(d_sel.mean()),
        "D_ctl_mean_on_minus_off": float(d_ctl.mean()),
        "interaction_mean_D_sel_minus_D_ctl": float(d_interaction.mean()),
        "selected_state_t": _safe_float(t_sel.statistic),
        "selected_state_p_two_sided": _safe_float(t_sel.pvalue),
        "selected_state_p_one_sided_less": _one_sided_p_from_t(
            _safe_float(t_sel.pvalue),
            _safe_float(t_sel.statistic),
            "less",
        ),
        "control_state_t": _safe_float(t_ctl.statistic),
        "control_state_p_two_sided": _safe_float(t_ctl.pvalue),
        "interaction_t": _safe_float(t_interaction.statistic),
        "interaction_p_two_sided": _safe_float(t_interaction.pvalue),
        "interaction_p_one_sided_less": _one_sided_p_from_t(
            _safe_float(t_interaction.pvalue),
            _safe_float(t_interaction.statistic),
            "less",
        ),
        "wilcoxon_selected_state_stat": _safe_float(wilcoxon_sel_stat),
        "wilcoxon_selected_state_p_two_sided": _safe_float(wilcoxon_sel_p),
        "wilcoxon_control_state_stat": _safe_float(wilcoxon_ctl_stat),
        "wilcoxon_control_state_p_two_sided": _safe_float(wilcoxon_ctl_p),
        "wilcoxon_interaction_stat": _safe_float(wilcoxon_interaction_stat),
        "wilcoxon_interaction_p_two_sided": _safe_float(wilcoxon_interaction_p),
        "claim_supports_on_preferential_stability": bool((float(d_sel.mean()) < 0.0) and (float(d_interaction.mean()) < 0.0)),
    }


def _plot_subject_means(
    long_df: pd.DataFrame,
    metric_label: str,
    figure_path: Path,
    paired_summary: dict[str, object],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2), sharey=True)
    colors = {"control": "#3d6f8e", "selected": "#d65a6f"}
    state_order = ["off", "on"]
    x_positions = np.array([0.0, 1.0], dtype=np.float64)
    panel_stats = {
        "control": {
            "delta": float(paired_summary["D_ctl_mean_on_minus_off"]),
            "delta_label": "D_ctl",
            "t": _safe_float(paired_summary["control_state_t"]),
            "p": _safe_float(paired_summary["control_state_p_two_sided"]),
        },
        "selected": {
            "delta": float(paired_summary["D_sel_mean_on_minus_off"]),
            "delta_label": "D_sel",
            "t": _safe_float(paired_summary["selected_state_t"]),
            "p": _safe_float(paired_summary["selected_state_p_two_sided"]),
        },
    }
    value_arr = long_df["value"].to_numpy(dtype=np.float64)
    value_arr = value_arr[np.isfinite(value_arr)]
    y_limits: tuple[float, float] | None = None
    if value_arr.size > 0:
        y_min = float(value_arr.min())
        y_max = float(value_arr.max())
        pad = max(0.2, 0.06 * (y_max - y_min if y_max > y_min else 1.0))
        y_limits = (y_min - pad, y_max + pad)

    for ax, selection in zip(axes, ("control", "selected")):
        subset = long_df.loc[long_df["selection"] == selection].copy()
        for subject, subject_df in subset.groupby("subject", sort=True):
            subject_df = subject_df.set_index("state").reindex(state_order)
            ys = subject_df["value"].to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(ys)
            if not np.any(finite_mask):
                continue
            ax.plot(
                x_positions[finite_mask],
                ys[finite_mask],
                color="0.78",
                alpha=0.9,
                linewidth=1.0,
                zorder=1,
            )
            ax.scatter(
                x_positions[finite_mask],
                ys[finite_mask],
                s=24,
                facecolor="white",
                edgecolor=colors[selection],
                linewidth=0.9,
                alpha=0.95,
                zorder=2,
            )

        state_means = subset.groupby("state")["value"].mean().reindex(state_order)
        state_sems = subset.groupby("state")["value"].apply(_sem).reindex(state_order).fillna(0.0)
        ax.errorbar(
            x_positions,
            state_means.to_numpy(dtype=np.float64),
            yerr=state_sems.to_numpy(dtype=np.float64),
            color=colors[selection],
            linewidth=2.6,
            marker="o",
            markersize=6.5,
            markerfacecolor=colors[selection],
            markeredgecolor="white",
            markeredgewidth=0.9,
            capsize=3.5,
            elinewidth=1.5,
            zorder=4,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["OFF", "ON"])
        ax.set_title(selection.capitalize(), fontsize=12, pad=8)
        ax.set_xlabel("Medication state")
        ax.grid(axis="y", color="0.9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        stats_text = (
            f"{panel_stats[selection]['delta_label']} = "
            f"{_fmt_signed(panel_stats[selection]['delta'])}\n"
            f"paired t = {_fmt_float(panel_stats[selection]['t'])}, "
            f"p = {_fmt_float(panel_stats[selection]['p'])}"
        )
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.8", alpha=0.95),
        )

    axes[0].set_ylabel(metric_label)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    interaction_text = (
        f"Interaction: D_sel - D_ctl = "
        f"{_fmt_signed(paired_summary['interaction_mean_D_sel_minus_D_ctl'])}; "
        f"p = {_fmt_float(_safe_float(paired_summary['interaction_p_two_sided']))}"
    )
    fig.text(
        0.5,
        0.015,
        interaction_text,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="0.25",
    )
    fig.savefig(figure_path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def _plot_state_selection_comparison(
    session_df: pd.DataFrame,
    state: str,
    metrics: list[tuple[str, str]],
    figure_path: Path,
) -> list[dict[str, object]]:
    subset = session_df.loc[session_df["state"] == state].copy().sort_values("subject")
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).ravel()
    colors = {"control": "#4c84a6", "selected": "#e74c5b"}
    summary_rows: list[dict[str, object]] = []

    for ax, (metric_name, metric_label) in zip(axes_arr, metrics):
        control_col = f"control_{metric_name}"
        selected_col = f"selected_{metric_name}"
        metric_df = subset.loc[:, ["subject", control_col, selected_col]].dropna().copy()

        for row in metric_df.itertuples(index=False):
            ax.plot([0, 1], [getattr(row, control_col), getattr(row, selected_col)], color="#666666", alpha=0.32, linewidth=1.0)
            ax.scatter([0], [getattr(row, control_col)], color=colors["control"], alpha=0.65, s=22, zorder=3)
            ax.scatter([1], [getattr(row, selected_col)], color=colors["selected"], alpha=0.65, s=22, zorder=3)

        control_mean = float(metric_df[control_col].mean())
        selected_mean = float(metric_df[selected_col].mean())
        delta = float(selected_mean - control_mean)
        t_result = stats.ttest_rel(metric_df[selected_col], metric_df[control_col], nan_policy="omit")
        p_two_sided = _safe_float(t_result.pvalue)

        ax.plot([0, 1], [control_mean, selected_mean], color="#111111", linewidth=2.2, marker="o", markersize=5, zorder=4)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Control", "Selected"])
        ax.set_title(metric_label)
        ax.set_xlabel("Voxel set")
        ax.set_ylabel("Metric value")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.04,
            0.96,
            (
                f"Control mean = {control_mean:.3f}\n"
                f"Selected mean = {selected_mean:.3f}\n"
                f"Selected - Control = {_fmt_signed(delta)}\n"
                f"p = {p_two_sided:.3f}" if p_two_sided is not None else
                f"Control mean = {control_mean:.3f}\n"
                f"Selected mean = {selected_mean:.3f}\n"
                f"Selected - Control = {_fmt_signed(delta)}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
        )

        summary_rows.append(
            {
                "state": state,
                "metric": metric_name,
                "metric_label": metric_label,
                "n_subject_sessions": int(metric_df.shape[0]),
                "control_mean": control_mean,
                "selected_mean": selected_mean,
                "selected_minus_control": delta,
                "paired_t": _safe_float(t_result.statistic),
                "paired_p_two_sided": p_two_sided,
            }
        )

    fig.suptitle(f"{state.upper()} sessions: Control vs Selected")
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary_rows


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    brain_flat, anat_shape = ph._load_brain_flat_indices(args.anat_path)
    selected_flat = ph._load_selected_flat_indices(args.selected_indices_path, anat_shape)
    selected_flat = selected_flat[np.isin(selected_flat, brain_flat)]
    if selected_flat.size == 0:
        raise ValueError("No selected voxels remain inside the anatomical non-zero mask.")

    motor_flat, motor_region_names, motor_region_counts, motor_patterns = ph._load_motor_flat_indices(
        motor_mask_path=args.motor_mask_path,
        anat_path=args.anat_path,
        label_patterns=ph._split_csv_patterns(args.motor_label_patterns),
        atlas_cache_dir=args.motor_atlas_cache_dir,
    )
    nonselected_flat = np.setdiff1d(motor_flat, selected_flat, assume_unique=False)
    if nonselected_flat.size == 0:
        raise ValueError("No control voxels remain after removing the selected voxel set from the motor pool.")

    target_flat = np.concatenate([selected_flat, nonselected_flat]).astype(np.int64, copy=False)
    n_selected = int(selected_flat.size)
    pre_normalize_each_file = True

    manifest_rows = ph._load_manifest_rows(args.manifest_path)
    session_groups = ph._group_manifest_rows_by_subject_session(manifest_rows)

    subject_session_rows: list[dict[str, object]] = []
    for session_label, session_rows in session_groups:
        session_num = int(session_rows[0].ses)
        state = "off" if session_num == 1 else "on"
        subject = str(session_rows[0].sub_tag)
        print(f"Processing {session_label} ({state}) with {len(session_rows)} runs...", flush=True)

        sub_norm_mean, sub_norm_std = ph._compute_per_voxel_mean_std(
            target_flat=target_flat,
            manifest_rows=session_rows,
            row_chunk_size=int(args.row_chunk_size),
            pre_normalize_each_file=pre_normalize_each_file,
        )
        sub_metric_values, _pair_counts, _variance_values, _trial_counts = ph._accumulate_consecutive_diff_and_variance_metrics(
            target_flat=target_flat,
            manifest_rows=session_rows,
            row_chunk_size=int(args.row_chunk_size),
            per_run_normalization=False,
            pre_normalize_each_file=pre_normalize_each_file,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            sub_cv_values = np.where(
                np.abs(sub_norm_mean) > 1e-8,
                sub_norm_std / np.abs(sub_norm_mean),
                np.nan,
            )
            sub_norm_diff_values = np.where(
                np.abs(sub_norm_mean) > 1e-8,
                sub_metric_values / np.abs(sub_norm_mean),
                np.nan,
            )

        subject_session_rows.append(
            {
                "subject": subject,
                "session": session_num,
                "state": state,
                "n_runs": int(len(session_rows)),
                "selected_cv_mean": float(np.nanmean(sub_cv_values[:n_selected])),
                "control_cv_mean": float(np.nanmean(sub_cv_values[n_selected:])),
                "selected_norm_diff_mean": float(np.nanmean(sub_norm_diff_values[:n_selected])),
                "control_norm_diff_mean": float(np.nanmean(sub_norm_diff_values[n_selected:])),
            }
        )

    session_df = pd.DataFrame(subject_session_rows).sort_values(["subject", "session"]).reset_index(drop=True)
    session_csv_path = args.output_dir / "subject_session_state_selection_means.csv"
    session_df.to_csv(session_csv_path, index=False)

    metric_specs = [
        ("cv_mean", "Coefficient of Variation (std/|mean|)"),
        ("norm_diff_mean", "Normalized |Delta| (consecutive diff / |mean|)"),
    ]
    summary_rows: list[dict[str, object]] = []
    mixedlm_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    state_selection_rows: list[dict[str, object]] = []

    for state in ("off", "on"):
        figure_path = args.output_dir / f"{state}_state_control_vs_selected.png"
        state_selection_rows.extend(
            _plot_state_selection_comparison(
                session_df=session_df,
                state=state,
                metrics=metric_specs,
                figure_path=figure_path,
            )
        )

    for metric_name, metric_label in metric_specs:
        long_df = pd.concat(
            [
                session_df.loc[:, ["subject", "session", "state", "n_runs", f"selected_{metric_name}"]]
                .rename(columns={f"selected_{metric_name}": "value"})
                .assign(selection="selected"),
                session_df.loc[:, ["subject", "session", "state", "n_runs", f"control_{metric_name}"]]
                .rename(columns={f"control_{metric_name}": "value"})
                .assign(selection="control"),
            ],
            ignore_index=True,
        )

        long_df = long_df.loc[np.isfinite(long_df["value"].to_numpy(dtype=np.float64))].copy()

        mixedlm_summary = _fit_mixedlm(long_df)
        mixedlm_summary.update({"metric": metric_name, "metric_label": metric_label})
        mixedlm_rows.append(mixedlm_summary)

        wide_df = session_df.pivot(index="subject", columns="state")
        wide_df.columns = [f"{col}_{state}" for col, state in wide_df.columns]
        required_cols = [
            f"selected_{metric_name}_off",
            f"selected_{metric_name}_on",
            f"control_{metric_name}_off",
            f"control_{metric_name}_on",
        ]
        paired_df = wide_df.dropna(subset=required_cols).reset_index()

        paired_summary = _paired_state_summary(paired_df, metric_name=metric_name)
        paired_summary.update({"metric": metric_name, "metric_label": metric_label})
        paired_rows.append(paired_summary)

        plot_path = args.output_dir / f"{metric_name}_subject_state_lines.png"
        _plot_subject_means(
            long_df=long_df,
            metric_label=metric_label,
            figure_path=plot_path,
            paired_summary=paired_summary,
        )

        for state in ("off", "on"):
            subset = session_df.loc[:, ["subject", "session", "state", f"selected_{metric_name}", f"control_{metric_name}"]]
            state_subset = subset.loc[subset["state"] == state]
            summary_rows.append(
                {
                    "metric": metric_name,
                    "metric_label": metric_label,
                    "state": state,
                    "n_sessions": int(state_subset.shape[0]),
                    "selected_mean": float(state_subset[f"selected_{metric_name}"].mean()),
                    "control_mean": float(state_subset[f"control_{metric_name}"].mean()),
                    "selected_median": float(state_subset[f"selected_{metric_name}"].median()),
                    "control_median": float(state_subset[f"control_{metric_name}"].median()),
                }
            )

    state_summary_df = pd.DataFrame(summary_rows)
    mixedlm_df = pd.DataFrame(mixedlm_rows)
    paired_df = pd.DataFrame(paired_rows)
    state_selection_df = pd.DataFrame(state_selection_rows)

    state_summary_csv = args.output_dir / "state_selection_summary.csv"
    mixedlm_csv = args.output_dir / "mixedlm_interaction_summary.csv"
    paired_csv = args.output_dir / "paired_delta_summary.csv"
    state_selection_csv = args.output_dir / "within_state_control_vs_selected_summary.csv"
    state_summary_df.to_csv(state_summary_csv, index=False)
    mixedlm_df.to_csv(mixedlm_csv, index=False)
    paired_df.to_csv(paired_csv, index=False)
    state_selection_df.to_csv(state_selection_csv, index=False)

    summary_json = {
        "selected_indices_path": str(args.selected_indices_path),
        "anat_path": str(args.anat_path),
        "manifest_path": str(args.manifest_path),
        "motor_mask_source": str(args.motor_mask_path) if args.motor_mask_path is not None else "harvard_oxford_auto",
        "motor_label_patterns": list(ph._split_csv_patterns(args.motor_label_patterns)),
        "motor_region_names": list(motor_region_names),
        "motor_region_counts": [int(v) for v in motor_region_counts],
        "motor_pool_size": int(motor_flat.size),
        "selected_count_total": int(selected_flat.size),
        "control_count_total": int(nonselected_flat.size),
        "n_subject_sessions": int(session_df.shape[0]),
        "n_unique_subjects": int(session_df["subject"].nunique()),
        "subjects_with_both_states": int(
            session_df.groupby("subject")["state"].nunique().eq(2).sum()
        ),
        "state_summary_csv": str(state_summary_csv),
        "mixedlm_summary_csv": str(mixedlm_csv),
        "paired_summary_csv": str(paired_csv),
        "within_state_selection_summary_csv": str(state_selection_csv),
        "subject_session_means_csv": str(session_csv_path),
        "off_state_control_vs_selected_png": str(args.output_dir / "off_state_control_vs_selected.png"),
        "on_state_control_vs_selected_png": str(args.output_dir / "on_state_control_vs_selected.png"),
        "metrics": {
            row["metric"]: {
                "metric_label": row["metric_label"],
                "mixedlm": mixedlm_df.loc[mixedlm_df["metric"] == row["metric"]].iloc[0].drop(labels=["metric", "metric_label"]).to_dict(),
                "paired_delta": paired_df.loc[paired_df["metric"] == row["metric"]].iloc[0].drop(labels=["metric", "metric_label"]).to_dict(),
                "state_summary": state_summary_df.loc[state_summary_df["metric"] == row["metric"]].drop(columns=["metric", "metric_label"]).to_dict(orient="records"),
            }
            for row in paired_rows
        },
    }

    summary_json_path = args.output_dir / "state_selection_stability_summary.json"
    summary_json_path.write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print(f"Saved subject-session means: {session_csv_path}", flush=True)
    print(f"Saved state summary: {state_summary_csv}", flush=True)
    print(f"Saved mixed-model summary: {mixedlm_csv}", flush=True)
    print(f"Saved paired-delta summary: {paired_csv}", flush=True)
    print(f"Saved within-state selected-vs-control summary: {state_selection_csv}", flush=True)
    print(f"Saved summary JSON: {summary_json_path}", flush=True)


if __name__ == "__main__":
    main()
