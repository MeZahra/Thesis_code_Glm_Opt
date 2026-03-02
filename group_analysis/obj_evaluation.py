#!/usr/bin/env python3
"""Evaluate obj_param ablation runs from SLURM logs with a composite generalization metric.

Metric (per fold f and model m):
    S = w_test_corr * norm(|corr_test|)
      + w_corr_stability * (1 - norm(| |corr_train| - |corr_test| |))
      + w_loss_stability * (1 - norm(|loss_test - loss_train| / (|loss_train| + eps)))
      + w_complexity * complexity_bonus

where complexity_bonus is based on the number of non-zero penalty coefficients in the model.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.stats import mannwhitneyu
except Exception:
    mannwhitneyu = None


COMBO_RE = re.compile(
    r"^Fold\s+(?P<fold>\d+): optimization with task=(?P<task>[-+0-9.eE]+), "
    r"bold=(?P<bold>[-+0-9.eE]+), beta=(?P<beta>[-+0-9.eE]+), "
    r"smooth=(?P<smooth>[-+0-9.eE]+), gamma=(?P<gamma>[-+0-9.eE]+)"
)
TRAIN_LOSS_RE = re.compile(r"Total loss \(train objective\):\s*(?P<val>[-+0-9.eE]+)")
TEST_LOSS_RE = re.compile(r"Total loss \(test objective\):\s*(?P<val>[-+0-9.eE]+)")
TRAIN_CORR_RE = re.compile(r"Train metrics -> corr:\s*(?P<val>[-+0-9.eE]+)")
TEST_CORR_RE = re.compile(r"Test metrics\s+-> corr:\s*(?P<val>[-+0-9.eE]+)")


@dataclass(frozen=True)
class ComboKey:
    task: float
    bold: float
    beta: float
    smooth: float
    gamma: float

    def label(self) -> str:
        return (
            f"task={self.task:g}, bold={self.bold:g}, "
            f"beta={self.beta:g}, smooth={self.smooth:g}, gamma={self.gamma:g}"
        )


@dataclass
class FoldRecord:
    log_file: str
    fold: int
    combo: ComboKey
    train_loss: float
    test_loss: float
    train_corr: float
    test_corr: float


def _parse_log(log_path: Path) -> List[FoldRecord]:
    records: List[FoldRecord] = []
    current_combo: Optional[Tuple[int, ComboKey]] = None
    current_values: Dict[str, float] = {}

    def flush_if_complete() -> None:
        nonlocal current_combo, current_values
        if current_combo is None:
            return
        needed = ("train_loss", "test_loss", "train_corr", "test_corr")
        if not all(k in current_values for k in needed):
            return
        fold_id, combo = current_combo
        records.append(
            FoldRecord(
                log_file=str(log_path),
                fold=fold_id,
                combo=combo,
                train_loss=float(current_values["train_loss"]),
                test_loss=float(current_values["test_loss"]),
                train_corr=float(current_values["train_corr"]),
                test_corr=float(current_values["test_corr"]),
            )
        )
        current_combo = None
        current_values = {}

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            combo_match = COMBO_RE.match(line)
            if combo_match:
                flush_if_complete()
                combo = ComboKey(
                    task=float(combo_match.group("task")),
                    bold=float(combo_match.group("bold")),
                    beta=float(combo_match.group("beta")),
                    smooth=float(combo_match.group("smooth")),
                    gamma=float(combo_match.group("gamma")),
                )
                current_combo = (int(combo_match.group("fold")), combo)
                current_values = {}
                continue

            if current_combo is None:
                continue

            train_loss_match = TRAIN_LOSS_RE.search(line)
            if train_loss_match:
                current_values["train_loss"] = float(train_loss_match.group("val"))

            test_loss_match = TEST_LOSS_RE.search(line)
            if test_loss_match:
                current_values["test_loss"] = float(test_loss_match.group("val"))

            train_corr_match = TRAIN_CORR_RE.search(line)
            if train_corr_match:
                current_values["train_corr"] = float(train_corr_match.group("val"))

            test_corr_match = TEST_CORR_RE.search(line)
            if test_corr_match:
                current_values["test_corr"] = float(test_corr_match.group("val").rstrip(","))

            flush_if_complete()

    return records


def _deduplicate_records(records: Sequence[FoldRecord]) -> List[FoldRecord]:
    unique: Dict[Tuple, FoldRecord] = {}
    for rec in records:
        key = (
            Path(rec.log_file).name,
            rec.fold,
            rec.combo.task,
            rec.combo.bold,
            rec.combo.beta,
            rec.combo.smooth,
            rec.combo.gamma,
            round(rec.train_loss, 6),
            round(rec.test_loss, 6),
            round(rec.train_corr, 6),
            round(rec.test_corr, 6),
        )
        if key not in unique:
            unique[key] = rec
    return list(unique.values())


def _robust_minmax(values: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    finite = np.isfinite(values)
    scaled = np.zeros(values.shape, dtype=np.float64)
    if not np.any(finite):
        return scaled

    lo = float(np.nanpercentile(values[finite], low_pct))
    hi = float(np.nanpercentile(values[finite], high_pct))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-9

    scaled[finite] = (values[finite] - lo) / (hi - lo)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    return scaled


def _compute_dataframe(
    records: Sequence[FoldRecord],
    low_pct: float,
    high_pct: float,
    w_test_corr: float,
    w_corr_stability: float,
    w_loss_stability: float,
    w_complexity: float,
    eps: float,
) -> pd.DataFrame:
    rows = []

    active_counts = [
        int((rec.combo.task > 0) + (rec.combo.bold > 0) + (rec.combo.beta > 0) + (rec.combo.smooth > 0))
        for rec in records
    ]
    k_min = int(min(active_counts)) if active_counts else 1
    k_max = int(max(active_counts)) if active_counts else 1
    k_denom = float(max(1, k_max - k_min))

    for rec, k in zip(records, active_counts):
        abs_train_corr = abs(rec.train_corr)
        abs_test_corr = abs(rec.test_corr)
        corr_gap = abs(abs_train_corr - abs_test_corr)
        loss_gap = abs(rec.test_loss - rec.train_loss)
        loss_gap_rel = loss_gap / (abs(rec.train_loss) + eps)
        complexity_bonus = (k - k_min) / k_denom
        log_name = Path(rec.log_file).name
        combo_label = rec.combo.label()
        model_id = f"{log_name} | {combo_label}"
        rows.append(
            {
                "log_file": rec.log_file,
                "log_name": log_name,
                "fold": rec.fold,
                "combo_label": combo_label,
                "model_id": model_id,
                "task": rec.combo.task,
                "bold": rec.combo.bold,
                "beta": rec.combo.beta,
                "smooth": rec.combo.smooth,
                "gamma": rec.combo.gamma,
                "active_param_count": k,
                "train_loss": rec.train_loss,
                "test_loss": rec.test_loss,
                "train_corr": rec.train_corr,
                "test_corr": rec.test_corr,
                "abs_train_corr": abs_train_corr,
                "abs_test_corr": abs_test_corr,
                "corr_gap": corr_gap,
                "loss_gap": loss_gap,
                "loss_gap_rel": loss_gap_rel,
                "complexity_bonus": complexity_bonus,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["norm_test_corr"] = _robust_minmax(df["abs_test_corr"].to_numpy(), low_pct, high_pct)
    df["norm_corr_gap"] = _robust_minmax(df["corr_gap"].to_numpy(), low_pct, high_pct)
    df["norm_loss_gap_rel"] = _robust_minmax(df["loss_gap_rel"].to_numpy(), low_pct, high_pct)

    df["score_component_test_corr"] = w_test_corr * df["norm_test_corr"]
    df["score_component_corr_stability"] = w_corr_stability * (1.0 - df["norm_corr_gap"])
    df["score_component_loss_stability"] = w_loss_stability * (1.0 - df["norm_loss_gap_rel"])
    df["score_component_complexity"] = w_complexity * df["complexity_bonus"]

    df["evaluation_score"] = (
        df["score_component_test_corr"]
        + df["score_component_corr_stability"]
        + df["score_component_loss_stability"]
        + df["score_component_complexity"]
    )

    return df


def _make_summary(df: pd.DataFrame, group_by: str = "log_combo") -> pd.DataFrame:
    if group_by == "log_combo":
        group_cols = ["model_id", "log_name", "combo_label", "task", "bold", "beta", "smooth", "gamma"]
    elif group_by == "combo":
        group_cols = ["combo_label", "task", "bold", "beta", "smooth", "gamma"]
    else:
        raise ValueError(f"Unsupported group_by option: {group_by}")

    grouped = (
        df.groupby(group_cols, as_index=False)
        .agg(
            folds=("evaluation_score", "size"),
            score_mean=("evaluation_score", "mean"),
            score_std=("evaluation_score", "std"),
            test_corr_mean=("abs_test_corr", "mean"),
            train_corr_mean=("abs_train_corr", "mean"),
            corr_gap_mean=("corr_gap", "mean"),
            loss_gap_rel_mean=("loss_gap_rel", "mean"),
            train_loss_mean=("train_loss", "mean"),
            test_loss_mean=("test_loss", "mean"),
            active_param_count=("active_param_count", "max"),
            comp_test_corr_mean=("score_component_test_corr", "mean"),
            comp_corr_stability_mean=("score_component_corr_stability", "mean"),
            comp_loss_stability_mean=("score_component_loss_stability", "mean"),
            comp_complexity_mean=("score_component_complexity", "mean"),
        )
    )
    grouped["rank"] = grouped["score_mean"].rank(ascending=False, method="dense").astype(int)
    grouped = grouped.sort_values(["rank", "score_mean"], ascending=[True, False]).reset_index(drop=True)
    return grouped


def _plot_results(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    output_png: Path,
    title: str,
    label_col: str = "model_id",
) -> pd.DataFrame:
    combo_order = summary[label_col].tolist()
    if combo_order:
        mean_scores = (
            df.groupby(label_col, as_index=True)["evaluation_score"]
            .mean()
            .sort_values(ascending=False)
        )
        first_label = combo_order[0]
        remaining_sorted = [lbl for lbl in mean_scores.index.tolist() if lbl != first_label]
        combo_order = [first_label] + remaining_sorted

    summary_ordered = summary.set_index(label_col).reindex(combo_order).reset_index()
    pos_map = {label: idx for idx, label in enumerate(combo_order)}
    param_keys = list(
        zip(
            summary_ordered["task"],
            summary_ordered["bold"],
            summary_ordered["beta"],
            summary_ordered["smooth"],
            summary_ordered["gamma"],
        )
    )
    totals_per_param: Dict[Tuple[float, float, float, float, float], int] = {}
    for key in param_keys:
        totals_per_param[key] = totals_per_param.get(key, 0) + 1
    seen_per_param: Dict[Tuple[float, float, float, float, float], int] = {}
    compact_label_map: Dict[str, str] = {}
    for _, row in summary_ordered.iterrows():
        key = (row["task"], row["bold"], row["beta"], row["smooth"], row["gamma"])
        seen_per_param[key] = seen_per_param.get(key, 0) + 1
        gamma_text = "" if np.isclose(float(row["gamma"]), 1.0) else f"\ng={row['gamma']:g}"
        base_label = (
            f"t={row['task']:g}\n"
            f"b={row['bold']:g}\n"
            f"be={row['beta']:g}\n"
            f"s={row['smooth']:g}"
            f"{gamma_text}"
        )
        if totals_per_param[key] > 1:
            base_label = f"{base_label}\nrun {seen_per_param[key]}"
        compact_label_map[row[label_col]] = base_label
    compact_labels = [compact_label_map[label] for label in combo_order]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left panel: fold-level score distribution
    ax = axes[0]
    grouped_vals = [df.loc[df[label_col] == label, "evaluation_score"].to_numpy() for label in combo_order]
    bplot = ax.boxplot(grouped_vals, positions=np.arange(len(combo_order)), patch_artist=True, widths=0.65)

    for i, box in enumerate(bplot["boxes"]):
        if i == 0:
            box.set(facecolor="#d62728", alpha=0.8)
        else:
            box.set(facecolor="#4c72b0", alpha=0.55)
    for median in bplot["medians"]:
        median.set(color="black", linewidth=1.6)

    stat_rows: List[Dict[str, object]] = []
    if len(grouped_vals) > 1:
        first_vals = grouped_vals[0]
        raw_p_values: List[float] = []
        comparisons: List[Tuple[int, np.ndarray]] = []
        for idx in range(1, len(grouped_vals)):
            other_vals = grouped_vals[idx]
            comparisons.append((idx, other_vals))
            if first_vals.size < 2 or other_vals.size < 2 or mannwhitneyu is None:
                raw_p_values.append(np.nan)
                continue
            result = mannwhitneyu(first_vals, other_vals, alternative="two-sided")
            raw_p_values.append(float(result.pvalue))

        m = max(1, len(comparisons))
        adjusted_p_values = [
            min(1.0, p * m) if np.isfinite(p) else np.nan
            for p in raw_p_values
        ]

        valid_max = [float(np.nanmax(vals)) for vals in grouped_vals if vals.size]
        y_base = max(valid_max) if valid_max else 0.0
        y_min = min(float(np.nanmin(vals)) for vals in grouped_vals if vals.size) if valid_max else 0.0
        y_span = max(1e-6, y_base - y_min)
        y_step = 0.06 * y_span
        top_for_ylim = y_base

        for (idx, other_vals), p_raw, p_adj in zip(comparisons, raw_p_values, adjusted_p_values):
            if np.isfinite(p_adj) and p_adj < 0.001:
                star = "***"
            elif np.isfinite(p_adj) and p_adj < 0.01:
                star = "**"
            elif np.isfinite(p_adj) and p_adj < 0.05:
                star = "*"
            else:
                star = ""

            local_max = float(np.nanmax(other_vals)) if other_vals.size else y_base
            star_y = max(local_max + y_step, y_base + 0.5 * y_step)
            top_for_ylim = max(top_for_ylim, star_y)
            if star:
                ax.text(
                    idx,
                    star_y,
                    star,
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="black",
                    fontweight="bold",
                )
            stat_rows.append(
                {
                    "first_model": combo_order[0],
                    "compared_model": combo_order[idx],
                    "test": "mannwhitneyu_two_sided_bonferroni",
                    "n_first": int(first_vals.size),
                    "n_compared": int(other_vals.size),
                    "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan,
                    "p_bonferroni": float(p_adj) if np.isfinite(p_adj) else np.nan,
                    "significant_0p05": bool(np.isfinite(p_adj) and p_adj < 0.05),
                    "star": star,
                }
            )

        if top_for_ylim > y_base:
            y_low, _ = ax.get_ylim()
            ax.set_ylim(y_low, top_for_ylim + 1.8 * y_step)

    ax.text(
        0.01,
        0.01,
        "*  p<0.05\n** p<0.01\n*** p<0.001",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="black", linewidth=0.8),
    )

    rng = np.random.default_rng(7)
    for label in combo_order:
        vals = df.loc[df[label_col] == label, "evaluation_score"].to_numpy()
        x_center = pos_map[label]
        jitter = rng.normal(0.0, 0.06, size=vals.size)
        ax.scatter(np.full(vals.size, x_center) + jitter, vals, s=16, c="black", alpha=0.35, linewidths=0)

    ax.set_xticks(np.arange(len(combo_order)))
    ax.set_xticklabels(compact_labels)
    ax.set_ylabel("Evaluation score")
    ax.set_title("Fold-level score distribution")
    ax.grid(alpha=0.25, axis="y")

    # Right panel: mean score components by model
    ax = axes[1]
    x = np.arange(len(combo_order))
    width = 0.2
    comp_names = [
        ("comp_test_corr_mean", "Test corr", "#1b9e77"),
        ("comp_corr_stability_mean", "Corr stability", "#d95f02"),
        ("comp_loss_stability_mean", "Loss stability", "#7570b3"),
        ("comp_complexity_mean", "Complexity bonus", "#e7298a"),
    ]

    for idx, (col, label, color) in enumerate(comp_names):
        ax.bar(
            x + (idx - 1.5) * width,
            summary_ordered[col].to_numpy(),
            width=width,
            label=label,
            color=color,
            alpha=0.9,
        )

    for row_idx, row in summary_ordered.iterrows():
        ax.text(
            row_idx,
            row["score_mean"] + 0.01,
            f"rank {int(row['rank'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(compact_labels)
    ax.set_ylabel("Mean weighted component")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(alpha=0.25, axis="y")

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.18, top=0.90, wspace=0.22)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    return pd.DataFrame(stat_rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate obj_param ablations from SLURM logs.")
    parser.add_argument(
        "--logs",
        nargs="+",
        default=["slurm-8078053.out", "slurm-8169162.out"],
        help="SLURM log paths.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/ablation",
        help="Output directory for CSV + figure.",
    )
    parser.add_argument("--low-pct", type=float, default=5.0, help="Low percentile for robust scaling.")
    parser.add_argument("--high-pct", type=float, default=95.0, help="High percentile for robust scaling.")

    parser.add_argument("--w-test-corr", type=float, default=0.40)
    parser.add_argument("--w-corr-stability", type=float, default=0.20)
    parser.add_argument("--w-loss-stability", type=float, default=0.30)
    parser.add_argument("--w-complexity", type=float, default=0.10)

    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--group-by",
        choices=("log_combo", "combo"),
        default="log_combo",
        help="How to define a model in summaries/plots.",
    )
    parser.add_argument(
        "--title",
        default="Model evaluation from train/test behavior and loss stability (all log models)",
        help="Figure title.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    weight_sum = args.w_test_corr + args.w_corr_stability + args.w_loss_stability + args.w_complexity
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("Score weights must have a positive finite sum.")

    log_paths = [Path(p) for p in args.logs]
    missing_logs = [p for p in log_paths if not p.exists()]
    if missing_logs:
        missing_text = ", ".join(str(p) for p in missing_logs)
        raise FileNotFoundError(f"Missing log file(s): {missing_text}")

    all_records: List[FoldRecord] = []
    for path in log_paths:
        parsed = _parse_log(path)
        print(f"Parsed {len(parsed)} fold records from {path}")
        all_records.extend(parsed)

    deduped = _deduplicate_records(all_records)
    print(f"Total parsed records: {len(all_records)} | after deduplication: {len(deduped)}")
    if not deduped:
        raise RuntimeError("No fold records were extracted from the provided logs.")

    df = _compute_dataframe(
        deduped,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        w_test_corr=args.w_test_corr,
        w_corr_stability=args.w_corr_stability,
        w_loss_stability=args.w_loss_stability,
        w_complexity=args.w_complexity,
        eps=args.eps,
    )
    summary = _make_summary(df, group_by=args.group_by)
    summary_combo = _make_summary(df, group_by="combo")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_csv = out_dir / "obj_evaluation_fold_metrics.csv"
    summary_csv = out_dir / "obj_evaluation_model_summary.csv"
    summary_combo_csv = out_dir / "obj_evaluation_model_summary_combined_by_combo.csv"
    fig_png = out_dir / "obj_evaluation_metric_comparison.png"

    df.sort_values(["log_name", "combo_label", "fold"]).to_csv(fold_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_combo.to_csv(summary_combo_csv, index=False)

    label_col = "model_id" if args.group_by == "log_combo" else "combo_label"
    stats_csv = out_dir / "obj_evaluation_first_vs_others_stats.csv"
    stats_df = _plot_results(df, summary, fig_png, args.title, label_col=label_col)
    stats_df.to_csv(stats_csv, index=False)

    print("\nTop models by score:")
    show_cols = [
        "rank",
        "model_id" if args.group_by == "log_combo" else "combo_label",
        "folds",
        "score_mean",
        "test_corr_mean",
        "corr_gap_mean",
        "loss_gap_rel_mean",
        "active_param_count",
    ]
    print(summary[show_cols].to_string(index=False))

    print("\nSaved files:")
    print(f"  fold-level metrics: {fold_csv}")
    print(f"  model summary:      {summary_csv}")
    print(f"  combo summary:      {summary_combo_csv}")
    print(f"  stat tests:         {stats_csv}")
    print(f"  figure:             {fig_png}")


if __name__ == "__main__":
    main()
