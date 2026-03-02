#!/usr/bin/env python3
"""Run-level connectivity metric search for Sham vs GVS and GVS subtype separation.

This script builds run-wise condition trial matrices from:
1) selected voxel beta matrix (voxels x trials),
2) run offsets in concat_manifest_group.tsv,
3) each run's trial_keep mask.

For each run and condition (sham, GVS1..GVS8), it computes connectivity in:
- voxel space (selected voxels as nodes),
- PC space (PCA scores as nodes).

Connectivity estimators include linear and non-linear methods:
- linear_pearson_corr
- linear_partial_corr
- nonlinear_spearman_corr
- nonlinear_distance_corr
- nonlinear_binned_nmi

Each method/space is evaluated with leave-one-run-out testing:
1) binary discrimination: sham vs all GVS
2) GVS subtype discrimination: GVS1..GVS8

The run is the independent test unit in all reported fold metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DEFAULT_BETA_PATH = "results/connectivity/selected_beta_trials.npy"
DEFAULT_MANIFEST_PATH = "results/connectivity/concat_manifest_group.tsv"
DEFAULT_OUTPUT_DIR = "results/connectivity/tmp2/run_level_connectivity_metric_search"

GVS_LABELS = ["sham", "GVS1", "GVS2", "GVS3", "GVS4", "GVS5", "GVS6", "GVS7", "GVS8"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute run-level connectivity features across Sham/GVS conditions and "
            "rank linear/non-linear connectivity estimators by discriminability."
        )
    )
    parser.add_argument("--beta-path", default=DEFAULT_BETA_PATH)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--voxel-selection",
        choices=["topvar", "random"],
        default="topvar",
        help="Voxel node selection strategy.",
    )
    parser.add_argument(
        "--voxel-nodes",
        type=int,
        default=24,
        help="Number of voxel nodes in voxel-space connectivity.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=12,
        help="Number of PCA nodes for PC-space connectivity.",
    )
    parser.add_argument(
        "--min-trials-per-condition",
        type=int,
        default=7,
        help="Minimum run-level trials required per condition.",
    )
    parser.add_argument(
        "--require-complete-runs",
        action="store_true",
        default=True,
        help=(
            "Keep only runs with at least --min-trials-per-condition for all "
            "nine conditions (default: enabled)."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-runs",
        action="store_true",
        help="Allow runs with missing/low-count conditions.",
    )
    parser.add_argument(
        "--nmi-bins",
        type=int,
        default=5,
        help="Quantile bins for binned NMI connectivity.",
    )
    parser.add_argument(
        "--methods",
        default="all",
        help=(
            "Comma-separated method IDs or 'all'. Available: "
            "linear_pearson_corr,linear_partial_corr,nonlinear_spearman_corr,"
            "nonlinear_distance_corr,nonlinear_binned_nmi"
        ),
    )
    parser.add_argument(
        "--spaces",
        default="voxel,pc",
        help="Comma-separated spaces to evaluate: voxel,pc",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional debug cap on number of runs after filtering.",
    )
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--random-state", type=int, default=17)
    return parser.parse_args()


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx].astype(np.float64, copy=False)


def _standardize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n_cols = int(x.shape[1])
    col_median = np.zeros(n_cols, dtype=np.float64)
    for j in range(n_cols):
        col = x[:, j]
        finite = col[np.isfinite(col)]
        col_median[j] = float(np.median(finite)) if finite.size else 0.0
    bad = ~np.isfinite(x)
    if np.any(bad):
        x = x.copy()
        x[bad] = col_median[np.where(bad)[1]]
    mu = np.mean(x, axis=0, keepdims=True)
    sd = np.std(x, axis=0, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (x - mu) / sd


def _corr_from_features(x: np.ndarray) -> np.ndarray:
    x_std = _standardize_columns(x)
    n = int(x_std.shape[0])
    if n < 2:
        p = int(x_std.shape[1])
        out = np.zeros((p, p), dtype=np.float64)
        np.fill_diagonal(out, 1.0)
        return out
    cov = (x_std.T @ x_std) / float(max(n - 1, 1))
    d = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(d, d)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 1e-12)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _choose_voxels_topvar(beta_mm: np.ndarray, n_select: int) -> np.ndarray:
    n_vox = int(beta_mm.shape[0])
    n_select = min(max(1, int(n_select)), n_vox)
    var_all = np.empty(n_vox, dtype=np.float64)
    chunk = 2048
    for start in range(0, n_vox, chunk):
        end = min(start + chunk, n_vox)
        block = np.asarray(beta_mm[start:end, :], dtype=np.float32)
        v = np.nanvar(block, axis=1).astype(np.float64, copy=False)
        var_all[start:end] = np.nan_to_num(v, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    idx = np.argpartition(var_all, -n_select)[-n_select:]
    return np.sort(idx.astype(np.int64, copy=False))


def _choose_voxels_random(beta_mm: np.ndarray, n_select: int, rng: np.random.Generator) -> np.ndarray:
    n_vox = int(beta_mm.shape[0])
    n_select = min(max(1, int(n_select)), n_vox)
    idx = rng.choice(n_vox, size=n_select, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _build_run_entries(
    manifest_df: pd.DataFrame,
    total_trials: int,
    min_trials_per_condition: int,
    require_complete_runs: bool,
) -> Tuple[List[dict], pd.DataFrame]:
    runs: List[dict] = []
    count_rows: List[dict] = []

    for _, row in manifest_df.iterrows():
        sub_tag = str(row["sub_tag"])
        ses = int(row["ses"])
        run = int(row["run"])
        run_key = f"{sub_tag}_ses-{ses}_run-{run}"

        offset_start = int(row["offset_start"])
        offset_end = int(row["offset_end"])
        if offset_start < 0 or offset_end > total_trials or offset_end < offset_start:
            raise ValueError(
                f"Invalid offsets for {run_key}: start={offset_start}, end={offset_end}, total={total_trials}"
            )
        out_cols = np.arange(offset_start, offset_end, dtype=np.int64)

        trial_keep_path = str(row["trial_keep_path"])
        trial_keep = np.asarray(np.load(trial_keep_path), dtype=bool).ravel()
        kept_local = np.flatnonzero(trial_keep).astype(np.int64, copy=False)

        if out_cols.size == kept_local.size:
            source_local = kept_local
        elif out_cols.size == trial_keep.size:
            source_local = np.arange(trial_keep.size, dtype=np.int64)
        else:
            raise ValueError(
                f"Trial mismatch for {run_key}: output={out_cols.size}, kept={kept_local.size}, source={trial_keep.size}"
            )

        cond_ids = source_local % len(GVS_LABELS)
        cond_cols: Dict[str, np.ndarray] = {}
        cond_counts: Dict[str, int] = {}
        for cond_id, label in enumerate(GVS_LABELS):
            cols = out_cols[cond_ids == cond_id]
            cond_cols[label] = cols
            cond_counts[label] = int(cols.size)

        run_payload = {
            "run_key": run_key,
            "sub_tag": sub_tag,
            "ses": ses,
            "run": run,
            "condition_cols": cond_cols,
            "condition_counts": cond_counts,
            "n_trials_run": int(out_cols.size),
        }

        is_complete = all(v >= int(min_trials_per_condition) for v in cond_counts.values())
        run_payload["is_complete"] = bool(is_complete)
        count_rows.append(
            {
                "run_key": run_key,
                "sub_tag": sub_tag,
                "ses": ses,
                "run": run,
                "n_trials_run": int(out_cols.size),
                **cond_counts,
                "is_complete": bool(is_complete),
            }
        )

        if require_complete_runs and (not is_complete):
            continue
        runs.append(run_payload)

    if not runs:
        raise RuntimeError("No runs left after applying run/condition trial filters.")

    count_df = pd.DataFrame(count_rows).sort_values(["sub_tag", "ses", "run"]).reset_index(drop=True)
    return runs, count_df


def _fit_global_pca(
    samples: List[dict],
    n_components: int,
    random_state: int,
) -> Tuple[SimpleImputer, StandardScaler, PCA]:
    x_pool = np.vstack([s["x_raw"] for s in samples]).astype(np.float32, copy=False)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)
    x_imp = imputer.fit_transform(x_pool)
    x_std = scaler.fit_transform(x_imp)

    n_components = min(int(n_components), x_std.shape[0] - 1, x_std.shape[1])
    if n_components < 3:
        raise RuntimeError(f"Too few PCA components available: {n_components}")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(random_state))
    pca.fit(x_std)
    return imputer, scaler, pca


def _pearson_connectivity(x: np.ndarray) -> np.ndarray:
    return _corr_from_features(x)


def _partial_corr_connectivity(x: np.ndarray) -> np.ndarray:
    try:
        lw = LedoitWolf().fit(x)
        precision = lw.precision_.astype(np.float64, copy=False)
        d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
        p = -precision / np.outer(d, d)
        p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        p = np.clip(p, -1.0, 1.0)
        np.fill_diagonal(p, 1.0)
        return p
    except Exception:
        n = int(x.shape[1])
        out = np.zeros((n, n), dtype=np.float64)
        np.fill_diagonal(out, 1.0)
        return out


def _spearman_connectivity(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    ranks = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        ranks[:, j] = rankdata(x[:, j], method="average")
    return _corr_from_features(ranks)


def _distance_corr_connectivity(x: np.ndarray) -> np.ndarray:
    n, p = x.shape
    flat = np.empty((p, n * n), dtype=np.float64)

    for k in range(p):
        v = x[:, k].astype(np.float64, copy=False)
        d = np.abs(v[:, None] - v[None, :])
        row_mean = d.mean(axis=1, keepdims=True)
        col_mean = d.mean(axis=0, keepdims=True)
        grand = float(d.mean())
        centered = d - row_mean - col_mean + grand
        flat[k, :] = centered.reshape(-1)

    gram = (flat @ flat.T) / float(n * n)
    diag = np.clip(np.diag(gram), 1e-12, None)
    denom = np.sqrt(np.outer(diag, diag))
    dcor2 = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    dcor2 = np.clip(dcor2, 0.0, 1.0)
    dcor = np.sqrt(dcor2)
    np.fill_diagonal(dcor, 1.0)
    return dcor


def _binned_nmi_connectivity(x: np.ndarray, n_bins: int) -> np.ndarray:
    n, p = x.shape
    n_bins = max(2, int(n_bins))

    ranks = np.empty((n, p), dtype=np.float64)
    for j in range(p):
        ranks[:, j] = rankdata(x[:, j], method="average") / (n + 1.0)

    bins = np.floor(ranks * n_bins).astype(np.int32, copy=False)
    bins = np.clip(bins, 0, n_bins - 1)

    out = np.eye(p, dtype=np.float64)
    for i in range(p):
        xi = bins[:, i]
        uniq_i = int(np.unique(xi).size)
        for j in range(i + 1, p):
            xj = bins[:, j]
            uniq_j = int(np.unique(xj).size)
            if uniq_i <= 1 or uniq_j <= 1:
                val = 0.0
            else:
                val = float(normalized_mutual_info_score(xi, xj))
                if not np.isfinite(val):
                    val = 0.0
            out[i, j] = val
            out[j, i] = val
    np.fill_diagonal(out, 1.0)
    return out


def _wilcoxon_greater(scores: np.ndarray, chance: float) -> Tuple[float, float]:
    diff = np.asarray(scores, dtype=np.float64) - float(chance)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return np.nan, np.nan
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    try:
        stat, pval = stats.wilcoxon(diff, alternative="greater", zero_method="wilcox", mode="auto")
        return float(stat), float(pval)
    except Exception:
        return np.nan, np.nan


def _leave_one_run_out_eval(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    chance_level: float,
    model_factory: Callable[[], LinearSVC],
) -> Tuple[dict, pd.DataFrame, pd.DataFrame]:
    unique_runs = np.unique(groups)
    fold_rows = []
    y_true_all: List[str] = []
    y_pred_all: List[str] = []

    for run_key in unique_runs:
        test_mask = groups == run_key
        train_mask = ~test_mask
        if (not np.any(test_mask)) or (not np.any(train_mask)):
            continue
        y_train = y[train_mask]
        y_test = y[test_mask]
        if np.unique(y_train).size < 2:
            continue

        scaler = StandardScaler(with_mean=True, with_std=True)
        x_train = scaler.fit_transform(x[train_mask])
        x_test = scaler.transform(x[test_mask])

        model = model_factory()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        bal_acc = float(balanced_accuracy_score(y_test, y_pred))
        acc = float(accuracy_score(y_test, y_pred))
        fold_rows.append(
            {
                "run_key": str(run_key),
                "n_test": int(test_mask.sum()),
                "balanced_accuracy": bal_acc,
                "accuracy": acc,
            }
        )
        y_true_all.extend(y_test.astype(str).tolist())
        y_pred_all.extend(np.asarray(y_pred, dtype=str).tolist())

    if not fold_rows:
        empty = pd.DataFrame(columns=["run_key", "n_test", "balanced_accuracy", "accuracy"])
        return (
            {
                "n_runs_tested": 0,
                "balanced_accuracy_mean": np.nan,
                "balanced_accuracy_std": np.nan,
                "balanced_accuracy_overall": np.nan,
                "accuracy_overall": np.nan,
                "chance_level": float(chance_level),
                "wilcoxon_stat": np.nan,
                "wilcoxon_p_greater": np.nan,
                "mean_minus_chance": np.nan,
            },
            empty,
            pd.DataFrame(),
        )

    fold_df = pd.DataFrame(fold_rows)
    fold_scores = fold_df["balanced_accuracy"].to_numpy(dtype=np.float64)
    w_stat, w_p = _wilcoxon_greater(fold_scores, chance=float(chance_level))

    y_true_arr = np.asarray(y_true_all, dtype=str)
    y_pred_arr = np.asarray(y_pred_all, dtype=str)
    labels = sorted(np.unique(y_true_arr).tolist())
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels, normalize="true")
    cm_df = pd.DataFrame(cm, index=[f"true:{x}" for x in labels], columns=[f"pred:{x}" for x in labels])

    summary = {
        "n_runs_tested": int(fold_df.shape[0]),
        "balanced_accuracy_mean": float(np.mean(fold_scores)),
        "balanced_accuracy_std": float(np.std(fold_scores)),
        "balanced_accuracy_overall": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "accuracy_overall": float(accuracy_score(y_true_arr, y_pred_arr)),
        "chance_level": float(chance_level),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_p_greater": float(w_p) if np.isfinite(w_p) else np.nan,
        "mean_minus_chance": float(np.mean(fold_scores) - float(chance_level)),
    }
    return summary, fold_df, cm_df


def _add_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    out["binary_lift"] = out["binary_bal_acc_mean"] - 0.5
    out["gvs_lift"] = out["gvs_bal_acc_mean"] - (1.0 / 8.0)

    for col in ["binary_lift", "gvs_lift"]:
        vals = out[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        norm = np.full(vals.shape, np.nan, dtype=np.float64)
        if np.any(finite):
            vmin = float(np.min(vals[finite]))
            vmax = float(np.max(vals[finite]))
            if np.isclose(vmin, vmax):
                norm[finite] = 1.0
            else:
                norm[finite] = (vals[finite] - vmin) / (vmax - vmin)
        out[f"norm_{col}"] = norm

    out["combined_rank_score"] = 0.5 * out["norm_binary_lift"] + 0.5 * out["norm_gvs_lift"]
    out = out.sort_values(
        ["combined_rank_score", "binary_bal_acc_mean", "gvs_bal_acc_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return out


def _method_space_label(space: str, method_id: str) -> str:
    method_short = (
        str(method_id)
        .replace("linear_", "lin_")
        .replace("nonlinear_", "nlin_")
        .replace("_corr", "")
    )
    return f"{space}:{method_short}"


def _plot_ranked_balanced_accuracy(summary_ranked: pd.DataFrame, out_path: Path) -> None:
    if summary_ranked.empty:
        return

    plot_df = summary_ranked.copy()
    plot_df["label"] = [
        _method_space_label(s, m) for s, m in zip(plot_df["space"], plot_df["method_id"])
    ]
    x = np.arange(plot_df.shape[0], dtype=np.float64)
    colors = ["#4C78A8" if s == "pc" else "#F58518" for s in plot_df["space"]]

    fig, axes = plt.subplots(
        2, 1, figsize=(max(11.0, 0.85 * plot_df.shape[0]), 8.2), sharex=True
    )

    axes[0].bar(
        x,
        plot_df["binary_bal_acc_mean"].to_numpy(dtype=np.float64),
        yerr=plot_df["binary_bal_acc_std"].to_numpy(dtype=np.float64),
        color=colors,
        alpha=0.92,
        edgecolor="black",
        linewidth=0.4,
        capsize=2,
    )
    axes[0].axhline(0.5, color="black", linewidth=1.1, linestyle="--", label="Chance=0.5")
    axes[0].set_ylabel("Balanced Accuracy")
    axes[0].set_title("Run-Level Sham vs GVS Discrimination")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    gvs_chance = float(np.nanmedian(plot_df["gvs_chance_level"].to_numpy(dtype=np.float64)))
    if (not np.isfinite(gvs_chance)) or gvs_chance <= 0:
        gvs_chance = 1.0 / 8.0
    axes[1].bar(
        x,
        plot_df["gvs_bal_acc_mean"].to_numpy(dtype=np.float64),
        yerr=plot_df["gvs_bal_acc_std"].to_numpy(dtype=np.float64),
        color=colors,
        alpha=0.92,
        edgecolor="black",
        linewidth=0.4,
        capsize=2,
    )
    axes[1].axhline(gvs_chance, color="black", linewidth=1.1, linestyle="--", label=f"Chance={gvs_chance:.3f}")
    axes[1].set_ylabel("Balanced Accuracy")
    axes[1].set_title("Run-Level GVS Type Discrimination")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_df["label"].tolist(), rotation=55, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_lift_scatter(summary_ranked: pd.DataFrame, out_path: Path) -> None:
    if summary_ranked.empty:
        return

    plot_df = summary_ranked.copy()
    plot_df["label"] = [
        _method_space_label(s, m) for s, m in zip(plot_df["space"], plot_df["method_id"])
    ]

    color_map = {"pc": "#4C78A8", "voxel": "#F58518"}
    marker_map = {"linear": "o", "nonlinear": "s"}

    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    for _, row in plot_df.iterrows():
        x = float(row["binary_mean_minus_chance"])
        y = float(row["gvs_mean_minus_chance"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ax.scatter(
            x,
            y,
            s=70,
            c=color_map.get(str(row["space"]), "#888888"),
            marker=marker_map.get(str(row["family"]), "o"),
            edgecolor="black",
            linewidth=0.4,
            alpha=0.95,
        )
        ax.annotate(
            str(row["label"]),
            (x, y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Binary Lift (Sham vs GVS, BA - 0.5)")
    ax.set_ylabel("GVS-Type Lift (BA - chance)")
    ax.set_title("Method-Space Tradeoff: Binary vs GVS-Type Discrimination")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_pvalue_heatmaps(summary_ranked: pd.DataFrame, out_path: Path) -> None:
    if summary_ranked.empty:
        return

    plot_df = summary_ranked.copy()
    methods = sorted(plot_df["method_id"].astype(str).unique().tolist())
    spaces = sorted(plot_df["space"].astype(str).unique().tolist())

    def _build_matrix(col: str) -> np.ndarray:
        mat = np.full((len(methods), len(spaces)), np.nan, dtype=np.float64)
        for i, method in enumerate(methods):
            for j, space in enumerate(spaces):
                sel = plot_df[(plot_df["method_id"] == method) & (plot_df["space"] == space)]
                if sel.empty:
                    continue
                p = float(sel.iloc[0][col])
                if np.isfinite(p):
                    mat[i, j] = -np.log10(np.clip(p, 1e-12, 1.0))
        return mat

    mat_bin = _build_matrix("binary_p_wilcoxon_greater")
    mat_gvs = _build_matrix("gvs_p_wilcoxon_greater")

    vmax = np.nanmax(np.concatenate([mat_bin[np.isfinite(mat_bin)], mat_gvs[np.isfinite(mat_gvs)]]))
    if not np.isfinite(vmax):
        vmax = 1.0
    vmax = max(1.0, float(vmax))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.0, max(5.0, 0.5 * len(methods) + 2.0)),
        constrained_layout=True,
    )
    for ax, mat, title in [
        (axes[0], mat_bin, "Binary Task\n-log10(p)"),
        (axes[1], mat_gvs, "GVS-Type Task\n-log10(p)"),
    ]:
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_xticks(np.arange(len(spaces)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(spaces)
        ax.set_yticklabels(methods)
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    cbar.ax.set_ylabel("-log10(p)")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_fold_boxplots(
    fold_plot_df: pd.DataFrame,
    summary_ranked: pd.DataFrame,
    out_path: Path,
) -> None:
    if fold_plot_df.empty or summary_ranked.empty:
        return

    order_df = summary_ranked.copy()
    order_df["label"] = [
        _method_space_label(s, m) for s, m in zip(order_df["space"], order_df["method_id"])
    ]
    ordered_labels = order_df["label"].tolist()
    label_to_space = dict(zip(order_df["label"], order_df["space"]))
    color_map = {"pc": "#4C78A8", "voxel": "#F58518"}

    fig, axes = plt.subplots(2, 1, figsize=(max(11.0, 0.85 * len(ordered_labels)), 8.3), sharex=True)
    task_defs = [
        ("binary", 0.5, "Fold Balanced Accuracy (Sham vs GVS)"),
        ("gvs", 1.0 / 8.0, "Fold Balanced Accuracy (GVS Type)"),
    ]

    for ax, (task_name, chance, title) in zip(axes, task_defs):
        tdf = fold_plot_df[fold_plot_df["task"] == task_name].copy()
        data = []
        labels = []
        box_colors = []
        for label in ordered_labels:
            vals = tdf.loc[tdf["label"] == label, "balanced_accuracy"].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            data.append(vals)
            labels.append(label)
            box_colors.append(color_map.get(label_to_space.get(label, "pc"), "#888888"))
        if not data:
            ax.set_visible(False)
            continue
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            widths=0.6,
        )
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.4)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)
        ax.axhline(chance, color="black", linewidth=1.0, linestyle="--", label=f"Chance={chance:.3f}")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")

    axes[-1].tick_params(axis="x", labelrotation=55)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.random_state))

    if args.allow_incomplete_runs:
        require_complete_runs = False
    else:
        require_complete_runs = bool(args.require_complete_runs)

    beta_path = Path(args.beta_path).resolve()
    manifest_path = Path(args.manifest_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    beta_mm = np.load(beta_path, mmap_mode="r")
    if beta_mm.ndim != 2:
        raise ValueError(f"Expected beta matrix to be 2D, got {beta_mm.shape}")
    n_voxels_total, n_trials_total = int(beta_mm.shape[0]), int(beta_mm.shape[1])

    manifest_df = pd.read_csv(manifest_path, sep="\t")
    required_cols = {
        "offset_start",
        "offset_end",
        "sub_tag",
        "ses",
        "run",
        "trial_keep_path",
    }
    missing = required_cols.difference(manifest_df.columns.tolist())
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    manifest_df = manifest_df.sort_values(["sub_tag", "ses", "run"]).reset_index(drop=True)

    runs, run_count_df = _build_run_entries(
        manifest_df=manifest_df,
        total_trials=n_trials_total,
        min_trials_per_condition=int(args.min_trials_per_condition),
        require_complete_runs=require_complete_runs,
    )
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    if not runs:
        raise RuntimeError("No runs available after filtering.")

    selected_run_keys = {r["run_key"] for r in runs}
    run_count_used_df = run_count_df[run_count_df["run_key"].isin(selected_run_keys)].copy()
    run_count_used_df = run_count_used_df.sort_values(["sub_tag", "ses", "run"]).reset_index(drop=True)
    run_count_df.to_csv(output_dir / "run_condition_trial_counts_all.csv", index=False)
    run_count_used_df.to_csv(output_dir / "run_condition_trial_counts_used.csv", index=False)

    if args.voxel_selection == "topvar":
        voxel_idx = _choose_voxels_topvar(beta_mm=beta_mm, n_select=int(args.voxel_nodes))
    else:
        voxel_idx = _choose_voxels_random(beta_mm=beta_mm, n_select=int(args.voxel_nodes), rng=rng)

    samples: List[dict] = []
    sample_rows: List[dict] = []
    for run_entry in runs:
        run_key = run_entry["run_key"]
        for condition in GVS_LABELS:
            cols = run_entry["condition_cols"][condition]
            n_trials_cond = int(cols.size)
            if n_trials_cond < int(args.min_trials_per_condition):
                continue
            x_raw = np.asarray(beta_mm[np.ix_(voxel_idx, cols)], dtype=np.float32).T
            payload = {
                "run_key": run_key,
                "sub_tag": run_entry["sub_tag"],
                "ses": int(run_entry["ses"]),
                "run": int(run_entry["run"]),
                "condition": condition,
                "is_gvs": bool(condition.lower().startswith("gvs")),
                "x_raw": x_raw,
                "n_trials": n_trials_cond,
            }
            samples.append(payload)
            sample_rows.append(
                {
                    "run_key": run_key,
                    "sub_tag": run_entry["sub_tag"],
                    "ses": int(run_entry["ses"]),
                    "run": int(run_entry["run"]),
                    "condition": condition,
                    "n_trials": n_trials_cond,
                }
            )

    if not samples:
        raise RuntimeError("No run-condition samples available.")

    sample_df = pd.DataFrame(sample_rows).sort_values(["sub_tag", "ses", "run", "condition"]).reset_index(drop=True)
    sample_df.to_csv(output_dir / "sample_manifest.csv", index=False)

    imputer, scaler, pca = _fit_global_pca(
        samples=samples,
        n_components=int(args.pca_components),
        random_state=int(args.random_state),
    )

    for sample in samples:
        x_raw = sample["x_raw"]
        sample["x_voxel"] = _standardize_columns(x_raw)
        x_pc = pca.transform(scaler.transform(imputer.transform(x_raw))).astype(np.float64, copy=False)
        sample["x_pc"] = _standardize_columns(x_pc)

    method_defs = [
        {"method_id": "linear_pearson_corr", "family": "linear", "func": _pearson_connectivity},
        {"method_id": "linear_partial_corr", "family": "linear", "func": _partial_corr_connectivity},
        {"method_id": "nonlinear_spearman_corr", "family": "nonlinear", "func": _spearman_connectivity},
        {"method_id": "nonlinear_distance_corr", "family": "nonlinear", "func": _distance_corr_connectivity},
        {
            "method_id": "nonlinear_binned_nmi",
            "family": "nonlinear",
            "func": lambda x: _binned_nmi_connectivity(x, int(args.nmi_bins)),
        },
    ]

    if str(args.methods).strip().lower() != "all":
        wanted = {m.strip() for m in str(args.methods).split(",") if m.strip()}
        method_defs = [m for m in method_defs if m["method_id"] in wanted]
        if not method_defs:
            raise ValueError(f"No valid methods selected from --methods={args.methods}")

    spaces = [s.strip().lower() for s in str(args.spaces).split(",") if s.strip()]
    valid_spaces = {"voxel", "pc"}
    bad_spaces = sorted(set(spaces).difference(valid_spaces))
    if bad_spaces:
        raise ValueError(f"Invalid spaces: {bad_spaces}. Valid: {sorted(valid_spaces)}")
    if not spaces:
        raise ValueError("No valid spaces selected.")

    summary_rows: List[dict] = []
    scalar_rows: List[dict] = []
    fold_plot_rows: List[dict] = []

    for space in spaces:
        x_key = f"x_{space}"
        for method in method_defs:
            method_id = method["method_id"]
            conn_func = method["func"]

            edge_list = []
            labels_condition = []
            groups = []

            for sample in samples:
                x_data = sample[x_key]
                conn = conn_func(x_data).astype(np.float64, copy=False)
                edge_vec = _upper_triangle(conn)
                edge_vec = np.nan_to_num(edge_vec, nan=0.0, posinf=0.0, neginf=0.0)
                edge_list.append(edge_vec.astype(np.float32, copy=False))
                labels_condition.append(str(sample["condition"]))
                groups.append(str(sample["run_key"]))

                scalar_rows.append(
                    {
                        "space": space,
                        "method_id": method_id,
                        "family": method["family"],
                        "run_key": sample["run_key"],
                        "sub_tag": sample["sub_tag"],
                        "ses": sample["ses"],
                        "run": sample["run"],
                        "condition": sample["condition"],
                        "n_trials": sample["n_trials"],
                        "mean_abs_edge": float(np.mean(np.abs(edge_vec))),
                        "rms_edge": float(np.sqrt(np.mean(np.square(edge_vec)))),
                    }
                )

            x_edges = np.vstack(edge_list).astype(np.float64, copy=False)
            y_cond = np.asarray(labels_condition, dtype=object)
            group_arr = np.asarray(groups, dtype=object)

            y_bin = np.where(np.char.lower(y_cond.astype(str)) == "sham", "sham", "gvs").astype(object)
            binary_summary, binary_fold_df, binary_cm_df = _leave_one_run_out_eval(
                x=x_edges,
                y=y_bin,
                groups=group_arr,
                chance_level=0.5,
                model_factory=lambda: LinearSVC(
                    C=float(args.svm_c),
                    class_weight="balanced",
                    max_iter=20000,
                    random_state=int(args.random_state),
                ),
            )

            gvs_mask = np.array([str(lbl).lower().startswith("gvs") for lbl in y_cond], dtype=bool)
            x_gvs = x_edges[gvs_mask]
            y_gvs = y_cond[gvs_mask].astype(object)
            group_gvs = group_arr[gvs_mask]
            unique_gvs = sorted(np.unique(y_gvs).tolist())
            gvs_chance = 1.0 / float(len(unique_gvs))

            gvs_summary, gvs_fold_df, gvs_cm_df = _leave_one_run_out_eval(
                x=x_gvs,
                y=y_gvs,
                groups=group_gvs,
                chance_level=gvs_chance,
                model_factory=lambda: LinearSVC(
                    C=float(args.svm_c),
                    class_weight="balanced",
                    max_iter=30000,
                    random_state=int(args.random_state),
                ),
            )

            stem = f"{_safe_name(space)}__{_safe_name(method_id)}"
            binary_fold_df.to_csv(output_dir / f"{stem}__binary_fold_scores.csv", index=False)
            gvs_fold_df.to_csv(output_dir / f"{stem}__gvs_fold_scores.csv", index=False)
            if not binary_fold_df.empty:
                tmp = binary_fold_df.copy()
                tmp["space"] = space
                tmp["method_id"] = method_id
                tmp["family"] = method["family"]
                tmp["task"] = "binary"
                tmp["label"] = _method_space_label(space, method_id)
                fold_plot_rows.extend(tmp.to_dict(orient="records"))
            if not gvs_fold_df.empty:
                tmp = gvs_fold_df.copy()
                tmp["space"] = space
                tmp["method_id"] = method_id
                tmp["family"] = method["family"]
                tmp["task"] = "gvs"
                tmp["label"] = _method_space_label(space, method_id)
                fold_plot_rows.extend(tmp.to_dict(orient="records"))
            if not binary_cm_df.empty:
                binary_cm_df.to_csv(output_dir / f"{stem}__binary_confusion.csv")
            if not gvs_cm_df.empty:
                gvs_cm_df.to_csv(output_dir / f"{stem}__gvs_confusion.csv")

            summary_rows.append(
                {
                    "space": space,
                    "method_id": method_id,
                    "family": method["family"],
                    "n_samples": int(x_edges.shape[0]),
                    "n_runs": int(np.unique(group_arr).size),
                    "n_edges": int(x_edges.shape[1]),
                    "binary_bal_acc_mean": binary_summary["balanced_accuracy_mean"],
                    "binary_bal_acc_std": binary_summary["balanced_accuracy_std"],
                    "binary_bal_acc_overall": binary_summary["balanced_accuracy_overall"],
                    "binary_acc_overall": binary_summary["accuracy_overall"],
                    "binary_p_wilcoxon_greater": binary_summary["wilcoxon_p_greater"],
                    "binary_mean_minus_chance": binary_summary["mean_minus_chance"],
                    "gvs_bal_acc_mean": gvs_summary["balanced_accuracy_mean"],
                    "gvs_bal_acc_std": gvs_summary["balanced_accuracy_std"],
                    "gvs_bal_acc_overall": gvs_summary["balanced_accuracy_overall"],
                    "gvs_acc_overall": gvs_summary["accuracy_overall"],
                    "gvs_chance_level": gvs_chance,
                    "gvs_p_wilcoxon_greater": gvs_summary["wilcoxon_p_greater"],
                    "gvs_mean_minus_chance": gvs_summary["mean_minus_chance"],
                }
            )

            print(
                (
                    f"[{space} | {method_id}] "
                    f"binary_bal_acc_mean={binary_summary['balanced_accuracy_mean']:.4f} "
                    f"(p={binary_summary['wilcoxon_p_greater']:.3g}), "
                    f"gvs_bal_acc_mean={gvs_summary['balanced_accuracy_mean']:.4f} "
                    f"(p={gvs_summary['wilcoxon_p_greater']:.3g})"
                ),
                flush=True,
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_ranked = _add_rankings(summary_df)
    summary_ranked["significant_binary"] = summary_ranked["binary_p_wilcoxon_greater"] < float(args.alpha)
    summary_ranked["significant_gvs"] = summary_ranked["gvs_p_wilcoxon_greater"] < float(args.alpha)
    summary_ranked["significant_both"] = summary_ranked["significant_binary"] & summary_ranked["significant_gvs"]
    summary_ranked.to_csv(output_dir / "method_space_ranking_summary.csv", index=False)

    scalar_df = pd.DataFrame(scalar_rows)
    scalar_df.to_csv(output_dir / "run_level_connectivity_scalar_metrics.csv", index=False)
    fold_plot_df = pd.DataFrame(fold_plot_rows)
    if not fold_plot_df.empty:
        fold_plot_df.to_csv(output_dir / "fold_scores_all_methods.csv", index=False)

    figure_paths = {
        "ranked_balanced_accuracy": output_dir / "figure_ranked_balanced_accuracy.png",
        "lift_scatter": output_dir / "figure_binary_vs_gvs_lift_scatter.png",
        "pvalue_heatmaps": output_dir / "figure_wilcoxon_pvalue_heatmaps.png",
        "fold_boxplots": output_dir / "figure_fold_balanced_accuracy_boxplots.png",
    }
    _plot_ranked_balanced_accuracy(summary_ranked=summary_ranked, out_path=figure_paths["ranked_balanced_accuracy"])
    _plot_lift_scatter(summary_ranked=summary_ranked, out_path=figure_paths["lift_scatter"])
    _plot_pvalue_heatmaps(summary_ranked=summary_ranked, out_path=figure_paths["pvalue_heatmaps"])
    _plot_fold_boxplots(fold_plot_df=fold_plot_df, summary_ranked=summary_ranked, out_path=figure_paths["fold_boxplots"])

    best = summary_ranked.iloc[0].to_dict()
    manifest = {
        "beta_path": str(beta_path),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
        "n_voxels_total": n_voxels_total,
        "n_trials_total": n_trials_total,
        "n_runs_total_manifest": int(manifest_df.shape[0]),
        "n_runs_used": int(len(runs)),
        "min_trials_per_condition": int(args.min_trials_per_condition),
        "require_complete_runs": bool(require_complete_runs),
        "voxel_selection": str(args.voxel_selection),
        "voxel_nodes": int(voxel_idx.size),
        "spaces": spaces,
        "methods": [m["method_id"] for m in method_defs],
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "nmi_bins": int(args.nmi_bins),
        "random_state": int(args.random_state),
        "alpha": float(args.alpha),
        "best_method_space": best,
        "figures": {k: str(v) for k, v in figure_paths.items()},
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nTop-ranked method-space entries:", flush=True)
    print(
        summary_ranked[
            [
                "space",
                "method_id",
                "binary_bal_acc_mean",
                "binary_p_wilcoxon_greater",
                "gvs_bal_acc_mean",
                "gvs_p_wilcoxon_greater",
                "combined_rank_score",
                "significant_both",
            ]
        ]
        .head(10)
        .to_string(index=False),
        flush=True,
    )
    print(f"\nSaved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
