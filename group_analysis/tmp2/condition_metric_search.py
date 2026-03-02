#!/usr/bin/env python3
"""Search for condition metrics that separate sham/GVS and GVS subtypes.

This script evaluates multiple trial-distribution metrics on condition-split beta
matrices (`selected_beta_trials_<condition>.npy`) and reports which metrics show:
1) strong sham-vs-GVS separation and
2) strong separation among different GVS levels.

Linear metrics:
  - centroid cosine distance (in PCA space)
  - centroid Mahalanobis distance (in PCA space)
  - pairwise linear SVM balanced accuracy

Non-linear metrics:
  - RBF-kernel MMD^2 (with permutation p-values)
  - pairwise RBF SVM balanced accuracy
  - pairwise kNN balanced accuracy
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise as sk_pairwise
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


DEFAULT_INPUT_DIR = "results/connectivity"
DEFAULT_OUTPUT_DIR = "results/connectivity/tmp2_metrics"
FILE_PATTERN = "selected_beta_trials_*.npy"
FILE_PREFIX = "selected_beta_trials_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate linear and non-linear condition-discriminability metrics "
            "from condition-split beta matrices."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--voxel-subsample",
        type=int,
        default=4000,
        help="Number of voxels used for feature extraction (default: 4000).",
    )
    parser.add_argument(
        "--voxel-selection",
        choices=["topvar", "random"],
        default="topvar",
        help="Voxel selection strategy before PCA.",
    )
    parser.add_argument(
        "--trials-per-condition",
        type=int,
        default=None,
        help="If unset, use min available trials across conditions.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=50,
        help="Number of PCA components for metric computation.",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Cross-validation splits for classifier metrics.",
    )
    parser.add_argument(
        "--mmd-permutations",
        type=int,
        default=150,
        help="Permutation count for MMD p-values (set 0 to skip p-values).",
    )
    parser.add_argument(
        "--mmd-max-trials",
        type=int,
        default=240,
        help="Max per-condition trials used for MMD to control runtime.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=17,
    )
    return parser.parse_args()


def _condition_sort_key(label: str) -> Tuple[int, int | str]:
    s = str(label)
    if s.lower() == "sham":
        return (0, 0)
    m = re.match(r"^gvs(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return (1, int(m.group(1)))
    return (2, s.lower())


def discover_condition_files(input_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(input_dir.glob(FILE_PATTERN)):
        name = path.name
        if not name.startswith(FILE_PREFIX):
            continue
        label = name[len(FILE_PREFIX) : -4]
        if not label:
            continue
        out[label] = path
    return out


def _safe_name(metric_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_name.strip())


def _save_heatmap(matrix: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.3, 6.2))
    finite = np.isfinite(matrix)
    if finite.any():
        vmin = float(np.nanpercentile(matrix[finite], 5.0))
        vmax = float(np.nanpercentile(matrix[finite], 95.0))
        if np.isclose(vmin, vmax):
            vmin = float(np.nanmin(matrix[finite]))
            vmax = float(np.nanmax(matrix[finite]))
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
    else:
        vmin, vmax = -1.0, 1.0
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Metric value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _choose_voxels_topvar(files: Dict[str, Path], n_select: int) -> np.ndarray:
    pooled_var = None
    for condition, path in sorted(files.items(), key=lambda kv: _condition_sort_key(kv[0])):
        arr = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{condition} array is not 2D: {arr.shape}")
        var = np.nanvar(arr, axis=1).astype(np.float64, copy=False)
        if pooled_var is None:
            pooled_var = np.zeros_like(var, dtype=np.float64)
        pooled_var += np.nan_to_num(var, nan=0.0)
    if pooled_var is None:
        raise RuntimeError("No condition arrays found for voxel selection.")
    n_vox = int(pooled_var.shape[0])
    n_select = min(int(n_select), n_vox)
    idx = np.argpartition(pooled_var, -n_select)[-n_select:]
    idx = np.sort(idx.astype(np.int64, copy=False))
    return idx


def _choose_voxels_random(files: Dict[str, Path], n_select: int, rng: np.random.Generator) -> np.ndarray:
    first = np.load(next(iter(files.values())), mmap_mode="r")
    if first.ndim != 2:
        raise ValueError(f"Condition array is not 2D: {first.shape}")
    n_vox = int(first.shape[0])
    n_select = min(int(n_select), n_vox)
    idx = rng.choice(n_vox, size=n_select, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _build_feature_matrix(
    files: Dict[str, Path],
    ordered_conditions: List[str],
    voxel_idx: np.ndarray,
    trials_per_condition: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for condition in ordered_conditions:
        arr = np.asarray(np.load(files[condition], mmap_mode="r"), dtype=np.float32)
        subset = np.asarray(arr[voxel_idx, :], dtype=np.float32)
        n_trials = int(subset.shape[1])
        if n_trials < trials_per_condition:
            raise ValueError(
                f"{condition} has only {n_trials} trials (< {trials_per_condition})."
            )
        trial_idx = rng.choice(n_trials, size=trials_per_condition, replace=False)
        x_cond = subset[:, trial_idx].T
        xs.append(x_cond)
        ys.append(np.full(trials_per_condition, condition, dtype=object))
    x = np.vstack(xs).astype(np.float32, copy=False)
    y = np.concatenate(ys)
    return x, y


def _build_indices_by_condition(y: np.ndarray, ordered_conditions: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ordered_conditions:
        out[c] = np.where(y == c)[0]
    return out


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return np.nan
    return float(1.0 - np.dot(a, b) / denom)


def _mahalanobis_distance(a: np.ndarray, b: np.ndarray, precision: np.ndarray) -> float:
    d = (a - b).astype(np.float64, copy=False)
    val = float(d.T @ precision @ d)
    if val < 0:
        val = 0.0
    return float(np.sqrt(val))


def _pairwise_centroid_metrics(
    x: np.ndarray,
    ordered_conditions: List[str],
    indices: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    centroids = {}
    for c in ordered_conditions:
        centroids[c] = np.mean(x[indices[c]], axis=0).astype(np.float64, copy=False)

    lw = LedoitWolf().fit(x)
    precision = lw.precision_.astype(np.float64, copy=False)

    n = len(ordered_conditions)
    cosine_mat = np.full((n, n), np.nan, dtype=np.float64)
    maha_mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, ci in enumerate(ordered_conditions):
        cosine_mat[i, i] = 0.0
        maha_mat[i, i] = 0.0
        for j in range(i + 1, n):
            cj = ordered_conditions[j]
            cdist = _cosine_distance(centroids[ci], centroids[cj])
            mdist = _mahalanobis_distance(centroids[ci], centroids[cj], precision)
            cosine_mat[i, j] = cdist
            cosine_mat[j, i] = cdist
            maha_mat[i, j] = mdist
            maha_mat[j, i] = mdist

    return {
        "linear_centroid_cosine_distance": cosine_mat,
        "linear_centroid_mahalanobis_distance": maha_mat,
    }


def _mmd2_unbiased_from_kernel(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray) -> float:
    nx = kxx.shape[0]
    ny = kyy.shape[0]
    if nx < 2 or ny < 2:
        return np.nan
    term_x = (np.sum(kxx) - np.trace(kxx)) / (nx * (nx - 1))
    term_y = (np.sum(kyy) - np.trace(kyy)) / (ny * (ny - 1))
    term_xy = np.sum(kxy) / (nx * ny)
    return float(term_x + term_y - 2.0 * term_xy)


def _pairwise_mmd_rbf(
    x: np.ndarray,
    ordered_conditions: List[str],
    indices: Dict[str, np.ndarray],
    n_permutations: int,
    mmd_max_trials: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(ordered_conditions)
    mmd_mat = np.full((n, n), np.nan, dtype=np.float64)
    pval_mat = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        mmd_mat[i, i] = 0.0
        pval_mat[i, i] = 0.0

    for i, ci in enumerate(ordered_conditions):
        idx_i_full = indices[ci]
        if idx_i_full.size > mmd_max_trials:
            idx_i = rng.choice(idx_i_full, size=mmd_max_trials, replace=False)
        else:
            idx_i = idx_i_full
        xi = x[idx_i]
        for j in range(i + 1, n):
            cj = ordered_conditions[j]
            idx_j_full = indices[cj]
            if idx_j_full.size > mmd_max_trials:
                idx_j = rng.choice(idx_j_full, size=mmd_max_trials, replace=False)
            else:
                idx_j = idx_j_full
            xj = x[idx_j]

            xcat = np.vstack([xi, xj]).astype(np.float64, copy=False)
            d2 = pairwise_distances(xcat, metric="sqeuclidean")
            upper = d2[np.triu_indices_from(d2, k=1)]
            upper = upper[np.isfinite(upper) & (upper > 0)]
            median_d2 = float(np.median(upper)) if upper.size else 1.0
            if median_d2 <= 0:
                median_d2 = 1.0
            gamma = 1.0 / (2.0 * median_d2)

            k = sk_pairwise.rbf_kernel(xcat, xcat, gamma=gamma).astype(np.float64, copy=False)
            ni = xi.shape[0]
            nj = xj.shape[0]
            kxx = k[:ni, :ni]
            kyy = k[ni:, ni:]
            kxy = k[:ni, ni:]
            mmd_val = _mmd2_unbiased_from_kernel(kxx, kyy, kxy)
            mmd_mat[i, j] = mmd_val
            mmd_mat[j, i] = mmd_val

            if n_permutations > 0:
                n_tot = ni + nj
                null_vals = np.empty(n_permutations, dtype=np.float64)
                base_idx = np.arange(n_tot, dtype=np.int64)
                for p in range(n_permutations):
                    perm = rng.permutation(base_idx)
                    ii = perm[:ni]
                    jj = perm[ni:]
                    kxx_p = k[np.ix_(ii, ii)]
                    kyy_p = k[np.ix_(jj, jj)]
                    kxy_p = k[np.ix_(ii, jj)]
                    null_vals[p] = _mmd2_unbiased_from_kernel(kxx_p, kyy_p, kxy_p)
                pval = (1.0 + float(np.sum(null_vals >= mmd_val))) / (1.0 + n_permutations)
                pval_mat[i, j] = pval
                pval_mat[j, i] = pval

    return mmd_mat, pval_mat


def _pairwise_classifier_matrix(
    x: np.ndarray,
    y: np.ndarray,
    ordered_conditions: List[str],
    model_factory,
    cv_splits: int,
    random_state: int,
) -> np.ndarray:
    n = len(ordered_conditions)
    out = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        out[i, i] = 1.0
    for i, ci in enumerate(ordered_conditions):
        for j in range(i + 1, n):
            cj = ordered_conditions[j]
            mask = (y == ci) | (y == cj)
            x_pair = x[mask]
            y_pair = np.where(y[mask] == ci, 0, 1).astype(np.int32)
            class_counts = np.bincount(y_pair)
            max_splits = int(np.min(class_counts))
            n_splits = max(2, min(cv_splits, max_splits))
            cv = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            scores = cross_val_score(
                model_factory(),
                x_pair,
                y_pair,
                scoring="balanced_accuracy",
                cv=cv,
                n_jobs=1,
            )
            val = float(np.mean(scores))
            # Class label polarity is arbitrary in pairwise runs; fold accuracy
            # around 0.5 so "separable but label-flipped" still scores high.
            val = 0.5 + abs(val - 0.5)
            out[i, j] = val
            out[j, i] = val
    return out


def _summarize_pairwise_matrix(
    metric_name: str,
    matrix: np.ndarray,
    ordered_conditions: List[str],
    higher_is_better: bool,
) -> dict:
    sham_candidates = [k for k, c in enumerate(ordered_conditions) if c.lower() == "sham"]
    if not sham_candidates:
        raise RuntimeError("No sham label found in conditions.")
    sham_idx = sham_candidates[0]

    gvs_indices = [i for i, c in enumerate(ordered_conditions) if c.lower().startswith("gvs")]

    sham_vs_gvs = []
    for gi in gvs_indices:
        val = matrix[sham_idx, gi]
        if np.isfinite(val):
            sham_vs_gvs.append(float(val))

    gvs_vs_gvs = []
    for a in range(len(gvs_indices)):
        for b in range(a + 1, len(gvs_indices)):
            val = matrix[gvs_indices[a], gvs_indices[b]]
            if np.isfinite(val):
                gvs_vs_gvs.append(float(val))

    sham_mean = float(np.mean(sham_vs_gvs)) if sham_vs_gvs else np.nan
    gvs_mean = float(np.mean(gvs_vs_gvs)) if gvs_vs_gvs else np.nan
    orient = 1.0 if higher_is_better else -1.0
    return {
        "metric": metric_name,
        "higher_is_better": bool(higher_is_better),
        "mean_sham_vs_gvs": sham_mean,
        "mean_gvs_vs_gvs": gvs_mean,
        "oriented_sham_vs_gvs": orient * sham_mean if np.isfinite(sham_mean) else np.nan,
        "oriented_gvs_vs_gvs": orient * gvs_mean if np.isfinite(gvs_mean) else np.nan,
    }


def _add_rankings(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    for col in ["oriented_sham_vs_gvs", "oriented_gvs_vs_gvs"]:
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
    out["combined_rank_score"] = (
        0.5 * out["norm_oriented_sham_vs_gvs"] + 0.5 * out["norm_oriented_gvs_vs_gvs"]
    )
    out = out.sort_values("combined_rank_score", ascending=False)
    return out


def _global_classifier_summary(
    x: np.ndarray,
    y: np.ndarray,
    sham_label: str,
    cv_splits: int,
    random_state: int,
) -> pd.DataFrame:
    rows = []

    models = [
        (
            "linear_svm",
            lambda: SVC(kernel="linear", C=1.0, class_weight="balanced"),
        ),
        (
            "rbf_svm",
            lambda: SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
        ),
        (
            "knn",
            lambda: KNeighborsClassifier(n_neighbors=11, weights="distance"),
        ),
    ]

    tasks = []
    y_binary = np.where(np.asarray(y, dtype=object) == sham_label, sham_label, "gvs")
    tasks.append(("binary_sham_vs_all_gvs", x, y_binary))
    tasks.append(("multiclass_all_conditions", x, y))
    gvs_mask = np.array([str(lbl).lower().startswith("gvs") for lbl in y], dtype=bool)
    tasks.append(("multiclass_gvs_only", x[gvs_mask], y[gvs_mask]))

    for task_name, x_task, y_task in tasks:
        y_task = np.asarray(y_task)
        if x_task.shape[0] < 10:
            continue
        unique, counts = np.unique(y_task, return_counts=True)
        min_count = int(np.min(counts))
        n_splits = max(2, min(cv_splits, min_count))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
        }
        for model_name, model_factory in models:
            est = make_pipeline(StandardScaler(), model_factory())
            scores = cross_validate(
                est,
                x_task,
                y_task,
                scoring=scoring,
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )
            rows.append(
                {
                    "task": task_name,
                    "model": model_name,
                    "n_samples": int(x_task.shape[0]),
                    "n_classes": int(unique.shape[0]),
                    "cv_splits": int(n_splits),
                    "accuracy_mean": float(np.mean(scores["test_accuracy"])),
                    "accuracy_std": float(np.std(scores["test_accuracy"])),
                    "balanced_accuracy_mean": float(np.mean(scores["test_balanced_accuracy"])),
                    "balanced_accuracy_std": float(np.std(scores["test_balanced_accuracy"])),
                    "f1_macro_mean": float(np.mean(scores["test_f1_macro"])),
                    "f1_macro_std": float(np.std(scores["test_f1_macro"])),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.random_state))

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_files = discover_condition_files(input_dir)
    if not condition_files:
        raise FileNotFoundError(
            f"No files found in {input_dir} matching {FILE_PATTERN}."
        )
    if "sham" not in {k.lower() for k in condition_files}:
        raise RuntimeError("A sham condition is required (expected label 'sham').")

    ordered_conditions = sorted(condition_files.keys(), key=_condition_sort_key)
    sham_label = next(c for c in ordered_conditions if c.lower() == "sham")

    trial_counts = {}
    n_voxels = None
    for c in ordered_conditions:
        arr = np.asarray(np.load(condition_files[c], mmap_mode="r"))
        if arr.ndim != 2:
            raise ValueError(f"{c} array is not 2D: {arr.shape}")
        if n_voxels is None:
            n_voxels = int(arr.shape[0])
        elif int(arr.shape[0]) != n_voxels:
            raise ValueError(f"{c} voxel count {arr.shape[0]} != expected {n_voxels}")
        trial_counts[c] = int(arr.shape[1])

    min_trials = int(min(trial_counts.values()))
    trials_per_condition = (
        min_trials
        if args.trials_per_condition is None
        else int(min(args.trials_per_condition, min_trials))
    )
    if trials_per_condition < 20:
        raise RuntimeError(
            f"Trials per condition too low for stable metrics: {trials_per_condition}"
        )

    if args.voxel_selection == "topvar":
        voxel_idx = _choose_voxels_topvar(condition_files, int(args.voxel_subsample))
    else:
        voxel_idx = _choose_voxels_random(condition_files, int(args.voxel_subsample), rng)

    x_raw, y = _build_feature_matrix(
        files=condition_files,
        ordered_conditions=ordered_conditions,
        voxel_idx=voxel_idx,
        trials_per_condition=trials_per_condition,
        rng=rng,
    )

    x = SimpleImputer(strategy="median").fit_transform(x_raw)
    x = StandardScaler(with_mean=True, with_std=True).fit_transform(x)

    max_components = min(x.shape[0] - 1, x.shape[1], int(args.pca_components))
    if max_components < 3:
        raise RuntimeError(
            f"Too few PCA components available ({max_components})."
        )
    pca = PCA(n_components=max_components, svd_solver="randomized", random_state=int(args.random_state))
    x_pca = pca.fit_transform(x).astype(np.float64, copy=False)

    indices = _build_indices_by_condition(y, ordered_conditions)

    pairwise_metrics = {}
    pairwise_metrics.update(_pairwise_centroid_metrics(x_pca, ordered_conditions, indices))

    mmd_mat, mmd_pval = _pairwise_mmd_rbf(
        x=x_pca,
        ordered_conditions=ordered_conditions,
        indices=indices,
        n_permutations=int(args.mmd_permutations),
        mmd_max_trials=int(args.mmd_max_trials),
        rng=rng,
    )
    pairwise_metrics["nonlinear_rbf_mmd2"] = mmd_mat
    pairwise_metrics["nonlinear_rbf_mmd_pvalue"] = mmd_pval

    pairwise_metrics["linear_pairwise_svm_bal_acc"] = _pairwise_classifier_matrix(
        x=x_pca,
        y=y,
        ordered_conditions=ordered_conditions,
        model_factory=lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="linear", C=1.0, class_weight="balanced"),
        ),
        cv_splits=int(args.cv_splits),
        random_state=int(args.random_state),
    )
    pairwise_metrics["nonlinear_pairwise_rbf_svm_bal_acc"] = _pairwise_classifier_matrix(
        x=x_pca,
        y=y,
        ordered_conditions=ordered_conditions,
        model_factory=lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced"),
        ),
        cv_splits=int(args.cv_splits),
        random_state=int(args.random_state),
    )
    pairwise_metrics["nonlinear_pairwise_knn_bal_acc"] = _pairwise_classifier_matrix(
        x=x_pca,
        y=y,
        ordered_conditions=ordered_conditions,
        model_factory=lambda: make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=11, weights="distance"),
        ),
        cv_splits=int(args.cv_splits),
        random_state=int(args.random_state),
    )

    summary_rows = []
    higher_is_better_lookup = {
        "linear_centroid_cosine_distance": True,
        "linear_centroid_mahalanobis_distance": True,
        "nonlinear_rbf_mmd2": True,
        "nonlinear_rbf_mmd_pvalue": False,
        "linear_pairwise_svm_bal_acc": True,
        "nonlinear_pairwise_rbf_svm_bal_acc": True,
        "nonlinear_pairwise_knn_bal_acc": True,
    }

    for metric_name, matrix in pairwise_metrics.items():
        mat_df = pd.DataFrame(matrix, index=ordered_conditions, columns=ordered_conditions)
        csv_path = output_dir / f"{_safe_name(metric_name)}.csv"
        mat_df.to_csv(csv_path)
        heatmap_path = output_dir / f"{_safe_name(metric_name)}.png"
        _save_heatmap(matrix, ordered_conditions, metric_name, heatmap_path)

        summary_rows.append(
            _summarize_pairwise_matrix(
                metric_name=metric_name,
                matrix=matrix,
                ordered_conditions=ordered_conditions,
                higher_is_better=higher_is_better_lookup.get(metric_name, True),
            )
        )

    metric_summary = pd.DataFrame(summary_rows)
    metric_summary_ranked = _add_rankings(metric_summary)
    metric_summary_ranked.to_csv(output_dir / "metric_ranking_summary.csv", index=False)

    global_summary = _global_classifier_summary(
        x=x_pca,
        y=y,
        sham_label=sham_label,
        cv_splits=int(args.cv_splits),
        random_state=int(args.random_state),
    )
    global_summary.to_csv(output_dir / "global_classification_summary.csv", index=False)

    best_row = metric_summary_ranked.iloc[0].to_dict()
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "conditions": ordered_conditions,
        "trial_counts": {k: int(v) for k, v in trial_counts.items()},
        "trials_per_condition": int(trials_per_condition),
        "voxel_selection": args.voxel_selection,
        "voxel_subsample": int(voxel_idx.size),
        "pca_components": int(max_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "mmd_permutations": int(args.mmd_permutations),
        "mmd_max_trials": int(args.mmd_max_trials),
        "cv_splits": int(args.cv_splits),
        "random_state": int(args.random_state),
        "best_metric": best_row,
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    top_print = metric_summary_ranked.head(3)[
        ["metric", "mean_sham_vs_gvs", "mean_gvs_vs_gvs", "combined_rank_score"]
    ]
    print("Top metrics (higher combined_rank_score is better):", flush=True)
    print(top_print.to_string(index=False), flush=True)
    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
