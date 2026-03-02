#!/usr/bin/env python3
"""Search connectivity-only methods that separate sham/GVS conditions.

This script computes condition-wise connectivity matrices using linear and
non-linear connectivity estimators, then scores each estimator by how much its
condition-distance matrix separates:
1) sham vs GVS conditions, and
2) different GVS conditions from each other.

No classification models are used in this workflow.
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
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_DIR = "results/connectivity"
DEFAULT_OUTPUT_DIR = "results/connectivity/tmp2_connectivity_methods"
FILE_PATTERN = "selected_beta_trials_*.npy"
FILE_PREFIX = "selected_beta_trials_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate linear/non-linear connectivity estimators across "
            "condition-split beta trial matrices."
        )
    )
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--voxel-subsample",
        type=int,
        default=4000,
        help="Number of voxels selected before PCA (default: 4000).",
    )
    parser.add_argument(
        "--voxel-selection",
        choices=["topvar", "random"],
        default="topvar",
        help="Voxel selection strategy before PCA projection.",
    )
    parser.add_argument(
        "--trials-per-condition",
        type=int,
        default=None,
        help="If unset, use min available trial count across conditions.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=40,
        help="Number of PCA components used as network nodes.",
    )
    parser.add_argument(
        "--nmi-bins",
        type=int,
        default=8,
        help="Quantile bins for binned normalized mutual information.",
    )
    parser.add_argument("--random-state", type=int, default=17)
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


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip())


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
        raise RuntimeError("No condition arrays found.")
    n_vox = int(pooled_var.shape[0])
    n_select = min(int(n_select), n_vox)
    idx = np.argpartition(pooled_var, -n_select)[-n_select:]
    return np.sort(idx.astype(np.int64, copy=False))


def _choose_voxels_random(files: Dict[str, Path], n_select: int, rng: np.random.Generator) -> np.ndarray:
    first = np.load(next(iter(files.values())), mmap_mode="r")
    if first.ndim != 2:
        raise ValueError(f"Condition array is not 2D: {first.shape}")
    n_vox = int(first.shape[0])
    n_select = min(int(n_select), n_vox)
    idx = rng.choice(n_vox, size=n_select, replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _sample_condition_trials(
    files: Dict[str, Path],
    ordered_conditions: List[str],
    voxel_idx: np.ndarray,
    trials_per_condition: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for c in ordered_conditions:
        arr = np.asarray(np.load(files[c], mmap_mode="r"), dtype=np.float32)
        x = arr[voxel_idx, :]
        n_trials = int(x.shape[1])
        if n_trials < trials_per_condition:
            raise ValueError(f"{c} has {n_trials} trials (< {trials_per_condition}).")
        trial_idx = rng.choice(n_trials, size=trials_per_condition, replace=False)
        out[c] = x[:, trial_idx].T.astype(np.float32, copy=False)  # trials x voxels
    return out


def _fit_shared_pca(
    x_by_condition: Dict[str, np.ndarray],
    ordered_conditions: List[str],
    n_components: int,
    random_state: int,
) -> Tuple[SimpleImputer, StandardScaler, PCA]:
    x_pool = np.vstack([x_by_condition[c] for c in ordered_conditions]).astype(np.float32, copy=False)

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler(with_mean=True, with_std=True)

    x_imp = imputer.fit_transform(x_pool)
    x_std = scaler.fit_transform(x_imp)

    n_components = min(n_components, x_std.shape[0] - 1, x_std.shape[1])
    if n_components < 3:
        raise RuntimeError(f"Too few PCA components available: {n_components}")

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(random_state))
    pca.fit(x_std)
    return imputer, scaler, pca


def _project_scores(
    x: np.ndarray,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    pca: PCA,
) -> np.ndarray:
    x_imp = imputer.transform(x)
    x_std = scaler.transform(x_imp)
    scores = pca.transform(x_std).astype(np.float64, copy=False)
    # Standardize nodes within condition to avoid scale dominance in connectivity.
    mu = np.mean(scores, axis=0, keepdims=True)
    sd = np.std(scores, axis=0, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (scores - mu) / sd


def _pearson_connectivity(scores: np.ndarray) -> np.ndarray:
    c = np.corrcoef(scores, rowvar=False).astype(np.float64, copy=False)
    c = np.nan_to_num(c, nan=0.0)
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _partial_corr_connectivity(scores: np.ndarray) -> np.ndarray:
    lw = LedoitWolf().fit(scores)
    precision = lw.precision_.astype(np.float64, copy=False)
    d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    p = -precision / np.outer(d, d)
    p = np.nan_to_num(p, nan=0.0)
    p = np.clip(p, -1.0, 1.0)
    np.fill_diagonal(p, 1.0)
    return p


def _spearman_connectivity(scores: np.ndarray) -> np.ndarray:
    ranks = pd.DataFrame(scores).rank(axis=0, method="average").to_numpy(dtype=np.float64)
    c = np.corrcoef(ranks, rowvar=False).astype(np.float64, copy=False)
    c = np.nan_to_num(c, nan=0.0)
    c = np.clip(c, -1.0, 1.0)
    np.fill_diagonal(c, 1.0)
    return c


def _distance_corr_connectivity(scores: np.ndarray) -> np.ndarray:
    n, p = scores.shape
    flat = np.empty((p, n * n), dtype=np.float32)
    for k in range(p):
        v = scores[:, k].astype(np.float64, copy=False)
        d = np.abs(v[:, None] - v[None, :])
        row_mean = d.mean(axis=1, keepdims=True)
        col_mean = d.mean(axis=0, keepdims=True)
        grand = float(d.mean())
        a = d - row_mean - col_mean + grand
        flat[k, :] = a.reshape(-1).astype(np.float32, copy=False)

    gram = (flat @ flat.T).astype(np.float64, copy=False) / float(n * n)
    diag = np.clip(np.diag(gram), 1e-12, None)
    denom = np.sqrt(np.outer(diag, diag))
    dcor2 = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
    dcor2 = np.clip(dcor2, 0.0, 1.0)
    dcor = np.sqrt(dcor2)
    np.fill_diagonal(dcor, 1.0)
    return dcor


def _binned_nmi_connectivity(scores: np.ndarray, n_bins: int) -> np.ndarray:
    p = scores.shape[1]
    ranks = pd.DataFrame(scores).rank(axis=0, method="average", pct=True).to_numpy(dtype=np.float64)
    bins = np.floor(ranks * n_bins).astype(np.int32, copy=False)
    bins = np.clip(bins, 0, n_bins - 1)

    out = np.eye(p, dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            val = float(normalized_mutual_info_score(bins[:, i], bins[:, j]))
            out[i, j] = val
            out[j, i] = val
    return out


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu].astype(np.float64, copy=False)


def _rms_edge_delta(v1: np.ndarray, v2: np.ndarray) -> float:
    d = v1 - v2
    return float(np.sqrt(np.mean(np.square(d))))


def _pairwise_condition_distance(
    conn_by_condition: Dict[str, np.ndarray],
    ordered_conditions: List[str],
) -> np.ndarray:
    vecs = {c: _upper_triangle(conn_by_condition[c]) for c in ordered_conditions}
    n = len(ordered_conditions)
    out = np.zeros((n, n), dtype=np.float64)
    for i, ci in enumerate(ordered_conditions):
        for j in range(i + 1, n):
            cj = ordered_conditions[j]
            val = _rms_edge_delta(vecs[ci], vecs[cj])
            out[i, j] = val
            out[j, i] = val
    return out


def _summarize_distance_matrix(
    method_id: str,
    dist_mat: np.ndarray,
    ordered_conditions: List[str],
) -> dict:
    sham_idx = next(i for i, c in enumerate(ordered_conditions) if c.lower() == "sham")
    gvs_idx = [i for i, c in enumerate(ordered_conditions) if c.lower().startswith("gvs")]

    sham_vs_gvs = [float(dist_mat[sham_idx, i]) for i in gvs_idx]
    gvs_vs_gvs = []
    for a in range(len(gvs_idx)):
        for b in range(a + 1, len(gvs_idx)):
            gvs_vs_gvs.append(float(dist_mat[gvs_idx[a], gvs_idx[b]]))

    return {
        "method": method_id,
        "mean_sham_vs_gvs_dist": float(np.mean(sham_vs_gvs)),
        "mean_gvs_vs_gvs_dist": float(np.mean(gvs_vs_gvs)),
        "min_sham_vs_gvs_dist": float(np.min(sham_vs_gvs)),
        "max_sham_vs_gvs_dist": float(np.max(sham_vs_gvs)),
        "min_gvs_vs_gvs_dist": float(np.min(gvs_vs_gvs)),
        "max_gvs_vs_gvs_dist": float(np.max(gvs_vs_gvs)),
    }


def _add_rank_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = ["mean_sham_vs_gvs_dist", "mean_gvs_vs_gvs_dist"]
    for col in cols:
        vals = out[col].to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        norm = np.full(vals.shape, np.nan, dtype=np.float64)
        if finite.any():
            vmin = float(np.min(vals[finite]))
            vmax = float(np.max(vals[finite]))
            if np.isclose(vmin, vmax):
                norm[finite] = 1.0
            else:
                norm[finite] = (vals[finite] - vmin) / (vmax - vmin)
        out[f"norm_{col}"] = norm

    a = out["norm_mean_sham_vs_gvs_dist"].to_numpy(dtype=np.float64)
    b = out["norm_mean_gvs_vs_gvs_dist"].to_numpy(dtype=np.float64)
    out["combined_score_avg"] = 0.5 * (a + b)
    out["combined_score_hmean"] = np.divide(
        2.0 * a * b,
        a + b,
        out=np.zeros_like(a),
        where=(a + b) > 0,
    )
    out = out.sort_values("combined_score_hmean", ascending=False)
    return out


def _save_heatmap(mat: np.ndarray, labels: List[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    finite = np.isfinite(mat)
    if finite.any():
        vmin = float(np.nanpercentile(mat[finite], 5.0))
        vmax = float(np.nanpercentile(mat[finite], 95.0))
        if np.isclose(vmin, vmax):
            vmin = float(np.nanmin(mat[finite]))
            vmax = float(np.nanmax(mat[finite]))
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
    else:
        vmin, vmax = -1.0, 1.0
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("RMS edge delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.random_state))

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_condition_files(input_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {input_dir} matching {FILE_PATTERN}")

    ordered_conditions = sorted(files.keys(), key=_condition_sort_key)
    sham_labels = [c for c in ordered_conditions if c.lower() == "sham"]
    if not sham_labels:
        raise RuntimeError("Could not find sham condition label.")
    sham_label = sham_labels[0]

    trial_counts = {}
    n_vox = None
    for c in ordered_conditions:
        arr = np.load(files[c], mmap_mode="r")
        if arr.ndim != 2:
            raise ValueError(f"{c} is not 2D: {arr.shape}")
        trial_counts[c] = int(arr.shape[1])
        if n_vox is None:
            n_vox = int(arr.shape[0])
        elif int(arr.shape[0]) != n_vox:
            raise ValueError(f"Voxel mismatch for {c}: {arr.shape[0]} != {n_vox}")

    min_trials = int(min(trial_counts.values()))
    trials_per_condition = (
        min_trials
        if args.trials_per_condition is None
        else int(min(args.trials_per_condition, min_trials))
    )
    if trials_per_condition < 20:
        raise RuntimeError(f"Too few trials per condition: {trials_per_condition}")

    if args.voxel_selection == "topvar":
        voxel_idx = _choose_voxels_topvar(files, int(args.voxel_subsample))
    else:
        voxel_idx = _choose_voxels_random(files, int(args.voxel_subsample), rng)

    x_by_condition = _sample_condition_trials(
        files=files,
        ordered_conditions=ordered_conditions,
        voxel_idx=voxel_idx,
        trials_per_condition=trials_per_condition,
        rng=rng,
    )

    imputer, scaler, pca = _fit_shared_pca(
        x_by_condition=x_by_condition,
        ordered_conditions=ordered_conditions,
        n_components=int(args.pca_components),
        random_state=int(args.random_state),
    )

    scores_by_condition = {}
    for c in ordered_conditions:
        scores_by_condition[c] = _project_scores(
            x=x_by_condition[c],
            imputer=imputer,
            scaler=scaler,
            pca=pca,
        )

    method_defs = [
        {
            "method_id": "linear_pearson_corr",
            "full_name": "Pearson Correlation Connectivity",
            "family": "linear",
        },
        {
            "method_id": "linear_partial_corr",
            "full_name": "Partial Correlation Connectivity (Ledoit-Wolf precision)",
            "family": "linear",
        },
        {
            "method_id": "nonlinear_spearman_corr",
            "full_name": "Spearman Rank Correlation Connectivity",
            "family": "nonlinear",
        },
        {
            "method_id": "nonlinear_distance_corr",
            "full_name": "Distance Correlation Connectivity",
            "family": "nonlinear",
        },
        {
            "method_id": "nonlinear_binned_nmi",
            "full_name": "Binned Normalized Mutual Information Connectivity",
            "family": "nonlinear",
        },
    ]

    conn_method_funcs = {
        "linear_pearson_corr": lambda s: _pearson_connectivity(s),
        "linear_partial_corr": lambda s: _partial_corr_connectivity(s),
        "nonlinear_spearman_corr": lambda s: _spearman_connectivity(s),
        "nonlinear_distance_corr": lambda s: _distance_corr_connectivity(s),
        "nonlinear_binned_nmi": lambda s: _binned_nmi_connectivity(s, int(args.nmi_bins)),
    }

    summary_rows = []
    for method in method_defs:
        method_id = method["method_id"]
        full_name = method["full_name"]

        conn_by_condition = {}
        for c in ordered_conditions:
            conn = conn_method_funcs[method_id](scores_by_condition[c]).astype(np.float64, copy=False)
            conn_by_condition[c] = conn
            np.save(output_dir / f"{_safe_name(method_id)}__{_safe_name(c)}_connectivity.npy", conn)

        dist_mat = _pairwise_condition_distance(conn_by_condition, ordered_conditions)
        dist_df = pd.DataFrame(dist_mat, index=ordered_conditions, columns=ordered_conditions)
        dist_df.to_csv(output_dir / f"{_safe_name(method_id)}__condition_distance.csv")
        _save_heatmap(
            mat=dist_mat,
            labels=ordered_conditions,
            title=f"{full_name}: condition distance",
            out_path=output_dir / f"{_safe_name(method_id)}__condition_distance.png",
        )

        row = _summarize_distance_matrix(method_id, dist_mat, ordered_conditions)
        row["full_name"] = full_name
        row["family"] = method["family"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_ranked = _add_rank_scores(summary_df)
    summary_ranked.to_csv(output_dir / "connectivity_method_ranking_summary.csv", index=False)
    pd.DataFrame(method_defs).to_csv(output_dir / "connectivity_method_definitions.csv", index=False)

    best_row = summary_ranked.iloc[0].to_dict()
    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "conditions": ordered_conditions,
        "sham_label": sham_label,
        "trial_counts": {k: int(v) for k, v in trial_counts.items()},
        "trials_per_condition": int(trials_per_condition),
        "voxel_selection": str(args.voxel_selection),
        "voxel_subsample": int(voxel_idx.size),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "nmi_bins": int(args.nmi_bins),
        "random_state": int(args.random_state),
        "best_method": best_row,
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Top methods by combined_score_hmean:", flush=True)
    print(
        summary_ranked[
            ["method", "full_name", "mean_sham_vs_gvs_dist", "mean_gvs_vs_gvs_dist", "combined_score_hmean"]
        ]
        .head(5)
        .to_string(index=False),
        flush=True,
    )
    print(f"Saved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
