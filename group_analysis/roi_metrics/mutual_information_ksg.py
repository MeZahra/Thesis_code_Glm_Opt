from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

from .common import ensure_2d_roi_ts, impute_nan_by_row_median


METRIC_NAME = "mutual_information_ksg"
METRIC_DESCRIPTION = (
    "Pairwise continuous mutual information using the Kraskov-Stogbauer-Grassberger "
    "kNN estimator (KSG, estimator I), without non-negativity clipping."
)


def _ksg_mi_pairwise(x: np.ndarray, y: np.ndarray, k: int, jitter: float, rng: np.random.Generator) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size or x.size == 0:
        return np.nan

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n_samples = int(x.size)
    if n_samples <= 3:
        return np.nan

    k = int(np.clip(int(k), 1, n_samples - 1))
    jitter = float(max(0.0, jitter))
    if jitter > 0.0:
        x_scale = float(np.nanstd(x))
        y_scale = float(np.nanstd(y))
        if not np.isfinite(x_scale) or x_scale <= 1e-12:
            x_scale = 1.0
        if not np.isfinite(y_scale) or y_scale <= 1e-12:
            y_scale = 1.0
        x = x + rng.normal(0.0, jitter * x_scale, size=n_samples)
        y = y + rng.normal(0.0, jitter * y_scale, size=n_samples)

    xy = np.column_stack([x, y])
    joint_tree = cKDTree(xy)
    distances, _ = joint_tree.query(xy, k=k + 1, p=np.inf)
    eps = np.asarray(distances[:, k], dtype=np.float64)
    # Query marginal neighborhoods with strict "< eps" behavior.
    eps = np.nextafter(eps, 0.0)

    x_tree = cKDTree(x[:, None])
    y_tree = cKDTree(y[:, None])

    nx = np.empty(n_samples, dtype=np.int32)
    ny = np.empty(n_samples, dtype=np.int32)
    for idx in range(n_samples):
        px = [float(x[idx])]
        py = [float(y[idx])]
        radius = float(eps[idx])
        nx[idx] = max(0, len(x_tree.query_ball_point(px, r=radius, p=np.inf)) - 1)
        ny[idx] = max(0, len(y_tree.query_ball_point(py, r=radius, p=np.inf)) - 1)

    mi = float(digamma(k) + digamma(n_samples) - np.mean(digamma(nx + 1) + digamma(ny + 1)))
    if not np.isfinite(mi):
        return np.nan
    return mi


def compute_metric(roi_ts: np.ndarray, k: int = 3, jitter: float = 1e-10) -> dict:
    x = impute_nan_by_row_median(ensure_2d_roi_ts(roi_ts))
    n_nodes = x.shape[0]
    k = int(max(1, k))
    jitter = float(max(0.0, jitter))
    rng = np.random.default_rng(0)

    mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            val = _ksg_mi_pairwise(x[i], x[j], k=k, jitter=jitter, rng=rng)
            mat[i, j] = val
            mat[j, i] = val

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 0.0)

    finite = mat[np.isfinite(mat)]
    if finite.size == 0:
        vmin = -1e-3
        vmax = 1e-3
    else:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if not np.isfinite(vmin):
            vmin = -1e-3
        if not np.isfinite(vmax):
            vmax = 1e-3
        if vmax <= vmin:
            if abs(vmax) <= 1e-12:
                vmin = -1e-3
                vmax = 1e-3
            else:
                delta = max(1e-6, 0.05 * abs(vmax))
                vmin -= delta
                vmax += delta
    return {
        "matrix": mat,
        "vmin": float(vmin),
        "vmax": float(vmax),
        "cmap": "jet",
        "directed": False,
        "description": METRIC_DESCRIPTION,
        "k": int(k),
        "jitter": float(jitter),
        "units": "nats",
    }
