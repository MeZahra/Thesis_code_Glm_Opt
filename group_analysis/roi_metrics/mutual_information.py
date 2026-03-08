from __future__ import annotations

import numpy as np

from .common import (
    ensure_2d_roi_ts,
    impute_nan_by_row_median,
    quantile_bins,
    robust_positive_vmax,
)


METRIC_NAME = "mutual_information"
METRIC_DESCRIPTION = "Pairwise discrete mutual information between ROI node trial series."


def _mi_from_bins(xb: np.ndarray, yb: np.ndarray, n_bins: int) -> float:
    if xb.size != yb.size or xb.size == 0:
        return np.nan

    hist2d = np.histogram2d(
        xb,
        yb,
        bins=[n_bins, n_bins],
        range=[[0, n_bins], [0, n_bins]],
    )[0].astype(np.float64)

    total = float(np.sum(hist2d))
    if total <= 0:
        return np.nan

    pxy = hist2d / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    pxpy = np.outer(px, py)
    valid = (pxy > 0) & (pxpy > 0)
    mi = float(np.sum(pxy[valid] * np.log((pxy[valid] + 1e-12) / (pxpy[valid] + 1e-12))))
    return mi


def compute_metric(roi_ts: np.ndarray, n_bins: int = 8) -> dict:
    x = impute_nan_by_row_median(ensure_2d_roi_ts(roi_ts))
    n_nodes = x.shape[0]
    n_bins = int(max(2, n_bins))

    binned = np.vstack([quantile_bins(x[i], n_bins=n_bins) for i in range(n_nodes)])

    mat = np.eye(n_nodes, dtype=np.float64)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            val = _mi_from_bins(binned[i], binned[j], n_bins=n_bins)
            mat[i, j] = val
            mat[j, i] = val

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 0.0)
    vmax = robust_positive_vmax(mat, percentile=99.0, fallback=1e-3)

    return {
        "matrix": mat,
        "vmin": 0.0,
        "vmax": float(vmax),
        "cmap": "jet",
        "directed": False,
        "description": METRIC_DESCRIPTION,
        "n_bins": int(n_bins),
        "units": "nats",
    }
