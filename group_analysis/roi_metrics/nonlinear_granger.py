from __future__ import annotations

import numpy as np

from .common import ensure_2d_roi_ts, granger_directional_strength, impute_nan_by_row_median, robust_positive_vmax


METRIC_NAME = "nonlinear_granger"
METRIC_DESCRIPTION = (
    "Pairwise directional nonlinear Granger strength with quadratic and interaction lag terms "
    "(source row -> target column)."
)


def compute_metric(roi_ts: np.ndarray, max_lag: int = 3, ridge: float = 1e-6) -> dict:
    x = impute_nan_by_row_median(ensure_2d_roi_ts(roi_ts))
    n_nodes = x.shape[0]

    mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for src in range(n_nodes):
        for tgt in range(n_nodes):
            if src == tgt:
                continue
            mat[src, tgt] = granger_directional_strength(
                x_source=x[src],
                y_target=x[tgt],
                max_lag=int(max_lag),
                ridge=float(ridge),
                nonlinear=True,
            )

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 0.0)

    vmax = robust_positive_vmax(mat, percentile=99.0, fallback=1e-3)
    return {
        "matrix": mat,
        "vmin": 0.0,
        "vmax": float(vmax),
        "cmap": "jet",
        "directed": True,
        "description": METRIC_DESCRIPTION,
        "max_lag": int(max_lag),
    }
