from __future__ import annotations

import numpy as np

from .common import prepare_trial_by_node


METRIC_NAME = "partial_correlation"
METRIC_DESCRIPTION = "Inverse-covariance partial correlation between ROI nodes."


def compute_metric(roi_ts: np.ndarray, ridge: float = 1e-4) -> dict:
    x = prepare_trial_by_node(roi_ts, zscore=True)
    cov = np.cov(x, rowvar=False)
    if cov.ndim != 2:
        raise ValueError(f"Unexpected covariance shape: {cov.shape}")

    n = cov.shape[0]
    cov_reg = cov + float(ridge) * np.eye(n, dtype=np.float64)
    precision = np.linalg.pinv(cov_reg)

    d = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    mat = -precision / np.outer(d, d)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.clip(mat, -1.0, 1.0)
    np.fill_diagonal(mat, 1.0)

    return {
        "matrix": mat,
        "vmin": -1.0,
        "vmax": 1.0,
        "cmap": "jet",
        "directed": False,
        "description": METRIC_DESCRIPTION,
    }
