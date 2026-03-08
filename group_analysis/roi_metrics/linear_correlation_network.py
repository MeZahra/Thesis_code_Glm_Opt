from __future__ import annotations

import numpy as np

from .common import ensure_2d_roi_ts, safe_corrcoef_rows


METRIC_NAME = "linear_correlation_network"
METRIC_DESCRIPTION = "Pearson correlation between ROI-averaged node trial series."


def compute_metric(roi_ts: np.ndarray) -> dict:
    x = ensure_2d_roi_ts(roi_ts)

    linear_corr = safe_corrcoef_rows(x, min_overlap=3)
    linear_corr = np.nan_to_num(linear_corr, nan=0.0, posinf=0.0, neginf=0.0)
    linear_corr = np.clip(linear_corr, -1.0, 1.0)
    np.fill_diagonal(linear_corr, 1.0)

    return {
        "matrix": linear_corr,
        "vmin": -1.0,
        "vmax": 1.0,
        "cmap": "jet",
        "directed": False,
        "description": METRIC_DESCRIPTION,
    }
