from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from .common import ensure_2d_roi_ts


METRIC_NAME = "instantaneous_phase_sync"
METRIC_DESCRIPTION = "Instantaneous phase-locking value (PLV) between ROI node pairs."


def compute_metric(roi_ts: np.ndarray) -> dict:
    x = ensure_2d_roi_ts(roi_ts)
    analytic = hilbert(x, axis=1)
    phase = np.angle(analytic)

    n_nodes = phase.shape[0]
    mat = np.eye(n_nodes, dtype=np.float64)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            dphi = phase[i] - phase[j]
            val = np.abs(np.mean(np.exp(1j * dphi)))
            mat[i, j] = float(val)
            mat[j, i] = float(val)

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.clip(mat, 0.0, 1.0)
    np.fill_diagonal(mat, 1.0)

    return {
        "matrix": mat,
        "vmin": 0.0,
        "vmax": 1.0,
        "cmap": "jet",
        "directed": False,
        "description": METRIC_DESCRIPTION,
    }
