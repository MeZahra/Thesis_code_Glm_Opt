from __future__ import annotations

import numpy as np
from scipy import stats


def ensure_2d_roi_ts(roi_ts: np.ndarray) -> np.ndarray:
    x = np.asarray(roi_ts, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected ROI time series with shape (n_nodes, n_time), got {x.shape}")
    return x


def impute_nan_by_row_median(roi_ts: np.ndarray) -> np.ndarray:
    x = ensure_2d_roi_ts(roi_ts).copy()
    row_med = np.nanmedian(x, axis=1, keepdims=True)
    row_med = np.where(np.isfinite(row_med), row_med, 0.0)
    bad = ~np.isfinite(x)
    if np.any(bad):
        rows, cols = np.where(bad)
        x[rows, cols] = row_med[rows, 0]
    return x


def zscore_rows(roi_ts: np.ndarray) -> np.ndarray:
    x = ensure_2d_roi_ts(roi_ts)
    mu = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    out = (x - mu) / sd
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def prepare_trial_by_node(roi_ts: np.ndarray, zscore: bool = True) -> np.ndarray:
    x = impute_nan_by_row_median(roi_ts)
    if zscore:
        x = zscore_rows(x)
    return x.T.astype(np.float64, copy=False)


def safe_corrcoef_rows(series: np.ndarray, min_overlap: int = 3) -> np.ndarray:
    x = ensure_2d_roi_ts(series)
    n_rows = x.shape[0]
    out = np.full((n_rows, n_rows), np.nan, dtype=np.float64)
    for i in range(n_rows):
        out[i, i] = 1.0
    for i in range(n_rows):
        xi = x[i]
        for j in range(i + 1, n_rows):
            xj = x[j]
            valid = np.isfinite(xi) & np.isfinite(xj)
            n_valid = int(np.count_nonzero(valid))
            if n_valid < min_overlap:
                continue
            xi0 = xi[valid]
            xj0 = xj[valid]
            xi0 = xi0 - np.mean(xi0)
            xj0 = xj0 - np.mean(xj0)
            denom = np.sqrt(np.dot(xi0, xi0) * np.dot(xj0, xj0))
            if denom <= 1e-12:
                continue
            val = float(np.dot(xi0, xj0) / denom)
            val = float(np.clip(val, -1.0, 1.0))
            out[i, j] = val
            out[j, i] = val
    return out


def quantile_bins(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(0, dtype=np.int32)
    n_bins = int(max(2, n_bins))
    ranks = stats.rankdata(np.where(np.isfinite(x), x, np.nanmedian(x)), method="average")
    if x.size == 1:
        return np.zeros(1, dtype=np.int32)
    pct = (ranks - 1.0) / float(x.size - 1)
    bins = np.floor(pct * n_bins).astype(np.int32, copy=False)
    bins = np.clip(bins, 0, n_bins - 1)
    return bins


def robust_positive_vmax(matrix: np.ndarray, percentile: float = 99.0, fallback: float = 1e-3) -> float:
    m = np.asarray(matrix, dtype=np.float64)
    vals = np.abs(m[np.isfinite(m)])
    if vals.size == 0:
        return float(fallback)
    vmax = float(np.percentile(vals, float(np.clip(percentile, 50.0, 100.0))))
    return float(max(vmax, fallback))


def _build_lag_design(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have same length for Granger computation.")
    n = x.size
    p = int(max(1, max_lag))
    if n <= p + 4:
        return np.zeros(0), np.zeros((0, p)), np.zeros((0, p))

    target = y[p:]
    y_lags = [y[p - lag : n - lag] for lag in range(1, p + 1)]
    x_lags = [x[p - lag : n - lag] for lag in range(1, p + 1)]
    y_lag_mat = np.column_stack(y_lags)
    x_lag_mat = np.column_stack(x_lags)

    valid = np.isfinite(target)
    valid &= np.all(np.isfinite(y_lag_mat), axis=1)
    valid &= np.all(np.isfinite(x_lag_mat), axis=1)

    return target[valid], y_lag_mat[valid], x_lag_mat[valid]


def _residual_variance(target: np.ndarray, design: np.ndarray, ridge: float) -> float:
    if target.size == 0 or design.shape[0] != target.size:
        return np.nan
    x = np.column_stack([np.ones(target.size, dtype=np.float64), design])
    if ridge > 0:
        xtx = x.T @ x
        reg = float(ridge) * np.eye(xtx.shape[0], dtype=np.float64)
        beta = np.linalg.solve(xtx + reg, x.T @ target)
    else:
        beta, *_ = np.linalg.lstsq(x, target, rcond=None)
    resid = target - x @ beta
    return float(np.mean(np.square(resid)))


def granger_directional_strength(
    x_source: np.ndarray,
    y_target: np.ndarray,
    max_lag: int = 3,
    ridge: float = 1e-6,
    nonlinear: bool = False,
) -> float:
    target, y_lag, x_lag = _build_lag_design(x_source, y_target, max_lag=max_lag)
    if target.size == 0:
        return np.nan

    var_restricted = _residual_variance(target, y_lag, ridge=ridge)
    if not np.isfinite(var_restricted) or var_restricted <= 1e-12:
        return np.nan

    if nonlinear:
        nonlinear_terms = [x_lag * x_lag, y_lag * y_lag, x_lag * y_lag]
        design_full = np.column_stack([y_lag, x_lag] + nonlinear_terms)
    else:
        design_full = np.column_stack([y_lag, x_lag])

    var_full = _residual_variance(target, design_full, ridge=ridge)
    if not np.isfinite(var_full) or var_full <= 1e-12:
        return np.nan

    val = np.log((var_restricted + 1e-12) / (var_full + 1e-12))
    if not np.isfinite(val):
        return np.nan
    return float(max(0.0, val))
