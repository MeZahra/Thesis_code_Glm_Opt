from __future__ import annotations

import numpy as np
from scipy import stats

from .common import ensure_2d_roi_ts, impute_nan_by_row_median, robust_positive_vmax, zscore_rows


METRIC_NAME = "nonlinear_granger"
METRIC_DESCRIPTION = (
    "Pairwise directional kernel Granger-causality strength with FDR-filtered additional kernel "
    "components (source row -> target column)."
)


def _build_lag_design(x_source: np.ndarray, y_target: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_source, dtype=np.float64).ravel()
    y = np.asarray(y_target, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError("x_source and y_target must have the same length.")

    p = int(max(1, max_lag))
    n = x.size
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


def _center_gram_matrix(k: np.ndarray) -> np.ndarray:
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError(f"Expected square Gram matrix, got {k.shape}")
    if k.size == 0:
        return k
    row_mean = np.mean(k, axis=1, keepdims=True)
    col_mean = np.mean(k, axis=0, keepdims=True)
    grand_mean = np.mean(k)
    centered = k - row_mean - col_mean + grand_mean
    return 0.5 * (centered + centered.T)


def _pairwise_sq_dists_from_gram(gram: np.ndarray) -> np.ndarray:
    diag = np.diag(gram)
    d2 = diag[:, None] + diag[None, :] - 2.0 * gram
    return np.clip(d2, 0.0, None)


def _median_heuristic_sigma(design: np.ndarray) -> float:
    gram = design @ design.T
    d2 = _pairwise_sq_dists_from_gram(gram)
    upper = d2[np.triu_indices_from(d2, k=1)]
    finite = upper[np.isfinite(upper) & (upper > 0.0)]
    if finite.size == 0:
        return 1.0
    sigma = float(np.sqrt(np.median(finite)))
    return sigma if np.isfinite(sigma) and sigma > 0.0 else 1.0


def _kernel_matrix(
    design: np.ndarray,
    kernel: str,
    degree: int,
    sigma: float | None,
    ridge: float,
) -> np.ndarray:
    x = np.asarray(design, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D design matrix, got {x.shape}")
    n_samples = int(x.shape[0])
    if n_samples == 0:
        return np.zeros((0, 0), dtype=np.float64)

    gram = x @ x.T
    kernel_key = str(kernel).strip().lower()
    if kernel_key in {"ip", "poly", "polynomial", "inhomogeneous_polynomial", "p"}:
        poly_order = int(max(1, degree))
        k = np.power(1.0 + gram, poly_order)
    elif kernel_key in {"gaussian", "rbf", "g"}:
        d2 = _pairwise_sq_dists_from_gram(gram)
        width = float(sigma) if sigma is not None and np.isfinite(sigma) else _median_heuristic_sigma(x)
        width = max(width, 1e-8)
        k = np.exp(-d2 / (2.0 * width * width))
    else:
        raise ValueError(f"Unsupported kernel type: {kernel!r}. Use 'ip' or 'gaussian'.")

    k = _center_gram_matrix(k)
    if ridge > 0.0:
        k = k + float(ridge) * np.eye(n_samples, dtype=np.float64)
    return 0.5 * (k + k.T)


def _principal_components_from_gram(k: np.ndarray, eig_frac: float) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(k.shape[0])
    if n_samples == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    evals, evecs = np.linalg.eigh(k)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    if evals.size == 0:
        return np.zeros((n_samples, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    lam_max = float(evals[0])
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return np.zeros((n_samples, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    thresh = max(float(eig_frac), 1e-12) * lam_max
    keep = np.isfinite(evals) & (evals > thresh)
    if not np.any(keep):
        return np.zeros((n_samples, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    return evecs[:, keep], evals[keep]


def _fdr_bh_mask(pvals: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(pvals, dtype=np.float64).ravel()
    keep = np.zeros(p.size, dtype=bool)
    if p.size == 0 or alpha <= 0.0:
        return keep

    valid = np.isfinite(p)
    if not np.any(valid):
        return keep

    p_valid = p[valid]
    m = p_valid.size
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    crit = float(alpha) * np.arange(1, m + 1, dtype=np.float64) / float(m)
    passed = ranked <= crit
    if not np.any(passed):
        return keep

    kmax = int(np.max(np.flatnonzero(passed)))
    threshold = float(ranked[kmax])
    sel_valid = p_valid <= threshold
    keep[np.flatnonzero(valid)[sel_valid]] = True
    return keep


def _fisher_two_tailed_pvals(rvals: np.ndarray, n_samples: int) -> np.ndarray:
    r = np.clip(np.asarray(rvals, dtype=np.float64), -0.999999, 0.999999)
    if n_samples <= 3:
        return np.ones_like(r, dtype=np.float64)
    scale = np.sqrt(float(max(n_samples - 3, 1)))
    z = 0.5 * np.log((1.0 + r) / (1.0 - r)) * scale
    p = 2.0 * stats.norm.sf(np.abs(z))
    return np.where(np.isfinite(p), p, 1.0)


def kernel_granger_directional_strength(
    x_source: np.ndarray,
    y_target: np.ndarray,
    max_lag: int = 3,
    ridge: float = 1e-6,
    kernel: str = "ip",
    degree: int = 2,
    sigma: float | None = None,
    eig_frac: float = 1e-6,
    alpha: float = 0.05,
) -> float:
    target, y_lag, x_lag = _build_lag_design(x_source, y_target, max_lag=max_lag)
    if target.size == 0:
        return np.nan

    y = target - np.mean(target)
    y_norm = float(np.linalg.norm(y))
    if not np.isfinite(y_norm) or y_norm <= 1e-12:
        return np.nan
    y = y / y_norm

    design_restricted = y_lag
    design_full = np.column_stack([y_lag, x_lag])

    k_restricted = _kernel_matrix(
        design=design_restricted,
        kernel=kernel,
        degree=degree,
        sigma=sigma,
        ridge=ridge,
    )
    k_full = _kernel_matrix(
        design=design_full,
        kernel=kernel,
        degree=degree,
        sigma=sigma,
        ridge=ridge,
    )

    v_restricted, _ = _principal_components_from_gram(k_restricted, eig_frac=eig_frac)
    if v_restricted.shape[1] == 0:
        y_resid = y.copy()
    else:
        y_resid = y - v_restricted @ (v_restricted.T @ y)

    if not np.isfinite(np.linalg.norm(y_resid)) or np.linalg.norm(y_resid) <= 1e-12:
        return 0.0

    if v_restricted.shape[1] == 0:
        k_extra = k_full
    else:
        proj = v_restricted @ v_restricted.T
        k_extra = k_full - proj @ k_full - k_full @ proj + proj @ k_full @ proj
        k_extra = 0.5 * (k_extra + k_extra.T)

    v_extra, _ = _principal_components_from_gram(k_extra, eig_frac=eig_frac)
    if v_extra.shape[1] == 0:
        return 0.0

    y_center = y_resid - np.mean(y_resid)
    y_den = float(np.linalg.norm(y_center))
    if y_den <= 1e-12:
        return 0.0

    v_center = v_extra - np.mean(v_extra, axis=0, keepdims=True)
    v_den = np.linalg.norm(v_center, axis=0)
    valid = v_den > 1e-12
    if not np.any(valid):
        return 0.0

    rvals = np.zeros(v_center.shape[1], dtype=np.float64)
    rvals[valid] = (y_center @ v_center[:, valid]) / (y_den * v_den[valid])
    rvals = np.clip(rvals, -1.0, 1.0)

    pvals = _fisher_two_tailed_pvals(rvals, n_samples=int(target.size))
    sig = _fdr_bh_mask(pvals, alpha=float(alpha))
    if not np.any(sig):
        return 0.0

    val = float(np.sum(np.square(rvals[sig])))
    if not np.isfinite(val):
        return np.nan
    return max(0.0, val)


def compute_metric(
    roi_ts: np.ndarray,
    max_lag: int = 3,
    ridge: float = 1e-6,
    kernel: str = "ip",
    degree: int = 2,
    sigma: float | None = None,
    eig_frac: float = 1e-6,
    alpha: float = 0.05,
) -> dict:
    x = zscore_rows(impute_nan_by_row_median(ensure_2d_roi_ts(roi_ts)))
    n_nodes = x.shape[0]

    mat = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for src in range(n_nodes):
        for tgt in range(n_nodes):
            if src == tgt:
                continue
            mat[src, tgt] = kernel_granger_directional_strength(
                x_source=x[src],
                y_target=x[tgt],
                max_lag=int(max_lag),
                ridge=float(ridge),
                kernel=str(kernel),
                degree=int(degree),
                sigma=None if sigma is None else float(sigma),
                eig_frac=float(eig_frac),
                alpha=float(alpha),
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
        "kernel": str(kernel),
        "degree": int(degree),
        "sigma": None if sigma is None else float(sigma),
        "eig_frac": float(eig_frac),
        "alpha": float(alpha),
    }
