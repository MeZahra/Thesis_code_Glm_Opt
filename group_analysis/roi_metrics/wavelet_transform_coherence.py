from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter

from .common import ensure_2d_roi_ts


METRIC_NAME = "wavelet_transform_coherence"
METRIC_DESCRIPTION = "Average Morlet wavelet coherence between ROI node pairs."


def _morlet_cwt(x: np.ndarray, widths: np.ndarray, omega0: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).ravel()
    coefs = np.empty((widths.size, x.size), dtype=np.complex128)

    def _morlet_wavelet(length: int, scale: float, w0: float) -> np.ndarray:
        t = np.arange(length, dtype=np.float64) - (length - 1.0) / 2.0
        xs = t / float(scale)
        envelope = np.exp(-0.5 * xs * xs)
        carrier = np.exp(1j * float(w0) * xs)
        # Same normalization used in scipy.signal.morlet2
        return (np.pi ** (-0.25)) * np.sqrt(1.0 / float(scale)) * envelope * carrier

    for idx, width in enumerate(widths):
        length = int(max(16, np.ceil(12.0 * float(width))))
        if length % 2 == 0:
            length += 1
        wavelet = _morlet_wavelet(length, scale=float(width), w0=float(omega0))
        coefs[idx] = signal.fftconvolve(x, np.conj(wavelet[::-1]), mode="same")
    return coefs


def compute_metric(
    roi_ts: np.ndarray,
    min_scale: int = 2,
    max_scale: int = 20,
    omega0: float = 6.0,
    smooth_scale_sigma: float = 1.0,
    smooth_time_sigma: float = 2.0,
) -> dict:
    x = ensure_2d_roi_ts(roi_ts)

    min_scale = int(max(1, min_scale))
    max_scale = int(max(min_scale + 1, max_scale))
    widths = np.arange(min_scale, max_scale + 1, dtype=np.float64)

    n_nodes = x.shape[0]
    coeffs = [_morlet_cwt(x[i], widths=widths, omega0=float(omega0)) for i in range(n_nodes)]

    mat = np.eye(n_nodes, dtype=np.float64)
    sigma = (float(max(0.0, smooth_scale_sigma)), float(max(0.0, smooth_time_sigma)))

    for i in range(n_nodes):
        wx = coeffs[i]
        sxx = gaussian_filter(np.abs(wx) ** 2, sigma=sigma)
        for j in range(i + 1, n_nodes):
            wy = coeffs[j]
            syy = gaussian_filter(np.abs(wy) ** 2, sigma=sigma)

            cross = wx * np.conj(wy)
            cross_real = gaussian_filter(np.real(cross), sigma=sigma)
            cross_imag = gaussian_filter(np.imag(cross), sigma=sigma)
            cross_power = (cross_real * cross_real) + (cross_imag * cross_imag)

            coherence = cross_power / (sxx * syy + 1e-12)
            val = float(np.nanmean(np.clip(np.real(coherence), 0.0, 1.0)))
            mat[i, j] = val
            mat[j, i] = val

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
        "min_scale": int(min_scale),
        "max_scale": int(max_scale),
        "omega0": float(omega0),
    }
