from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .common import ensure_2d_roi_ts


METRIC_NAME = "wavelet_transform_coherence"
METRIC_DESCRIPTION = (
    "Average Morlet wavelet coherence between ROI node pairs "
    "with scale-adaptive smoothing, frequency-band selection, and COI masking."
)


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


def _smooth_scale_adaptive(
    plane: np.ndarray,
    widths: np.ndarray,
    smooth_scale_sigma: float,
    smooth_time_sigma: float,
) -> np.ndarray:
    out = np.asarray(plane, dtype=np.float64)
    if out.ndim != 2:
        raise ValueError(f"Expected 2D scale-time plane, got shape {out.shape}")
    if out.shape[0] != widths.size:
        raise ValueError(
            f"Scale axis mismatch: plane has {out.shape[0]} rows but widths has {widths.size} values"
        )

    out = out.copy()
    if smooth_scale_sigma > 0.0:
        out = gaussian_filter1d(out, sigma=float(smooth_scale_sigma), axis=0, mode="nearest")

    # Time smoothing grows with scale, reducing low-frequency coherence inflation.
    if smooth_time_sigma > 0.0:
        base = float(smooth_time_sigma)
        for k, scale in enumerate(widths):
            sigma_t = base * float(scale)
            if sigma_t <= 1e-6:
                continue
            out[k] = gaussian_filter1d(out[k], sigma=sigma_t, mode="nearest")
    return out


def _scale_to_frequency_hz(widths: np.ndarray, omega0: float, tr: float) -> np.ndarray:
    # Morlet pseudo-frequency (cycles/sec) for sampling interval TR.
    return float(omega0) / (2.0 * np.pi * widths * float(tr))


def _cone_of_influence_mask(
    n_time: int,
    widths: np.ndarray,
    tr: float,
    coi_factor: float,
) -> np.ndarray:
    idx = np.arange(n_time, dtype=np.float64)
    edge_dist_sec = np.minimum(idx, (n_time - 1) - idx) * float(tr)
    coi_half_width_sec = float(coi_factor) * widths * float(tr)
    return edge_dist_sec[None, :] >= coi_half_width_sec[:, None]


def compute_metric(
    roi_ts: np.ndarray,
    min_scale: int = 2,
    max_scale: int = 20,
    omega0: float = 6.0,
    smooth_scale_sigma: float = 1.0,
    smooth_time_sigma: float = 2.0,
    tr: float = 1.0,
    fmin_hz: float | None = 0.01,
    fmax_hz: float | None = 0.1,
    mask_coi: bool = True,
    coi_factor: float = np.sqrt(2.0),
    range_tolerance: float = 1e-3,
) -> dict:
    x = ensure_2d_roi_ts(roi_ts)
    if tr <= 0.0:
        raise ValueError(f"TR must be positive, got {tr}")

    min_scale = int(max(1, min_scale))
    max_scale = int(max(min_scale + 1, max_scale))
    widths = np.arange(min_scale, max_scale + 1, dtype=np.float64)
    freqs_hz = _scale_to_frequency_hz(widths, omega0=float(omega0), tr=float(tr))

    if fmin_hz is None and fmax_hz is None:
        band_lo = None
        band_hi = None
        band_mask = np.ones(widths.size, dtype=bool)
    else:
        if fmin_hz is None:
            fmin_hz = float(np.nanmin(freqs_hz))
        if fmax_hz is None:
            fmax_hz = float(np.nanmax(freqs_hz))
        band_lo = float(min(fmin_hz, fmax_hz))
        band_hi = float(max(fmin_hz, fmax_hz))
        band_mask = (freqs_hz >= band_lo) & (freqs_hz <= band_hi)
        if not np.any(band_mask):
            raise ValueError(
                f"No scales in requested frequency band [{band_lo:.6f}, {band_hi:.6f}] Hz. "
                f"Available range is [{float(np.min(freqs_hz)):.6f}, {float(np.max(freqs_hz)):.6f}] Hz."
            )

    n_nodes, n_time = x.shape
    coeffs = [_morlet_cwt(x[i], widths=widths, omega0=float(omega0)) for i in range(n_nodes)]

    mat = np.eye(n_nodes, dtype=np.float64)
    smooth_scale_sigma = float(max(0.0, smooth_scale_sigma))
    smooth_time_sigma = float(max(0.0, smooth_time_sigma))
    range_tolerance = float(max(0.0, range_tolerance))
    inv_widths = 1.0 / widths[:, None]

    auto_power = []
    for i in range(n_nodes):
        wx = coeffs[i]
        sxx = _smooth_scale_adaptive(
            (np.abs(wx) ** 2) * inv_widths,
            widths=widths,
            smooth_scale_sigma=smooth_scale_sigma,
            smooth_time_sigma=smooth_time_sigma,
        )
        auto_power.append(np.maximum(sxx, 1e-12))

    coi_mask = (
        _cone_of_influence_mask(n_time=n_time, widths=widths, tr=float(tr), coi_factor=float(coi_factor))
        if bool(mask_coi)
        else np.ones((widths.size, n_time), dtype=bool)
    )
    support_mask = coi_mask & band_mask[:, None]
    invalid_fractions: list[float] = []

    for i in range(n_nodes):
        wx = coeffs[i]
        sxx = auto_power[i]
        for j in range(i + 1, n_nodes):
            wy = coeffs[j]
            syy = auto_power[j]

            cross = (wx * np.conj(wy)) * inv_widths
            cross_real = _smooth_scale_adaptive(
                np.real(cross),
                widths=widths,
                smooth_scale_sigma=smooth_scale_sigma,
                smooth_time_sigma=smooth_time_sigma,
            )
            cross_imag = _smooth_scale_adaptive(
                np.imag(cross),
                widths=widths,
                smooth_scale_sigma=smooth_scale_sigma,
                smooth_time_sigma=smooth_time_sigma,
            )
            cross_power = (cross_real * cross_real) + (cross_imag * cross_imag)

            coherence = cross_power / (sxx * syy + 1e-12)
            coherence = np.real(coherence)
            finite = np.isfinite(coherence) & support_mask
            out_of_range = finite & (
                (coherence < -range_tolerance) | (coherence > (1.0 + range_tolerance))
            )
            keep = finite & (~out_of_range)

            n_finite = int(np.count_nonzero(finite))
            invalid_fractions.append(float(np.count_nonzero(out_of_range)) / float(max(1, n_finite)))

            if np.any(keep):
                kept_values = np.clip(coherence[keep], 0.0, 1.0)
                val = float(np.mean(kept_values))
            else:
                val = np.nan

            mat[i, j] = val
            mat[j, i] = val

    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    mat = np.where(mat < -range_tolerance, 0.0, mat)
    mat = np.where(mat > (1.0 + range_tolerance), 0.0, mat)
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
        "tr": float(tr),
        "fmin_hz": band_lo,
        "fmax_hz": band_hi,
        "frequency_hz_min": float(np.min(freqs_hz)),
        "frequency_hz_max": float(np.max(freqs_hz)),
        "n_scales_total": int(widths.size),
        "n_scales_in_band": int(np.count_nonzero(band_mask)),
        "mask_coi": bool(mask_coi),
        "coi_factor": float(coi_factor),
        "range_tolerance": float(range_tolerance),
        "mean_out_of_range_fraction": (
            float(np.mean(invalid_fractions)) if invalid_fractions else 0.0
        ),
    }
