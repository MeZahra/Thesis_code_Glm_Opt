#!/usr/bin/env python3
"""Scatter-plot voxel weights from two NIfTI maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.stats import pearsonr


DEFAULT_X_MAP = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0_beta0_smooth0_gamma1_"
    "bold_original.nii.gz"
)
DEFAULT_Y_MAP = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_"
    "bold_original.nii.gz"
)
DEFAULT_OUT = "results/ablation/voxel_weights_original_scatter.png"
X_LABEL = "task-activation map"
Y_LABEL = "vigour-related map"


def _load_vector(path: str) -> tuple[np.ndarray, tuple[int, ...]]:
    img = nib.load(path)
    data = np.asarray(img.get_fdata(dtype=np.float32))
    return data.reshape(-1), data.shape


def plot_voxel_weight_scatter(x_map: str, y_map: str, out_path: str) -> None:
    x_values, x_shape = _load_vector(x_map)
    y_values, y_shape = _load_vector(y_map)

    if x_shape != y_shape:
        raise ValueError(f"Input maps must have the same shape; got {x_shape} and {y_shape}.")

    nonzero_voxels = (x_values != 0) | (y_values != 0)
    finite_voxels = np.isfinite(x_values) & np.isfinite(y_values)
    keep_voxels = nonzero_voxels & finite_voxels
    x_plot = x_values[keep_voxels]
    y_plot = y_values[keep_voxels]

    if x_plot.size == 0:
        raise ValueError("No voxels are non-zero in either input map.")

    r_value, p_value = pearsonr(x_plot, y_plot)

    fig, ax = plt.subplots(figsize=(7.5, 7.0), constrained_layout=True)
    ax.scatter(x_plot, y_plot, s=4, alpha=0.35, linewidths=0)
    ax.axhline(0, color="0.6", linewidth=0.8)
    ax.axvline(0, color="0.6", linewidth=0.8)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Non-zero voxels in either map (n={x_plot.size:,}, r={r_value:.3f})")
    ax.grid(True, linewidth=0.4, alpha=0.25)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)

    print(f"Saved scatter plot to {out}")
    print(f"Input shape: {x_shape}")
    print(f"Plotted voxels: {x_plot.size:,}")
    print(f"Excluded non-finite voxel pairs: {np.count_nonzero(nonzero_voxels & ~finite_voxels):,}")
    print(f"Pearson r: {r_value:.6f}")
    print(f"Pearson p-value: {p_value:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten two NIfTI maps and scatter-plot voxels that are non-zero "
            "in at least one map."
        )
    )
    parser.add_argument("--x-map", default=DEFAULT_X_MAP, help="NIfTI file for the x-axis.")
    parser.add_argument("--y-map", default=DEFAULT_Y_MAP, help="NIfTI file for the y-axis.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output image path.")
    args = parser.parse_args()

    plot_voxel_weight_scatter(args.x_map, args.y_map, args.out)


if __name__ == "__main__":
    main()
