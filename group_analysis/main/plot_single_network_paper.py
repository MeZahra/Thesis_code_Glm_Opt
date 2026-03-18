#!/usr/bin/env python3
"""Create a paper-ready multi-plane figure for a single thresholded voxel map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import image, plotting
from scipy import ndimage


DEFAULT_MAP = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_"
    "bold_thr90.nii.gz"
)
DEFAULT_ANAT = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"
DEFAULT_OUT_PNG = "results/ablation/map1_multiplane_contour_all_regions.png"
DEFAULT_OUT_PDF = "results/ablation/map1_multiplane_contour_all_regions.pdf"


def _slice_coord_from_index(
    affine: np.ndarray, shape: tuple[int, int, int], axis_idx: int, slice_idx: int
) -> float:
    ijk = np.array([(dim - 1) / 2.0 for dim in shape], dtype=float)
    ijk[axis_idx] = float(slice_idx)
    xyz = image.coord_transform(ijk[0], ijk[1], ijk[2], affine)
    return float(xyz[axis_idx])


def _cover_all_components_cut_coords(
    roi_img,
) -> dict[str, list[float]]:
    data = roi_img.get_fdata() > 0
    if not np.any(data):
        raise ValueError("The input map contains no active voxels.")

    labels, n_components = ndimage.label(data)
    coords: dict[str, list[float]] = {}
    for axis_idx, axis_name in enumerate(("x", "y", "z")):
        other_axes = tuple(i for i in range(3) if i != axis_idx)
        active_indices = np.flatnonzero(np.count_nonzero(data, axis=other_axes))
        slice_components: dict[int, set[int]] = {}
        slice_voxels: dict[int, int] = {}
        for slice_idx in active_indices:
            plane = np.take(labels, slice_idx, axis=axis_idx)
            component_ids = set(np.unique(plane).tolist())
            component_ids.discard(0)
            if not component_ids:
                continue
            slice_components[int(slice_idx)] = component_ids
            slice_voxels[int(slice_idx)] = int(np.count_nonzero(plane))

        uncovered = set(range(1, n_components + 1))
        selected_indices: list[int] = []
        while uncovered:
            best_idx: int | None = None
            best_key: tuple[int, int] | None = None
            for slice_idx, component_ids in slice_components.items():
                new_hits = component_ids & uncovered
                if not new_hits:
                    continue
                candidate_key = (len(new_hits), slice_voxels[slice_idx])
                if best_key is None or candidate_key > best_key:
                    best_idx = slice_idx
                    best_key = candidate_key
            if best_idx is None:
                break
            selected_indices.append(best_idx)
            uncovered -= slice_components[best_idx]

        if uncovered:
            raise RuntimeError(
                f"Could not find slices covering all components for axis {axis_name}: "
                f"{sorted(uncovered)}"
            )

        coords[axis_name] = [
            _slice_coord_from_index(roi_img.affine, data.shape, axis_idx, slice_idx)
            for slice_idx in sorted(selected_indices)
        ]
    return coords


def _save_figure(
    roi_img,
    anat_img,
    out_png: Path | None,
    out_pdf: Path | None,
    alpha: float,
    outline_width: float,
) -> dict[str, list[float]]:
    if out_png is None and out_pdf is None:
        raise ValueError("At least one output path must be provided.")

    cut_coords = _cover_all_components_cut_coords(roi_img)
    max_cuts = max(len(cuts) for cuts in cut_coords.values())
    fig_width = max(8.5, 2.35 * max_cuts)
    fig, axes = plt.subplots(3, 1, figsize=(fig_width, 7.9), facecolor="white")
    plt.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.03, hspace=0.04)

    fill_cmap = ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),
            (0.95, 0.2, 0.15, 1.0),
        ]
    )
    for ax, display_mode in zip(axes, ("x", "y", "z")):
        display = plotting.plot_roi(
            roi_img,
            bg_img=anat_img,
            axes=ax,
            display_mode=display_mode,
            cut_coords=cut_coords[display_mode],
            cmap=fill_cmap,
            threshold=0.5,
            alpha=alpha,
            black_bg=False,
            dim=-0.2,
            draw_cross=False,
            annotate=True,
            colorbar=False,
        )
        display.add_contours(
            roi_img,
            levels=[0.5],
            colors=["#c91f17"],
            linewidths=outline_width,
        )

    for out_path in (out_png, out_pdf):
        if out_path is None:
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved paper figure: {out_path}")

    plt.close(fig)
    return cut_coords


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a title-free multi-plane paper figure for a single map."
    )
    parser.add_argument("--map", default=DEFAULT_MAP, help="Path to the input NIfTI map.")
    parser.add_argument("--anat", default=DEFAULT_ANAT, help="Path to the anatomical NIfTI.")
    parser.add_argument("--out-png", default=DEFAULT_OUT_PNG, help="Output PNG path.")
    parser.add_argument("--out-pdf", default=DEFAULT_OUT_PDF, help="Output PDF path.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.72,
        help="Overlay alpha for the filled mask.",
    )
    parser.add_argument(
        "--outline-width",
        type=float,
        default=1.6,
        help="Contour line width.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    anat_img = image.load_img(args.anat)
    map_img = image.resample_to_img(
        image.load_img(args.map),
        anat_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    roi_img = image.math_img("1.0 * (img > 0)", img=map_img)
    roi_img = image.new_img_like(anat_img, roi_img.get_fdata().astype("float32"))
    cut_coords = _save_figure(
        roi_img=roi_img,
        anat_img=anat_img,
        out_png=Path(args.out_png) if args.out_png else None,
        out_pdf=Path(args.out_pdf) if args.out_pdf else None,
        alpha=args.alpha,
        outline_width=args.outline_width,
    )
    for axis_name in ("x", "y", "z"):
        coords_str = ", ".join(f"{coord:.1f}" for coord in cut_coords[axis_name])
        print(f"{axis_name}-cuts: {coords_str}")


if __name__ == "__main__":
    main()
