#!/usr/bin/env python3
"""Create an interactive HTML view for two voxel-weight maps on anatomy."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import datasets, image, plotting


DEFAULT_MAP_1 = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_"
    "bold_thr90.nii.gz"
)
DEFAULT_MAP_2 = (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0_beta0_smooth0_gamma1_"
    "bold_thr90_postcentral_boosted.nii.gz"
)
DEFAULT_ANAT = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"
DEFAULT_OUT = "results/ablation/two_networks_overlay_sub9_ses1_thr90.html"
DEFAULT_CUT_COORDS = [0.0, -20.0, 52.0]


def _label_cmap() -> ListedColormap:
    return ListedColormap(
        [
            (0.0, 0.0, 0.0, 0.0),      # background
            (0.86, 0.18, 0.18, 1.0),   # map 1 only (red)
            (0.17, 0.42, 0.77, 1.0),   # map 2 only (blue)
            (0.55, 0.00, 0.55, 1.0),   # overlap (purple)
        ]
    )


def _resolve_cut_coords(
    map_2_img: nib.Nifti1Image, requested_cut_coords: list[float]
) -> tuple[float, float, float]:
    atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    atlas_img = image.resample_to_img(
        atlas_img,
        map_2_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    labels = [
        lbl.decode("utf-8", errors="replace") if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas.labels
    ]
    postcentral_idx = [i for i, lbl in enumerate(labels) if "postcentral gyrus" in lbl.lower()]
    if not postcentral_idx:
        return tuple(requested_cut_coords)

    postcentral_mask = np.isin(atlas_img.get_fdata().astype(int), postcentral_idx)
    active_postcentral = (map_2_img.get_fdata() > 0) & postcentral_mask
    if not np.any(active_postcentral):
        return tuple(requested_cut_coords)

    ijk = np.argwhere(active_postcentral)
    xyz = nib.affines.apply_affine(map_2_img.affine, ijk)
    return tuple(np.mean(xyz, axis=0).tolist())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay two voxel-weight maps on one anatomical figure."
    )
    parser.add_argument("--map-1", default=DEFAULT_MAP_1, help="Path to first NIfTI map.")
    parser.add_argument("--map-2", default=DEFAULT_MAP_2, help="Path to second NIfTI map.")
    parser.add_argument("--anat", default=DEFAULT_ANAT, help="Path to anatomical NIfTI.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output HTML path.")
    parser.add_argument(
        "--out-map-1",
        default=None,
        help="Optional output HTML path for the first map viewer.",
    )
    parser.add_argument(
        "--out-map-2",
        default=None,
        help="Optional output HTML path for the second map viewer.",
    )
    parser.add_argument(
        "--cut-coords",
        nargs=3,
        type=float,
        default=DEFAULT_CUT_COORDS,
        metavar=("X", "Y", "Z"),
        help="Ortho cut coordinates. Defaults to active postcentral center when present.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    out_path = Path(args.out)
    out_map_1_path = (
        Path(args.out_map_1)
        if args.out_map_1 is not None
        else out_path.with_name(f"{out_path.stem}_map1.html")
    )
    out_map_2_path = (
        Path(args.out_map_2)
        if args.out_map_2 is not None
        else out_path.with_name(f"{out_path.stem}_map2.html")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_1_path.parent.mkdir(parents=True, exist_ok=True)
    out_map_2_path.parent.mkdir(parents=True, exist_ok=True)

    anat_img = image.load_img(args.anat)
    map_1_img = image.resample_to_img(
        image.load_img(args.map_1),
        anat_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )
    map_2_img = image.resample_to_img(
        image.load_img(args.map_2),
        anat_img,
        interpolation="continuous",
        force_resample=True,
        copy_header=True,
    )

    combined_img = image.math_img(
        "1.0 * (img1 > 0) + 2.0 * (img2 > 0)",
        img1=map_1_img,
        img2=map_2_img,
    )
    map_1_binary_img = image.math_img("1.0 * (img > 0)", img=map_1_img)
    map_2_binary_img = image.math_img("1.0 * (img > 0)", img=map_2_img)
    combined_img = image.new_img_like(anat_img, combined_img.get_fdata().astype("float32"))
    map_1_binary_img = image.new_img_like(
        anat_img, map_1_binary_img.get_fdata().astype("float32")
    )
    map_2_binary_img = image.new_img_like(
        anat_img, map_2_binary_img.get_fdata().astype("float32")
    )
    cut_coords = _resolve_cut_coords(map_2_binary_img, args.cut_coords)

    title = (
        "Red: map1 | Blue: map2 (postcentral-boosted) | Purple: overlap"
    )
    view = plotting.view_img(
        combined_img,
        bg_img=anat_img,
        cut_coords=cut_coords,
        threshold=0.5,
        vmax=3.0,
        symmetric_cmap=False,
        cmap=_label_cmap(),
        colorbar=True,
        title=title,
    )
    view.save_as_html(str(out_path))
    map_1_view = plotting.view_img(
        map_1_binary_img,
        bg_img=anat_img,
        cut_coords=cut_coords,
        threshold=0.5,
        vmax=1.0,
        symmetric_cmap=False,
        cmap="Reds",
        colorbar=True,
        title="Map 1",
    )
    map_1_view.save_as_html(str(out_map_1_path))
    map_2_view = plotting.view_img(
        map_2_binary_img,
        bg_img=anat_img,
        cut_coords=cut_coords,
        threshold=0.5,
        vmax=1.0,
        symmetric_cmap=False,
        cmap="Blues",
        colorbar=True,
        title="Map 2 (postcentral-boosted for display)",
    )
    map_2_view.save_as_html(str(out_map_2_path))
    print(f"Saved HTML viewer: {out_path}")
    print(f"Saved map 1 HTML viewer: {out_map_1_path}")
    print(f"Saved map 2 HTML viewer: {out_map_2_path}")


if __name__ == "__main__":
    main()
