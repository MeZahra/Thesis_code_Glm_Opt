#!/usr/bin/env python3
"""Spin-test null for the two-network ablation overlay.

This follows the network correspondence procedure described in
``s41467-025-58176-9.pdf``: volumetric maps are projected to fsaverage6
surface space, the reference map is randomly rotated on the sphere, and
Dice overlap with the fixed target map is recomputed for each rotation.
The primary p-value uses the NCT package's finite-sample correction:
``(n_null_greater + 1) / (n_permutations + 1)``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nilearn import datasets, surface
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_1 = REPO_ROOT / (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_"
    "bold_thr90.nii.gz"
)
DEFAULT_MAP_2 = REPO_ROOT / (
    "results/ablation/"
    "voxel_weights_mean_foldavg_sub9_ses1_task0.8_bold0_beta0_smooth0_gamma1_"
    "bold_thr90_postcentral_boosted.nii.gz"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT / "results/ablation/two_networks_overlay_sub9_ses1_thr90_spin_test.json"
)
DEFAULT_NULL_CSV = (
    REPO_ROOT / "results/ablation/two_networks_overlay_sub9_ses1_thr90_spin_null.csv"
)


def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    denom = int(mask_a.sum()) + int(mask_b.sum())
    if denom == 0:
        return float("nan")
    return float(2 * int(np.count_nonzero(mask_a & mask_b)) / denom)


def _load_volume_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return np.asarray(img.get_fdata() > 0, dtype=bool), img


def _surface_mask(
    img_path: Path,
    fsaverage: Any,
    hemi: str,
    *,
    interpolation: str,
    n_samples: int,
    surface_threshold: float,
) -> np.ndarray:
    values = surface.vol_to_surf(
        str(img_path),
        fsaverage[f"pial_{hemi}"],
        inner_mesh=fsaverage[f"white_{hemi}"],
        interpolation=interpolation,
        kind="depth",
        n_samples=n_samples,
    )
    values = np.nan_to_num(values, nan=0.0)
    return np.asarray(values > surface_threshold, dtype=bool)


def _sphere_coords(sphere_path: str | Path) -> np.ndarray:
    coords, _faces = surface.load_surf_mesh(str(sphere_path))
    coords = np.asarray(coords, dtype=np.float64)
    coords = coords - coords.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"Sphere mesh has zero-radius vertices: {sphere_path}")
    return coords / norms


def _tree_query(tree: cKDTree, coords: np.ndarray) -> np.ndarray:
    try:
        return tree.query(coords, k=1, workers=-1)[1]
    except TypeError:
        return tree.query(coords, k=1)[1]


def _spin_null_dice(
    reference_left: np.ndarray,
    reference_right: np.ndarray,
    target_left: np.ndarray,
    target_right: np.ndarray,
    sphere_left: np.ndarray,
    sphere_right: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left_tree = cKDTree(sphere_left)
    right_tree = cKDTree(sphere_right)
    reflect = np.diag([-1.0, 1.0, 1.0])
    null = np.empty(n_permutations, dtype=np.float64)

    for idx in range(n_permutations):
        rotation = Rotation.random(random_state=rng).as_matrix()
        right_rotation = reflect @ rotation @ reflect

        spun_left_idx = _tree_query(left_tree, sphere_left @ rotation.T)
        spun_right_idx = _tree_query(right_tree, sphere_right @ right_rotation.T)

        spun_left = reference_left[spun_left_idx]
        spun_right = reference_right[spun_right_idx]
        n_reference = int(spun_left.sum()) + int(spun_right.sum())
        n_target = int(target_left.sum()) + int(target_right.sum())
        overlap = int(np.count_nonzero(spun_left & target_left))
        overlap += int(np.count_nonzero(spun_right & target_right))
        null[idx] = 2.0 * overlap / (n_reference + n_target)

    return null


def _quantiles(values: np.ndarray) -> dict[str, float]:
    probs = [0.0, 0.025, 0.05, 0.5, 0.95, 0.975, 1.0]
    names = ["min", "q025", "q05", "median", "q95", "q975", "max"]
    return {name: float(val) for name, val in zip(names, np.quantile(values, probs))}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute spin-test permutation p-value for the two-network overlay."
    )
    parser.add_argument("--map-1", type=Path, default=DEFAULT_MAP_1, help="Reference NIfTI map.")
    parser.add_argument("--map-2", type=Path, default=DEFAULT_MAP_2, help="Fixed target NIfTI map.")
    parser.add_argument("--n-permutations", type=int, default=1000, help="Number of spins.")
    parser.add_argument("--seed", type=int, default=20260428, help="Random seed.")
    parser.add_argument(
        "--interpolation",
        choices=["linear", "nearest", "nearest_most_frequent"],
        default="nearest",
        help="Volume-to-surface sampling method.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of cortical-depth samples for volume-to-surface projection.",
    )
    parser.add_argument(
        "--surface-threshold",
        type=float,
        default=0.0,
        help="Surface values greater than this are considered active.",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON, help="Summary JSON.")
    parser.add_argument("--null-csv", type=Path, default=DEFAULT_NULL_CSV, help="Null Dice CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_permutations <= 0:
        raise ValueError("--n-permutations must be positive.")

    map_1 = args.map_1.expanduser().resolve()
    map_2 = args.map_2.expanduser().resolve()
    ref_volume_mask, ref_img = _load_volume_mask(map_1)
    target_volume_mask, target_img = _load_volume_mask(map_2)
    if ref_img.shape[:3] != target_img.shape[:3]:
        raise ValueError(f"Map shapes differ: {ref_img.shape} vs {target_img.shape}")
    if not np.allclose(ref_img.affine, target_img.affine):
        raise ValueError("Map affines differ; resample before running this script.")

    print("Fetching/loading fsaverage6 surfaces...", flush=True)
    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage6")
    print("Projecting maps from MNI volume to fsaverage6 surface...", flush=True)
    ref_left = _surface_mask(
        map_1,
        fsaverage,
        "left",
        interpolation=args.interpolation,
        n_samples=args.n_samples,
        surface_threshold=args.surface_threshold,
    )
    ref_right = _surface_mask(
        map_1,
        fsaverage,
        "right",
        interpolation=args.interpolation,
        n_samples=args.n_samples,
        surface_threshold=args.surface_threshold,
    )
    target_left = _surface_mask(
        map_2,
        fsaverage,
        "left",
        interpolation=args.interpolation,
        n_samples=args.n_samples,
        surface_threshold=args.surface_threshold,
    )
    target_right = _surface_mask(
        map_2,
        fsaverage,
        "right",
        interpolation=args.interpolation,
        n_samples=args.n_samples,
        surface_threshold=args.surface_threshold,
    )

    sphere_left = _sphere_coords(fsaverage["sphere_left"])
    sphere_right = _sphere_coords(fsaverage["sphere_right"])
    observed_surface_dice = _dice(
        np.concatenate([ref_left, ref_right]),
        np.concatenate([target_left, target_right]),
    )

    print(f"Running {args.n_permutations} hemisphere-preserving spins...", flush=True)
    null_dice = _spin_null_dice(
        ref_left,
        ref_right,
        target_left,
        target_right,
        sphere_left,
        sphere_right,
        n_permutations=args.n_permutations,
        seed=args.seed,
    )

    n_greater = int(np.count_nonzero(null_dice > observed_surface_dice))
    n_greater_equal = int(np.count_nonzero(null_dice >= observed_surface_dice))
    p_greater_plus_one = (n_greater + 1) / (args.n_permutations + 1)
    p_greater_equal_plus_one = (n_greater_equal + 1) / (args.n_permutations + 1)
    result = {
        "method": "local spin-test permutation null following s41467-025-58176-9/NCT",
        "map_1_rotated_reference": str(map_1),
        "map_2_fixed_target": str(map_2),
        "volume_space": {
            "n_reference_voxels": int(ref_volume_mask.sum()),
            "n_target_voxels": int(target_volume_mask.sum()),
            "n_overlap_voxels": int(np.count_nonzero(ref_volume_mask & target_volume_mask)),
            "dice": _dice(ref_volume_mask, target_volume_mask),
        },
        "surface_projection": {
            "surface": "fsaverage6",
            "interpolation": args.interpolation,
            "n_depth_samples": int(args.n_samples),
            "surface_threshold": float(args.surface_threshold),
            "n_reference_vertices": int(ref_left.sum() + ref_right.sum()),
            "n_target_vertices": int(target_left.sum() + target_right.sum()),
            "n_overlap_vertices": int(
                np.count_nonzero(ref_left & target_left)
                + np.count_nonzero(ref_right & target_right)
            ),
            "observed_dice": float(observed_surface_dice),
        },
        "spin_test": {
            "n_permutations": int(args.n_permutations),
            "seed": int(args.seed),
            "n_null_dice_greater_than_observed": n_greater,
            "n_null_dice_greater_or_equal_observed": n_greater_equal,
            "p_value": float(p_greater_plus_one),
            "p_value_definition": "(n_null_dice_greater_than_observed + 1) / (n_permutations + 1)",
            "p_text_greater_no_correction": float(n_greater / args.n_permutations),
            "p_greater_or_equal": float(n_greater_equal / args.n_permutations),
            "p_greater_plus_one": float(p_greater_plus_one),
            "p_greater_equal_plus_one": float(p_greater_equal_plus_one),
            "null_dice_summary": {
                "mean": float(np.mean(null_dice)),
                "std": float(np.std(null_dice, ddof=1)),
                **_quantiles(null_dice),
            },
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.null_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    with args.null_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["permutation", "dice"])
        for idx, dice_value in enumerate(null_dice, start=1):
            writer.writerow([idx, f"{dice_value:.12g}"])

    print(f"Saved summary JSON: {args.out_json}", flush=True)
    print(f"Saved null Dice CSV: {args.null_csv}", flush=True)
    print(
        "NCT-style p = "
        f"{result['spin_test']['p_value']:.6g} "
        f"(({n_greater} + 1)/({args.n_permutations} + 1); "
        "null Dice values > observed)",
        flush=True,
    )


if __name__ == "__main__":
    main()
