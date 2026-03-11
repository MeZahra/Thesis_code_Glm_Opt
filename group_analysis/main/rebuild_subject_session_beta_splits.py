#!/usr/bin/env python3
"""
Rebuild subject/session beta files using a coordinate CSV as the voxel selector.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild selected_beta_trials_sub-*_ses-*.npy files from a global "
            "selected beta matrix using voxel coordinates from a CSV file."
        )
    )
    parser.add_argument(
        "--input-beta",
        type=Path,
        required=True,
        help="Global beta matrix with shape (n_selected_voxels, n_trials).",
    )
    parser.add_argument(
        "--input-voxel-indices",
        type=Path,
        required=True,
        help="NPZ describing the rows of --input-beta.",
    )
    parser.add_argument(
        "--coords-csv",
        type=Path,
        required=True,
        help="CSV containing x,y,z voxel coordinates to keep.",
    )
    parser.add_argument(
        "--column-indices",
        type=Path,
        required=True,
        help="NPZ mapping subject/session labels to trial-column indices.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where selected_beta_trials_sub-*_ses-*.npy will be written.",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Optional voxel-weight image used to populate selected_weights in output metadata.",
    )
    parser.add_argument(
        "--output-voxel-indices",
        type=Path,
        default=None,
        help="Output NPZ path for the subset voxel metadata. Defaults to output-dir/selected_voxel_indices.npz.",
    )
    return parser.parse_args()


def _load_coords_csv(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        required = {"x", "y", "z"}
        if not required.issubset(fieldnames):
            raise ValueError(f"{path} must contain x,y,z columns; found {fieldnames}")
        rows = [(int(row["x"]), int(row["y"]), int(row["z"])) for row in reader]
    if not rows:
        raise ValueError(f"{path} is empty.")
    coords = np.asarray(rows, dtype=np.int32)
    unique_coords = np.unique(coords, axis=0)
    if unique_coords.shape[0] != coords.shape[0]:
        raise ValueError(f"{path} contains duplicate coordinates.")
    return coords


def _load_selected_ijk(voxel_pack: np.lib.npyio.NpzFile) -> np.ndarray:
    if "selected_ijk" in voxel_pack.files:
        ijk = np.asarray(voxel_pack["selected_ijk"], dtype=np.int32)
    else:
        raise KeyError("input voxel indices must contain selected_ijk.")
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected selected_ijk shape (N, 3), got {ijk.shape}")
    return ijk


def _map_coords_to_rows(coords: np.ndarray, selected_ijk: np.ndarray) -> np.ndarray:
    lookup = {tuple(row): idx for idx, row in enumerate(selected_ijk.tolist())}
    row_indices = np.empty(coords.shape[0], dtype=np.int64)
    missing: list[tuple[int, int, int]] = []
    for idx, row in enumerate(coords.tolist()):
        match = lookup.get(tuple(row))
        if match is None:
            missing.append(tuple(row))
        else:
            row_indices[idx] = match
    if missing:
        preview = ", ".join(str(item) for item in missing[:10])
        raise KeyError(f"{len(missing)} coordinates were not found in input voxel indices. First missing: {preview}")
    return row_indices


def _extract_weights(coords: np.ndarray, weights_path: Path | None) -> np.ndarray:
    if weights_path is None:
        return np.full(coords.shape[0], np.nan, dtype=np.float32)
    weight_img = nib.load(str(weights_path))
    weight_data = weight_img.get_fdata(dtype=np.float32)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    if (
        np.any(x < 0)
        or np.any(y < 0)
        or np.any(z < 0)
        or np.any(x >= weight_data.shape[0])
        or np.any(y >= weight_data.shape[1])
        or np.any(z >= weight_data.shape[2])
    ):
        raise ValueError(f"At least one coordinate lies outside weight image bounds {weight_data.shape[:3]}.")
    return np.asarray(weight_data[x, y, z], dtype=np.float32)


def _write_subset_voxel_indices(
    output_path: Path,
    input_pack: np.lib.npyio.NpzFile,
    row_indices: np.ndarray,
    coords: np.ndarray,
    selected_weights: np.ndarray,
) -> None:
    subset = {}
    if "selected_active_indices" in input_pack.files:
        subset["selected_active_indices"] = np.asarray(input_pack["selected_active_indices"], dtype=np.int64)[row_indices]
    if "selected_flat_indices" in input_pack.files:
        subset["selected_flat_indices"] = np.asarray(input_pack["selected_flat_indices"], dtype=np.int64)[row_indices]
    subset["selected_ijk"] = np.asarray(coords, dtype=np.int32)
    subset["selected_weights"] = np.asarray(selected_weights, dtype=np.float32)
    subset["weight_threshold"] = np.float32(0.0)
    np.savez(output_path, **subset)


def _write_subject_session_beta_files(
    input_beta: np.ndarray,
    row_indices: np.ndarray,
    column_pack: np.lib.npyio.NpzFile,
    output_dir: Path,
) -> list[tuple[str, tuple[int, int]]]:
    written: list[tuple[str, tuple[int, int]]] = []
    for label in column_pack.files:
        cols = np.asarray(column_pack[label], dtype=np.int64)
        if cols.ndim != 1:
            raise ValueError(f"Column indices for {label} must be 1D, got {cols.shape}")
        if cols.size and (int(cols.min()) < 0 or int(cols.max()) >= input_beta.shape[1]):
            raise ValueError(
                f"Column indices for {label} fall outside input beta trial range {input_beta.shape[1]}."
            )
        out_path = output_dir / f"selected_beta_trials_{label}.npy"
        subset = np.asarray(input_beta[row_indices, :][:, cols], dtype=np.float32)
        np.save(out_path, subset)
        written.append((out_path.name, subset.shape))
    return written


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_voxel_indices = args.output_voxel_indices or (output_dir / "selected_voxel_indices.npz")

    input_beta = np.load(args.input_beta, mmap_mode="r")
    if input_beta.ndim != 2:
        raise ValueError(f"Expected 2D input beta matrix, got {input_beta.shape}")

    input_pack = np.load(args.input_voxel_indices, allow_pickle=True)
    selected_ijk = _load_selected_ijk(input_pack)
    if selected_ijk.shape[0] != input_beta.shape[0]:
        raise ValueError(
            f"Input voxel metadata rows ({selected_ijk.shape[0]}) do not match input beta rows ({input_beta.shape[0]})."
        )

    coords = _load_coords_csv(args.coords_csv)
    row_indices = _map_coords_to_rows(coords, selected_ijk)
    if np.unique(row_indices).size != row_indices.size:
        raise ValueError("Coordinate-to-row mapping produced duplicate beta rows.")

    selected_weights = _extract_weights(coords, args.weights_path)
    _write_subset_voxel_indices(output_voxel_indices, input_pack, row_indices, coords, selected_weights)

    column_pack = np.load(args.column_indices, allow_pickle=True)
    written = _write_subject_session_beta_files(input_beta, row_indices, column_pack, output_dir)

    print(f"Selected voxel rows: {row_indices.size}")
    print(f"Saved voxel metadata: {output_voxel_indices}")
    print(f"Saved subject/session beta files: {len(written)}")
    for name, shape in written:
        print(f"  {name}: {shape}")


if __name__ == "__main__":
    main()
