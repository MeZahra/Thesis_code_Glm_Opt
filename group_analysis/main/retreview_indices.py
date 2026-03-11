import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.lib.format import open_memmap


DEFAULT_WEIGHTS_PATH = "results/connectivity/voxel_weights_absmean_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.6_gamma1.nii.gz"
DEFAULT_COORDS_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/active_coords_group.npy"
DEFAULT_FLAT_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/active_flat_indices__group.npy"
DEFAULT_BETA_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/beta_volume_filter_group.npy"
DEFAULT_OUTPUT_DIR = "results/connectivity"
DEFAULT_MANIFEST_PATH = "results/connectivity/concat_manifest_group.tsv"
FALLBACK_MANIFEST_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/concat_manifest_group.tsv"
GVS_LABELS = ["sham", "GVS1", "GVS2", "GVS3", "GVS4", "GVS5", "GVS6", "GVS7", "GVS8"]


def _normalize_coords(coords):
    coords = np.asarray(coords)
    if coords.dtype == object and coords.size == 3:
        coords = np.vstack([np.asarray(a, dtype=np.int64).ravel() for a in coords])
    elif coords.ndim == 2 and coords.shape[1] == 3:
        coords = coords.T
    if coords.ndim != 2 or coords.shape[0] != 3:
        raise ValueError(f"Expected coords to have shape (3, n_vox), got {coords.shape}")
    return coords.astype(np.int64, copy=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Retrieve selected voxel indices from a weight map and extract beta values across trials."
    )
    parser.add_argument("--weights-path", default=DEFAULT_WEIGHTS_PATH, help="Path to voxel-weights NIfTI.")
    parser.add_argument("--coords-path", default=DEFAULT_COORDS_PATH, help="Path to active_coords_group.npy.")
    parser.add_argument("--flat-path", default=DEFAULT_FLAT_PATH, help="Path to active_flat_indices__group.npy.")
    parser.add_argument("--beta-path", default=DEFAULT_BETA_PATH, help="Path to beta_volume_filter_group.npy.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save outputs.")
    parser.add_argument(
        "--manifest-path",
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest TSV with trial offsets and trial_keep paths.",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=0.0,
        help="Select voxels with finite weight and weight > threshold.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Row chunk size used when copying selected beta values to output.",
    )
    parser.add_argument(
        "--no-gvs-split",
        action="store_true",
        help="Skip splitting selected beta matrix into sham/GVS condition matrices.",
    )
    return parser.parse_args()


def _resolve_manifest_path(manifest_path):
    path = Path(manifest_path)
    if path.exists():
        return path
    fallback = Path(FALLBACK_MANIFEST_PATH)
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"Manifest not found at {path} or fallback {fallback}."
    )


def _build_condition_column_indices(manifest_path, total_output_trials):
    condition_columns = [[] for _ in range(len(GVS_LABELS))]
    coverage = np.zeros(total_output_trials, dtype=np.int16)

    with open(manifest_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    for row in rows:
        offset_start = int(row["offset_start"])
        offset_end = int(row["offset_end"])
        if offset_start < 0 or offset_end > total_output_trials or offset_end < offset_start:
            raise ValueError(
                f"Invalid offsets in manifest row: start={offset_start}, end={offset_end}, total={total_output_trials}"
            )

        run_cols = np.arange(offset_start, offset_end, dtype=np.int64)
        n_run_output = run_cols.size
        if n_run_output == 0:
            continue

        trial_keep_path = str(row.get("trial_keep_path", "")).strip()
        if not trial_keep_path:
            raise ValueError(
                f"Missing trial_keep_path for {row.get('sub_tag')} ses-{row.get('ses')} run-{row.get('run')}"
            )

        trial_keep = np.asarray(np.load(trial_keep_path), dtype=bool).ravel()
        kept_local = np.flatnonzero(trial_keep).astype(np.int64, copy=False)

        if n_run_output == kept_local.size:
            source_local_trials = kept_local
        elif n_run_output == trial_keep.size:
            source_local_trials = np.arange(trial_keep.size, dtype=np.int64)
        else:
            raise ValueError(
                f"Run trial mismatch for {row.get('sub_tag')} ses-{row.get('ses')} run-{row.get('run')}: "
                f"manifest output={n_run_output}, kept={kept_local.size}, source={trial_keep.size}"
            )

        cond_ids = source_local_trials % len(GVS_LABELS)
        for cond_id in range(len(GVS_LABELS)):
            cols = run_cols[cond_ids == cond_id]
            if cols.size:
                condition_columns[cond_id].extend(cols.tolist())

        coverage[run_cols] += 1

    if np.any(coverage != 1):
        bad = np.flatnonzero(coverage != 1)
        raise ValueError(
            f"Manifest/keep mapping does not cover each output trial exactly once. "
            f"Columns with invalid coverage: {bad.size}"
        )

    return [np.asarray(cols, dtype=np.int64) for cols in condition_columns]


def _split_selected_beta_by_gvs(selected_beta_path, output_dir, condition_columns, chunk_size):
    selected_beta = np.load(selected_beta_path, mmap_mode="r")
    n_voxels, n_trials = selected_beta.shape
    output_dir = Path(output_dir)

    for cond_id, label in enumerate(GVS_LABELS):
        cols = condition_columns[cond_id]
        if cols.size and int(cols.max()) >= n_trials:
            raise ValueError(
                f"Condition {label} includes trial index {int(cols.max())} outside selected_beta range {n_trials}."
            )
        out_path = output_dir / f"selected_beta_trials_{label}.npy"
        out_mm = open_memmap(out_path, mode="w+", dtype=selected_beta.dtype, shape=(n_voxels, cols.size))
        row_chunk = max(1, int(chunk_size))
        for start in range(0, n_voxels, row_chunk):
            end = min(start + row_chunk, n_voxels)
            out_mm[start:end, :] = selected_beta[start:end, :][:, cols]
        out_mm.flush()
        del out_mm
        print(f"Saved {label}: {out_path} | shape=({n_voxels}, {cols.size})")

    index_npz_path = output_dir / "selected_beta_trials_gvs_column_indices.npz"
    np.savez(index_npz_path, **{label: condition_columns[i] for i, label in enumerate(GVS_LABELS)})
    print(f"Saved GVS column-index map: {index_npz_path}")


def main():
    args = parse_args()

    weights_path = args.weights_path
    coords_path = args.coords_path
    flat_path = args.flat_path
    beta_path = args.beta_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _resolve_manifest_path(args.manifest_path)

    weights_img = nib.load(weights_path)
    weights_data = weights_img.get_fdata()
    volume_shape = weights_data.shape[:3]

    coords = _normalize_coords(np.load(coords_path, allow_pickle=True))
    active_flat = np.asarray(np.load(flat_path, mmap_mode="r"), dtype=np.int64).ravel()

    if coords.shape[1] != active_flat.size:
        raise ValueError(
            f"active_coords voxel count ({coords.shape[1]}) != active_flat voxel count ({active_flat.size})"
        )

    if np.any(coords < 0):
        raise ValueError("active_coords contains negative indices.")
    for dim, size in enumerate(volume_shape):
        if np.any(coords[dim] >= size):
            raise ValueError(
                f"active_coords out of bounds for axis {dim}: max={coords[dim].max()}, size={size}"
            )

    flat_from_coords = np.ravel_multi_index(tuple(coords), dims=volume_shape)
    if not np.array_equal(flat_from_coords, active_flat):
        raise ValueError("active_coords and active_flat indices are not aligned.")

    active_weights = weights_data[tuple(coords)]
    selected_mask = np.isfinite(active_weights) & (active_weights > float(args.weight_threshold))
    sel_active_idx = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    if sel_active_idx.size == 0:
        raise ValueError(
            f"No selected voxels found with finite weight > {args.weight_threshold}."
        )

    sel_flat_idx = active_flat[sel_active_idx]
    sel_ijk = np.column_stack(np.unravel_index(sel_flat_idx, volume_shape)).astype(np.int32, copy=False)
    sel_weights = np.asarray(active_weights[sel_active_idx], dtype=np.float32)

    beta_data = np.load(beta_path, mmap_mode="r")
    if beta_data.ndim != 2:
        raise ValueError(f"Expected beta matrix shape (n_voxels, n_trials), got {beta_data.shape}")
    if beta_data.shape[0] != active_flat.size:
        raise ValueError(
            f"beta rows ({beta_data.shape[0]}) != active voxel count ({active_flat.size})"
        )

    beta_out_path = output_dir / f"selected_beta_trials.npy"
    meta_out_path = output_dir / f"selected_voxel_indices.npz"

    selected_beta_mm = open_memmap(beta_out_path, mode="w+", dtype=beta_data.dtype, shape=(sel_active_idx.size, beta_data.shape[1]))
    chunk_size = max(1, int(args.chunk_size))
    for start in range(0, sel_active_idx.size, chunk_size):
        end = min(start + chunk_size, sel_active_idx.size)
        idx = sel_active_idx[start:end]
        selected_beta_mm[start:end, :] = beta_data[idx, :]

    selected_beta_mm.flush()
    del selected_beta_mm

    np.savez(
        meta_out_path,
        selected_active_indices=sel_active_idx,
        selected_flat_indices=sel_flat_idx,
        selected_ijk=sel_ijk,
        selected_weights=sel_weights,
        weight_threshold=np.float32(args.weight_threshold),
    )

    print(f"Selected voxels: {sel_active_idx.size}")
    print(f"Trials: {beta_data.shape[1]}")
    print(f"Saved beta matrix: {beta_out_path}")
    print(f"Saved metadata: {meta_out_path}")

    if not args.no_gvs_split:
        condition_columns = _build_condition_column_indices(manifest_path, int(beta_data.shape[1]))
        _split_selected_beta_by_gvs(beta_out_path, output_dir, condition_columns, args.chunk_size)


if __name__ == "__main__":
    main()
