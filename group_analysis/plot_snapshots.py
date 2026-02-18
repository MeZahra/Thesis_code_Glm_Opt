#!/usr/bin/env python3
"""Create axial-slice mosaics for each NIfTI in Results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import datasets, image, masking

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


def _strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def _is_nifti(path: Path) -> bool:
    return path.name.endswith(".nii.gz") or path.suffix == ".nii"


def _default_results_dir(script_dir: Path) -> Path:
    local_results = script_dir / "Results"
    if local_results.exists():
        return local_results
    repo_results = script_dir.parent / "results"
    if repo_results.exists():
        return repo_results
    return local_results


def _auto_background(results_dir: Path) -> Path | None:
    root = results_dir.parent
    candidates = [
        root / "mni_template_in_anat.nii.gz",
        root / "sub-pd009_ses-1_T1w_brain_anat_in_mni.nii.gz",
        Path("/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"),
        Path("/Data/zahra/anatomy_masks/MNI152_T1_1mm_brain.nii.gz"),
        Path("/usr/local/fsl/data/standard/MNI152_T1_2mm_brain.nii.gz"),
        Path("/usr/local/fsl/data/standard/MNI152_T1_1mm_brain.nii.gz"),
    ]
    candidates.extend(sorted(root.glob("*_T1w_brain_anat_in_mni.nii.gz")))
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_canonical(path: Path) -> nib.Nifti1Image:
    img = nib.load(str(path))
    return nib.as_closest_canonical(img)


def _load_brain_mask(mask_path: Path, ref_img: nib.Nifti1Image | None) -> np.ndarray:
    mask_img = _load_canonical(mask_path)
    if ref_img is not None and (
        mask_img.shape != ref_img.shape or not np.allclose(mask_img.affine, ref_img.affine)
    ):
        mask_img = image.resample_to_img(mask_img, ref_img, interpolation="nearest")
    mask_data = mask_img.get_fdata(dtype=np.float32)
    return mask_data > 0.5


def _compute_brain_mask(bg_img: nib.Nifti1Image, fallback_percentile: float) -> np.ndarray | None:
    try:
        mask_img = masking.compute_brain_mask(bg_img)
        return mask_img.get_fdata(dtype=np.float32) > 0
    except Exception as exc:  # pragma: no cover - best-effort fallback for unusual images
        print(f"Warning: auto brain-mask failed ({exc}); using percentile fallback.", flush=True)

    bg_data = np.nan_to_num(bg_img.get_fdata(dtype=np.float32))
    positive = bg_data[bg_data > 0]
    if positive.size == 0:
        return None
    threshold = np.percentile(positive, fallback_percentile)
    return bg_data > threshold


def _resample_overlay(overlay_img: nib.Nifti1Image, bg_img: nib.Nifti1Image) -> nib.Nifti1Image:
    if overlay_img.shape == bg_img.shape and np.allclose(overlay_img.affine, bg_img.affine):
        return overlay_img
    return image.resample_to_img(overlay_img, bg_img, interpolation="continuous")


def _pick_slices(n_z: int, n_slices: int, z_min: float, z_max: float) -> list[int]:
    if n_z <= 0:
        return []
    start = int(round(z_min * (n_z - 1)))
    end = int(round(z_max * (n_z - 1)))
    if end <= start:
        start, end = 0, n_z - 1
    return np.linspace(start, end, n_slices, dtype=int).tolist()


def _outline_mask_2d(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if mask.shape[0] < 3 or mask.shape[1] < 3:
        return mask
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _outline_edges(mask_3d: np.ndarray) -> np.ndarray:
    outline = np.zeros_like(mask_3d, dtype=bool)
    for z in range(mask_3d.shape[2]):
        outline[:, :, z] = _outline_mask_2d(mask_3d[:, :, z])
    return outline


def _normalize_labels(labels) -> list[str]:
    normalized: list[str] = []
    for label in labels:
        if isinstance(label, bytes):
            label = label.decode("utf-8", errors="replace")
        normalized.append(str(label))
    return normalized


def _as_canonical_img(img_or_path: nib.spatialimages.SpatialImage | str) -> nib.Nifti1Image:
    if isinstance(img_or_path, nib.spatialimages.SpatialImage):
        return nib.as_closest_canonical(img_or_path)
    return _load_canonical(Path(img_or_path))


def _resample_nearest(src_img: nib.Nifti1Image, ref_img: nib.Nifti1Image) -> nib.Nifti1Image:
    if src_img.shape == ref_img.shape and np.allclose(src_img.affine, ref_img.affine):
        return src_img
    return image.resample_to_img(src_img, ref_img, interpolation="nearest")


def _load_ho_atlas_context(ref_img: nib.Nifti1Image, data_dir: Path | None) -> dict | None:
    try:
        cort = datasets.fetch_atlas_harvard_oxford(
            "cort-maxprob-thr25-2mm", data_dir=str(data_dir) if data_dir else None, verbose=0
        )
        sub = datasets.fetch_atlas_harvard_oxford(
            "sub-maxprob-thr25-2mm", data_dir=str(data_dir) if data_dir else None, verbose=0
        )
    except Exception as exc:
        print(f"Warning: failed to fetch Harvard-Oxford atlases ({exc}).", flush=True)
        return None

    cort_img = _resample_nearest(_as_canonical_img(cort.maps), ref_img)
    sub_img = _resample_nearest(_as_canonical_img(sub.maps), ref_img)
    cort_data = np.rint(cort_img.get_fdata(dtype=np.float32)).astype(np.int32)
    sub_data = np.rint(sub_img.get_fdata(dtype=np.float32)).astype(np.int32)

    cort_labels = _normalize_labels(cort.labels)
    sub_labels = _normalize_labels(sub.labels)
    offset = max(0, len(cort_labels) - 1)
    combined = cort_data.copy()
    sub_mask = sub_data > 0
    combined[sub_mask] = sub_data[sub_mask] + offset
    combined_labels = list(cort_labels) + [f"sub:{label}" for label in sub_labels[1:]]

    return {
        "data": combined,
        "labels": combined_labels,
        "name": "Harvard-Oxford cortical+subcortical (thr25, 2mm)",
    }


def _atlas_key(img: nib.Nifti1Image) -> tuple:
    affine_key = tuple(np.round(img.affine, 6).ravel().tolist())
    return img.shape, affine_key


def _extract_active_voxel_rows(
    overlay_img: nib.Nifti1Image,
    overlay_data: np.ndarray,
    min_activation: float,
    atlas_context: dict | None,
    mni_only: bool = False,
) -> list[tuple]:
    active_coords = np.argwhere(overlay_data > min_activation)
    if active_coords.size == 0:
        return []

    flat_indices = np.ravel_multi_index(active_coords.T, overlay_data.shape)
    world_coords = nib.affines.apply_affine(overlay_img.affine, active_coords)
    values = overlay_data[tuple(active_coords.T)]

    atlas_values = None
    atlas_labels = None
    if atlas_context is not None:
        atlas_data = atlas_context["data"]
        if atlas_data.shape == overlay_data.shape:
            atlas_values = atlas_data[tuple(active_coords.T)]
            atlas_labels = atlas_context["labels"]

    rows: list[tuple] = []
    for idx, coord in enumerate(active_coords):
        wx, wy, wz = [float(v) for v in world_coords[idx]]
        if mni_only:
            rows.append((wx, wy, wz))
            continue

        atlas_index = int(atlas_values[idx]) if atlas_values is not None else -1
        atlas_label = "unknown"
        if atlas_values is not None and atlas_labels is not None:
            if 0 <= atlas_index < len(atlas_labels):
                atlas_label = atlas_labels[atlas_index]
        x, y, z = [int(v) for v in coord]
        rows.append(
            (
                x,
                y,
                z,
                int(flat_indices[idx]),
                wx,
                wy,
                wz,
                float(values[idx]),
                atlas_index,
                atlas_label,
            )
        )
    return rows


def _save_active_voxels_csv(csv_path: Path, rows: list[tuple], mni_only: bool = False) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if mni_only:
            writer.writerow(["mni_x", "mni_y", "mni_z"])
        else:
            writer.writerow(
                [
                    "x",
                    "y",
                    "z",
                    "flat_index",
                    "mni_x",
                    "mni_y",
                    "mni_z",
                    "activation",
                    "atlas_index",
                    "atlas_label",
                ]
            )
        writer.writerows(rows)


def _save_mosaic(
    bg_data: np.ndarray,
    overlay_data: np.ndarray,
    outline_mask: np.ndarray | None,
    z_indices: list[int],
    out_path: Path,
    n_cols: int,
    vmin: float,
    vmax: float,
    alpha: float,
    rotations: int,
    flip_vertical: bool,
    cmap: str,
    outline_alpha: float,
) -> None:
    n_slices = len(z_indices)
    n_cols = max(1, n_cols)
    n_rows = int(np.ceil(n_slices / n_cols))
    fig_w = n_cols * 1.4
    fig_h = n_rows * 1.4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), facecolor="black")
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])
    axes = axes.reshape(n_rows, n_cols)

    masked_overlay = np.ma.masked_less_equal(overlay_data, vmin)
    outline_cmap = ListedColormap([(0.0, 0.0, 0.0, 0.0), (1.0, 0.1, 0.1, 1.0)])
    for idx, z in enumerate(z_indices):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        ax.set_facecolor("black")
        bg_slice = bg_data[:, :, z]
        overlay_slice = masked_overlay[:, :, z]
        outline_slice = outline_mask[:, :, z] if outline_mask is not None else None
        if rotations:
            bg_slice = np.rot90(bg_slice, k=rotations)
            overlay_slice = np.rot90(overlay_slice, k=rotations)
            if outline_slice is not None:
                outline_slice = np.rot90(outline_slice, k=rotations)
        if flip_vertical:
            bg_slice = np.flipud(bg_slice)
            overlay_slice = np.flipud(overlay_slice)
            if outline_slice is not None:
                outline_slice = np.flipud(outline_slice)
        ax.imshow(bg_slice, cmap="gray", origin="lower", interpolation="nearest")
        ax.imshow(
            overlay_slice,
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )
        if outline_slice is not None and np.any(outline_slice):
            ax.imshow(
                outline_slice.astype(np.uint8),
                cmap=outline_cmap,
                origin="lower",
                interpolation="nearest",
                vmin=0,
                vmax=1,
                alpha=outline_alpha,
            )
        ax.axis("off")

    for idx in range(n_slices, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")
        axes[r, c].set_facecolor("black")

    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_results = _default_results_dir(script_dir)
    parser = argparse.ArgumentParser(description="Create axial-slice mosaics for each NIfTI in Results.")
    parser.add_argument("--results-dir", type=Path, default=default_results, help="Directory with NIfTI results.")
    parser.add_argument("--nii-file", type=Path, default=None, help="Single NIfTI file to plot.")
    parser.add_argument("--bg", type=Path, default=None, help="Background anatomical NIfTI.")
    parser.add_argument("--brain-mask", type=Path, default=None, help="Brain mask NIfTI to clean background.")
    parser.add_argument(
        "--no-brain-mask",
        action="store_true",
        help="Disable brain-masking of the background/overlay.",
    )
    parser.add_argument(
        "--mask-percentile",
        type=float,
        default=5.0,
        help="Fallback percentile for auto mask if brain-mask inference fails.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for mosaics.")
    parser.add_argument("--n-slices", type=int, default=48, help="Number of slices to show per mosaic.")
    parser.add_argument("--n-cols", type=int, default=8, help="Number of columns in the mosaic grid.")
    parser.add_argument("--z-min", type=float, default=0.1, help="Min Z percentile (0-1) for slices.")
    parser.add_argument("--z-max", type=float, default=0.9, help="Max Z percentile (0-1) for slices.")
    parser.add_argument("--min-activation", type=float, default=0.0, help="Minimum activation to display.")
    parser.add_argument("--vmax-percentile", type=float, default=99.0, help="Percentile for vmax scaling.")
    parser.add_argument("--alpha", type=float, default=0.9, help="Overlay alpha.")
    parser.add_argument("--cmap", type=str, default="Reds", help="Matplotlib colormap for activations.")
    parser.add_argument(
        "--outline-percentile",
        type=float,
        default=99.0,
        help="Percentile for activation outline (top 1-2%%).",
    )
    parser.add_argument("--outline-alpha", type=float, default=1.0, help="Outline alpha.")
    parser.add_argument("--rot90", type=int, default=1, help="Number of 90-degree rotations per slice.")
    parser.add_argument("--flip-vertical", action="store_true", help="Flip each slice vertically.")
    parser.add_argument("--abs", dest="use_abs", action="store_true", help="Use absolute activation values.")
    parser.add_argument(
        "--save-active-csv",
        action="store_true",
        help="Save active-voxel CSV for each NIfTI (indices, MNI coordinates, atlas labels).",
    )
    parser.add_argument(
        "--active-csv-dir",
        type=Path,
        default=None,
        help="Output directory for active-voxel CSV files (default: <out-dir>/active_voxels).",
    )
    parser.add_argument(
        "--no-atlas-labels",
        action="store_true",
        help="Disable atlas-based naming in active-voxel CSV files.",
    )
    parser.add_argument(
        "--atlas-data-dir",
        type=Path,
        default=None,
        help="Directory for nilearn atlas cache/downloads.",
    )
    parser.add_argument(
        "--mni-only-csv",
        action="store_true",
        help="Save only MNI coordinates (mni_x,mni_y,mni_z) in active-voxel CSV files.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip output files that already exist.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.nii_file is not None:
        nii_file = args.nii_file
        if not nii_file.exists() or not nii_file.is_file():
            raise FileNotFoundError(f"NIfTI file not found: {nii_file}")
        if not _is_nifti(nii_file):
            raise ValueError(f"Not a NIfTI file: {nii_file}")
        results_dir = nii_file.parent
        nifti_files = [nii_file]
    else:
        results_dir = args.results_dir
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")
        nifti_files = sorted([p for p in results_dir.iterdir() if p.is_file() and _is_nifti(p)])
        if not nifti_files:
            print(f"No NIfTI files found in {results_dir}", flush=True)
            return

    bg_path = args.bg or _auto_background(results_dir)
    bg_img = _load_canonical(bg_path) if bg_path else None
    brain_mask = None
    if bg_img is not None and not args.no_brain_mask:
        if args.brain_mask:
            brain_mask = _load_brain_mask(args.brain_mask, bg_img)
            print(f"Using brain mask: {args.brain_mask}", flush=True)
        else:
            brain_mask = _compute_brain_mask(bg_img, args.mask_percentile)
            if brain_mask is not None:
                print("Computed brain mask from background.", flush=True)
            else:
                print("Warning: could not derive a brain mask from background.", flush=True)
    if bg_path:
        print(f"Using background: {bg_path}", flush=True)
    else:
        print("No background provided; using black background.", flush=True)

    out_dir = args.out_dir or (results_dir / "snapshots")
    out_dir.mkdir(parents=True, exist_ok=True)
    active_csv_dir = None
    if args.save_active_csv:
        active_csv_dir = args.active_csv_dir or (out_dir / "active_voxels")
        active_csv_dir.mkdir(parents=True, exist_ok=True)

    atlas_cache: dict[tuple, dict | None] = {}
    atlas_announced = False

    for nii_path in nifti_files:
        base = _strip_nii_suffix(nii_path)
        out_path = out_dir / f"{base}_mosaic.png"
        if args.skip_existing and out_path.exists():
            continue

        overlay_img = _load_canonical(nii_path)
        if bg_img is not None:
            overlay_img = _resample_overlay(overlay_img, bg_img)
            bg_data = np.nan_to_num(bg_img.get_fdata(dtype=np.float32))
            if brain_mask is not None:
                bg_data = bg_data.copy()
                bg_data[~brain_mask] = 0
        else:
            bg_data = np.zeros(overlay_img.shape, dtype=np.float32)

        overlay_data = np.nan_to_num(overlay_img.get_fdata(dtype=np.float32))
        if brain_mask is not None and overlay_data.shape == brain_mask.shape:
            overlay_data = overlay_data.copy()
            overlay_data[~brain_mask] = 0
        if args.use_abs:
            overlay_data = np.abs(overlay_data)

        active = overlay_data[overlay_data > args.min_activation]
        if active.size == 0 and not args.use_abs:
            negative_active = overlay_data[overlay_data < -args.min_activation]
            if negative_active.size:
                overlay_data = np.abs(overlay_data)
                active = overlay_data[overlay_data > args.min_activation]
                print(f"{nii_path.name}: no positive activations; using absolute values.", flush=True)
        if active.size:
            vmax = np.percentile(active, args.vmax_percentile)
            if vmax <= args.min_activation:
                vmax = active.max()
        else:
            vmax = args.min_activation + 1.0

        outline_mask = None
        if active.size:
            outline_thr = np.percentile(active, args.outline_percentile)
            outline_mask = overlay_data >= outline_thr
            outline_mask = _outline_edges(outline_mask)

        if args.save_active_csv and active_csv_dir is not None:
            atlas_context = None
            if not args.no_atlas_labels and not args.mni_only_csv:
                key = _atlas_key(overlay_img)
                if key not in atlas_cache:
                    atlas_cache[key] = _load_ho_atlas_context(overlay_img, args.atlas_data_dir)
                atlas_context = atlas_cache[key]
                if atlas_context is not None and not atlas_announced:
                    print(f"Using atlas labels: {atlas_context['name']}", flush=True)
                    atlas_announced = True
            rows = _extract_active_voxel_rows(
                overlay_img=overlay_img,
                overlay_data=overlay_data,
                min_activation=args.min_activation,
                atlas_context=atlas_context,
                mni_only=args.mni_only_csv,
            )
            csv_path = active_csv_dir / f"{base}_active_voxels.csv"
            _save_active_voxels_csv(csv_path, rows, mni_only=args.mni_only_csv)
            print(f"Saved: {csv_path} ({len(rows)} active voxels)", flush=True)

        z_indices = _pick_slices(overlay_data.shape[2], args.n_slices, args.z_min, args.z_max)
        _save_mosaic(
            bg_data=bg_data,
            overlay_data=overlay_data,
            outline_mask=outline_mask,
            z_indices=z_indices,
            out_path=out_path,
            n_cols=args.n_cols,
            vmin=args.min_activation,
            vmax=vmax,
            alpha=args.alpha,
            rotations=args.rot90,
            flip_vertical=args.flip_vertical,
            cmap=args.cmap,
            outline_alpha=args.outline_alpha,
        )
        print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
