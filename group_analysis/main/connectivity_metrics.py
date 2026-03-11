#!/usr/bin/env python3
"""
Create a fitted ROI atlas for connectivity beta datasets in MNI152 2mm space.

Base ROIs (requested):
1. Amygdala
2. Cerebellum
3. Cingulate Cortex
4. Inferior Frontal Gyrus
5. Dorsolateral Prefrontal Cortex
6. vmPFC / dmPFC (Control & monitoring)
7. Parietal Cortex
8. Precentral
9. Temporal Cortex
10. Thalamus

If these do not fully cover active voxels, complementary relative ROIs are added.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, image, plotting

FSL_CEREBELLUM_MAXPROB_2MM = Path(
    "/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr25-2mm.nii.gz"
)


@dataclass
class ROIGroup:
    name: str
    source: str
    mask: np.ndarray
    matched_labels: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit requested ROI atlas to selected_beta_trials*.npy and add relative ROIs "
            "for uncovered active voxels."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/connectivity/data"),
        help="Directory containing selected_beta_trials*.npy and selected_voxel_indices.npz.",
    )
    parser.add_argument(
        "--anat-path",
        type=Path,
        default=Path("results/connectivity/data/MNI152_T1_2mm_brain.nii.gz"),
        help="Anatomy image in MNI space.",
    )
    parser.add_argument(
        "--voxel-indices-path",
        type=Path,
        default=Path("results/connectivity/data/selected_voxel_indices.npz"),
        help="NPZ with selected_ijk or selected_flat_indices.",
    )
    parser.add_argument(
        "--beta-pattern",
        default="selected_beta_trials*.npy",
        help="Glob for beta files used to compute pooled mean|beta| voxel scores.",
    )
    parser.add_argument(
        "--active-beta-file",
        default="selected_beta_trials_GVS1.npy",
        help=(
            "File used to define active voxels (non-zero finite along trials). "
            "Relative path is resolved inside --data-dir."
        ),
    )
    parser.add_argument(
        "--max-beta-files",
        type=int,
        default=0,
        help="Optional cap on number of beta files to use (0 means all).",
    )
    parser.add_argument(
        "--chunk-trials",
        type=int,
        default=256,
        help="Chunk size along trials when computing mean|beta|.",
    )
    parser.add_argument(
        "--no-relative-rois",
        dest="add_relative_rois",
        action="store_false",
        help="Disable complementary relative ROIs.",
    )
    parser.set_defaults(add_relative_rois=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/connectivity"),
        help="Output directory.",
    )
    return parser.parse_args()


def _to_label_list(labels) -> list[str]:
    out: list[str] = []
    for value in list(labels):
        if isinstance(value, bytes):
            out.append(value.decode("utf-8", errors="replace"))
        else:
            out.append(str(value))
    return out


def _load_selected_ijk(voxel_indices_path: Path, anat_shape: tuple[int, int, int]) -> np.ndarray:
    pack = np.load(voxel_indices_path, allow_pickle=True)
    if "selected_ijk" in pack.files:
        ijk = np.asarray(pack["selected_ijk"], dtype=np.int32)
    elif "selected_flat_indices" in pack.files:
        flat = np.asarray(pack["selected_flat_indices"], dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, anat_shape)).astype(np.int32, copy=False)
    else:
        raise KeyError(
            "selected_voxel_indices.npz must contain 'selected_ijk' or 'selected_flat_indices'."
        )

    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected selected indices shape (N, 3); got {ijk.shape}")

    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < anat_shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < anat_shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < anat_shape[2])
    )
    if not np.all(valid):
        dropped = int(np.count_nonzero(~valid))
        print(f"Warning: dropped {dropped} out-of-bounds selected voxels.", flush=True)
        ijk = ijk[valid]
    return ijk


def _mean_abs_per_voxel(beta_2d: np.ndarray, chunk_trials: int) -> np.ndarray:
    n_voxels, n_trials = beta_2d.shape
    sums = np.zeros(n_voxels, dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.float64)
    for start in range(0, n_trials, chunk_trials):
        stop = min(start + chunk_trials, n_trials)
        chunk = np.asarray(beta_2d[:, start:stop], dtype=np.float32)
        sums += np.nansum(np.abs(chunk), axis=1)
        counts += np.sum(np.isfinite(chunk), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return out.astype(np.float32, copy=False)


def _load_and_pool_beta_scores(
    beta_files: list[Path],
    n_voxels: int,
    chunk_trials: int,
) -> tuple[np.ndarray, list[str]]:
    pooled = np.zeros(n_voxels, dtype=np.float64)
    used_files: list[str] = []
    for beta_path in beta_files:
        beta = np.load(beta_path, mmap_mode="r")
        if beta.ndim != 2:
            print(f"Skipping {beta_path.name}: expected 2D, got {beta.shape}", flush=True)
            continue
        if beta.shape[0] != n_voxels:
            print(
                f"Skipping {beta_path.name}: voxel count {beta.shape[0]} != {n_voxels}",
                flush=True,
            )
            continue
        pooled += _mean_abs_per_voxel(beta, chunk_trials)
        used_files.append(beta_path.name)
        print(f"Loaded beta file: {beta_path.name} {beta.shape}", flush=True)
    if not used_files:
        raise RuntimeError("No valid beta files loaded.")
    pooled /= float(len(used_files))
    return pooled.astype(np.float32, copy=False), used_files


def _load_active_mask(active_beta_file: Path, n_voxels: int) -> np.ndarray:
    beta = np.load(active_beta_file, mmap_mode="r")
    if beta.ndim != 2:
        raise ValueError(f"Active-beta file must be 2D; got {beta.shape}")
    if beta.shape[0] != n_voxels:
        raise ValueError(
            f"Active-beta voxel count {beta.shape[0]} does not match selected voxels {n_voxels}."
        )
    active = np.any(np.isfinite(beta) & (beta != 0), axis=1)
    return active


def _resample_label_img(label_img: nib.Nifti1Image, anat_img: nib.Nifti1Image) -> np.ndarray:
    resampled = image.resample_to_img(
        label_img,
        anat_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )
    return resampled.get_fdata().astype(np.int32)


def _select_ids_by_patterns(
    labels: list[str],
    include: list[str],
    exclude: list[str] | None = None,
) -> tuple[list[int], list[str]]:
    include_lower = [item.lower() for item in include]
    exclude_lower = [item.lower() for item in (exclude or [])]
    ids: list[int] = []
    names: list[str] = []
    for idx, label in enumerate(labels):
        if idx == 0:
            continue
        lname = label.lower()
        if not any(pattern in lname for pattern in include_lower):
            continue
        if any(pattern in lname for pattern in exclude_lower):
            continue
        ids.append(idx)
        names.append(label)
    return ids, names


def _fetch_ho_cort_sub(cache_dir: Path) -> tuple[nib.Nifti1Image, list[str], nib.Nifti1Image, list[str]]:
    cortical = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm",
        data_dir=str(cache_dir),
    )
    subcortical = datasets.fetch_atlas_harvard_oxford(
        "sub-maxprob-thr25-2mm",
        data_dir=str(cache_dir),
    )
    cortical_img = cortical.maps if isinstance(cortical.maps, nib.Nifti1Image) else nib.load(cortical.maps)
    subcortical_img = (
        subcortical.maps if isinstance(subcortical.maps, nib.Nifti1Image) else nib.load(subcortical.maps)
    )
    return cortical_img, _to_label_list(cortical.labels), subcortical_img, _to_label_list(subcortical.labels)


def _load_cerebellum_label_img() -> nib.Nifti1Image:
    if not FSL_CEREBELLUM_MAXPROB_2MM.exists():
        raise FileNotFoundError(
            f"Cerebellum atlas not found: {FSL_CEREBELLUM_MAXPROB_2MM}"
        )
    return nib.load(str(FSL_CEREBELLUM_MAXPROB_2MM))


def _make_mask_from_labels(label_data: np.ndarray, label_ids: list[int], shape: tuple[int, int, int]) -> np.ndarray:
    if not label_ids:
        return np.zeros(shape, dtype=bool)
    return np.isin(label_data, label_ids)


def _build_base_requested_rois(
    anat_img: nib.Nifti1Image,
    cache_dir: Path,
) -> tuple[list[ROIGroup], dict, np.ndarray, list[str], np.ndarray, list[str], np.ndarray]:
    cort_img, cort_labels, sub_img, sub_labels = _fetch_ho_cort_sub(cache_dir)
    cereb_img = _load_cerebellum_label_img()

    cort_data = _resample_label_img(cort_img, anat_img)
    sub_data = _resample_label_img(sub_img, anat_img)
    cereb_data = _resample_label_img(cereb_img, anat_img)
    shape = anat_img.shape[:3]

    roi_groups: list[ROIGroup] = []

    def add_from_sub(name: str, patterns: list[str]):
        ids, names = _select_ids_by_patterns(sub_labels, include=patterns)
        roi_groups.append(
            ROIGroup(
                name=name,
                source="Harvard-Oxford Subcortical (thr25, 2mm)",
                mask=_make_mask_from_labels(sub_data, ids, shape),
                matched_labels=names,
            )
        )

    def add_from_cort(name: str, patterns: list[str], exclude: list[str] | None = None):
        ids, names = _select_ids_by_patterns(cort_labels, include=patterns, exclude=exclude)
        roi_groups.append(
            ROIGroup(
                name=name,
                source="Harvard-Oxford Cortical (thr25, 2mm)",
                mask=_make_mask_from_labels(cort_data, ids, shape),
                matched_labels=names,
            )
        )

    add_from_sub("Amygdala", ["amygdala"])
    roi_groups.append(
        ROIGroup(
            name="Cerebellum",
            source="FSL Cerebellum MNIfnirt (maxprob thr25, 2mm)",
            mask=cereb_data > 0,
            matched_labels=["All cerebellar labels (value > 0)"],
        )
    )
    add_from_cort("Cingulate Cortex", ["cingulate gyrus", "paracingulate gyrus"])
    add_from_cort("Inferior Frontal Gyrus", ["inferior frontal gyrus"])
    add_from_cort(
        "Dorsolateral Prefrontal Cortex",
        ["middle frontal gyrus", "superior frontal gyrus", "frontal pole"],
    )
    add_from_cort(
        "vmPFC / dmPFC (Control & monitoring)",
        ["frontal medial cortex", "frontal orbital cortex", "subcallosal cortex", "paracingulate gyrus"],
    )
    add_from_cort(
        "Parietal Cortex",
        [
            "superior parietal lobule",
            "supramarginal gyrus",
            "angular gyrus",
            "precuneous cortex",
            "parietal opercular cortex",
            "postcentral gyrus",
        ],
    )
    add_from_cort("Precentral", ["precentral gyrus", "juxtapositional lobule cortex"])
    add_from_cort(
        "Temporal Cortex",
        [
            "temporal pole",
            "superior temporal gyrus",
            "middle temporal gyrus",
            "inferior temporal gyrus",
            "temporal fusiform cortex",
            "temporal occipital fusiform cortex",
            "planum polare",
            "planum temporale",
            "heschl",
            "parahippocampal gyrus",
        ],
    )
    add_from_sub("Thalamus", ["thalamus"])

    atlas_info = {
        "cortical_atlas": "Harvard-Oxford Cortical (thr25, 2mm)",
        "subcortical_atlas": "Harvard-Oxford Subcortical (thr25, 2mm)",
        "cerebellum_atlas": str(FSL_CEREBELLUM_MAXPROB_2MM),
    }
    return roi_groups, atlas_info, cort_data, cort_labels, sub_data, sub_labels, cereb_data


def _selected_mask_from_groups(
    groups: list[ROIGroup],
    selected_ijk: np.ndarray,
) -> np.ndarray:
    x, y, z = selected_ijk.T
    covered = np.zeros(selected_ijk.shape[0], dtype=bool)
    for group in groups:
        covered |= group.mask[x, y, z]
    return covered


def _add_relative_rois(
    base_groups: list[ROIGroup],
    selected_ijk: np.ndarray,
    active_mask: np.ndarray,
    anat_shape: tuple[int, int, int],
    cort_data: np.ndarray,
    cort_labels: list[str],
    sub_data: np.ndarray,
    sub_labels: list[str],
) -> tuple[list[ROIGroup], int]:
    x, y, z = selected_ijk.T
    covered = _selected_mask_from_groups(base_groups, selected_ijk)
    uncovered_active = active_mask & (~covered)

    if not np.any(uncovered_active):
        return [], 0

    extras: list[ROIGroup] = []
    remaining = uncovered_active.copy()

    # Ordered from specific cortical gaps to broad fallback compartments.
    candidates = [
        (
            "Occipital Cortex (relative)",
            "Harvard-Oxford Cortical (thr25, 2mm)",
            "cort",
            ["lateral occipital cortex", "occipital pole", "cuneal", "lingual", "intracalcarine", "supracalcarine"],
            None,
        ),
        (
            "Insular / Opercular Cortex (relative)",
            "Harvard-Oxford Cortical (thr25, 2mm)",
            "cort",
            ["insular cortex", "opercular cortex", "frontal opercular cortex", "central opercular cortex"],
            None,
        ),
        (
            "Hippocampus (relative)",
            "Harvard-Oxford Subcortical (thr25, 2mm)",
            "sub",
            ["hippocampus"],
            None,
        ),
        (
            "Basal Ganglia (relative)",
            "Harvard-Oxford Subcortical (thr25, 2mm)",
            "sub",
            ["caudate", "putamen", "pallidum", "accumbens"],
            None,
        ),
        (
            "Other Cerebral Cortex (relative)",
            "Harvard-Oxford Subcortical (thr25, 2mm)",
            "sub",
            ["cerebral cortex"],
            None,
        ),
    ]

    for roi_name, source, atlas_kind, include, exclude in candidates:
        if atlas_kind == "cort":
            ids, names = _select_ids_by_patterns(cort_labels, include=include, exclude=exclude)
            mask = _make_mask_from_labels(cort_data, ids, anat_shape)
        else:
            ids, names = _select_ids_by_patterns(sub_labels, include=include, exclude=exclude)
            mask = _make_mask_from_labels(sub_data, ids, anat_shape)

        covered_now = remaining & mask[x, y, z]
        n_new = int(np.count_nonzero(covered_now))
        if n_new == 0:
            continue
        extras.append(
            ROIGroup(
                name=roi_name,
                source=source,
                mask=mask,
                matched_labels=names,
            )
        )
        remaining &= ~mask[x, y, z]
        if not np.any(remaining):
            break

    # Final fallback so the created atlas can fully fit active selected voxels.
    if np.any(remaining):
        fallback_mask = np.zeros(anat_shape, dtype=bool)
        fallback_mask[x[remaining], y[remaining], z[remaining]] = True
        extras.append(
            ROIGroup(
                name="Unassigned Active Voxels (relative)",
                source="Data-driven fallback",
                mask=fallback_mask,
                matched_labels=["No atlas label matched these active selected voxels."],
            )
        )
        remaining[:] = False

    return extras, int(np.count_nonzero(remaining))


def _build_display_roi_image(groups: list[ROIGroup], anat_img: nib.Nifti1Image) -> nib.Nifti1Image:
    data = np.zeros(anat_img.shape[:3], dtype=np.int16)
    for idx, group in enumerate(groups, start=1):
        write_mask = group.mask & (data == 0)
        data[write_mask] = idx
    return nib.Nifti1Image(data, anat_img.affine, anat_img.header)


def _compute_roi_rows(
    groups: list[ROIGroup],
    selected_ijk: np.ndarray,
    active_mask: np.ndarray,
    voxel_scores: np.ndarray,
) -> list[dict]:
    rows: list[dict] = []
    x, y, z = selected_ijk.T
    n_selected_total = selected_ijk.shape[0]
    n_active_total = int(np.count_nonzero(active_mask))
    for idx, group in enumerate(groups, start=1):
        in_roi = group.mask[x, y, z]
        in_roi_active = in_roi & active_mask
        n_selected = int(np.count_nonzero(in_roi))
        n_active = int(np.count_nonzero(in_roi_active))
        rows.append(
            {
                "roi_id": idx,
                "roi_name": group.name,
                "n_selected_voxels": n_selected,
                "selected_coverage": float(n_selected) / float(n_selected_total) if n_selected_total else 0.0,
                "n_active_voxels": n_active,
                "active_coverage": float(n_active) / float(n_active_total) if n_active_total else 0.0,
                "mean_abs_beta_active": float(np.mean(voxel_scores[in_roi_active])) if n_active > 0 else float("nan"),
                "atlas_source": group.source,
                "matched_labels": "; ".join(group.matched_labels),
            }
        )
    return rows


def _write_roi_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "roi_id",
        "roi_name",
        "n_selected_voxels",
        "selected_coverage",
        "n_active_voxels",
        "active_coverage",
        "mean_abs_beta_active",
        "atlas_source",
        "matched_labels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_roi_figure(
    roi_img: nib.Nifti1Image,
    anat_img: nib.Nifti1Image,
    rows: list[dict],
    out_png: Path,
) -> None:
    n_rois = len(rows)
    cmap_name = "tab20" if n_rois > 10 else "tab10"
    cmap = plt.get_cmap(cmap_name, max(n_rois, 1))

    fig = plt.figure(figsize=(17, 8))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.1])
    ax_brain = fig.add_subplot(grid[0, 0])
    ax_text = fig.add_subplot(grid[0, 1])

    cut_coords = plotting.find_xyz_cut_coords(roi_img)
    plotting.plot_roi(
        roi_img,
        bg_img=anat_img,
        display_mode="ortho",
        cut_coords=cut_coords,
        draw_cross=False,
        cmap=cmap,
        alpha=0.65,
        colorbar=False,
        figure=fig,
        axes=ax_brain,
    )

    ax_text.axis("off")
    y = 0.99
    line_step = min(0.082, 0.90 / max(n_rois, 1))
    font_size = 8 if n_rois > 12 else 9
    for row in rows:
        idx = int(row["roi_id"]) - 1
        color = cmap(idx)
        ax_text.add_patch(
            plt.Rectangle(
                (0.0, y - 0.017),
                0.026,
                0.016,
                color=color,
                transform=ax_text.transAxes,
                clip_on=False,
            )
        )
        text = f"{row['roi_name']} ({row['n_active_voxels']} active voxels)"
        ax_text.text(0.035, y, text, fontsize=font_size, va="top", transform=ax_text.transAxes)
        y -= line_step

    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.05, wspace=0.08)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    anat_path = args.anat_path.expanduser().resolve()
    voxel_indices_path = args.voxel_indices_path.expanduser().resolve()
    cache_dir = data_dir / "atlas_cache"

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not anat_path.exists():
        raise FileNotFoundError(f"Anatomy not found: {anat_path}")
    if not voxel_indices_path.exists():
        raise FileNotFoundError(f"Voxel indices not found: {voxel_indices_path}")

    active_beta_file = Path(args.active_beta_file)
    if not active_beta_file.is_absolute():
        active_beta_file = (data_dir / active_beta_file).resolve()
    if not active_beta_file.exists():
        raise FileNotFoundError(f"Active-beta file not found: {active_beta_file}")

    beta_files = sorted(data_dir.glob(args.beta_pattern))
    if args.max_beta_files > 0:
        beta_files = beta_files[: args.max_beta_files]
    if not beta_files:
        raise FileNotFoundError(f"No beta files matched '{args.beta_pattern}' in {data_dir}")

    anat_img = nib.load(str(anat_path))
    anat_shape = anat_img.shape[:3]
    print(f"Anatomy: {anat_path}", flush=True)
    print(f"Shape: {anat_shape}, zooms: {tuple(float(v) for v in anat_img.header.get_zooms()[:3])}", flush=True)

    selected_ijk = _load_selected_ijk(voxel_indices_path, anat_shape)
    print(f"Selected voxels: {selected_ijk.shape[0]}", flush=True)

    active_mask = _load_active_mask(active_beta_file, selected_ijk.shape[0])
    print(
        f"Active voxels from {active_beta_file.name}: {int(np.count_nonzero(active_mask))} / {selected_ijk.shape[0]}",
        flush=True,
    )

    voxel_scores, used_beta_files = _load_and_pool_beta_scores(
        beta_files=beta_files,
        n_voxels=selected_ijk.shape[0],
        chunk_trials=args.chunk_trials,
    )
    print(f"Pooled mean|beta| from {len(used_beta_files)} beta files.", flush=True)

    (
        base_groups,
        atlas_info,
        cort_data,
        cort_labels,
        sub_data,
        sub_labels,
        cereb_data,
    ) = _build_base_requested_rois(anat_img=anat_img, cache_dir=cache_dir)

    x, y, z = selected_ijk.T
    sub_labels_at_selected = sub_data[x, y, z]
    second_atlas_cov = float(np.mean(sub_labels_at_selected[active_mask] > 0)) if np.any(active_mask) else 0.0

    # Coverage of "second suggested atlas + cerebellum"
    second_plus_cereb = (sub_labels_at_selected > 0) | (cereb_data[x, y, z] > 0)
    second_plus_cereb_cov = (
        float(np.mean(second_plus_cereb[active_mask])) if np.any(active_mask) else 0.0
    )

    groups = list(base_groups)
    relative_groups: list[ROIGroup] = []
    remaining_uncovered_after_relative = 0
    if args.add_relative_rois:
        relative_groups, remaining_uncovered_after_relative = _add_relative_rois(
            base_groups=groups,
            selected_ijk=selected_ijk,
            active_mask=active_mask,
            anat_shape=anat_shape,
            cort_data=cort_data,
            cort_labels=cort_labels,
            sub_data=sub_data,
            sub_labels=sub_labels,
        )
        groups.extend(relative_groups)

    roi_img = _build_display_roi_image(groups, anat_img=anat_img)
    roi_data = roi_img.get_fdata().astype(np.int16)
    final_labels_at_selected = roi_data[x, y, z]
    final_active_cov = (
        float(np.mean(final_labels_at_selected[active_mask] > 0)) if np.any(active_mask) else 0.0
    )

    rows = _compute_roi_rows(
        groups=groups,
        selected_ijk=selected_ijk,
        active_mask=active_mask,
        voxel_scores=voxel_scores,
    )

    roi_img_path = out_dir / "created_rois_fitted.nii.gz"
    fig_path = out_dir / "created_rois_fitted.png"
    csv_path = out_dir / "created_roi_stats.csv"
    summary_path = out_dir / "created_roi_summary.json"

    nib.save(roi_img, str(roi_img_path))
    _plot_roi_figure(roi_img=roi_img, anat_img=anat_img, rows=rows, out_png=fig_path)
    _write_roi_csv(csv_path, rows)

    summary = {
        "anat_path": str(anat_path),
        "active_beta_file": str(active_beta_file),
        "used_beta_files": used_beta_files,
        "atlas_info": atlas_info,
        "coverage": {
            "second_suggested_atlas_active_coverage": second_atlas_cov,
            "second_atlas_plus_cerebellum_active_coverage": second_plus_cereb_cov,
            "base_requested_rois_active_coverage": float(
                np.mean(_selected_mask_from_groups(base_groups, selected_ijk)[active_mask]) if np.any(active_mask) else 0.0
            ),
            "final_created_atlas_active_coverage": final_active_cov,
            "remaining_uncovered_active_voxels_after_relative": remaining_uncovered_after_relative,
        },
        "n_base_requested_rois": len(base_groups),
        "n_relative_rois_added": len(relative_groups),
        "relative_roi_names": [group.name for group in relative_groups],
        "all_roi_names_in_order": [group.name for group in groups],
        "roi_rows": rows,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Second suggested atlas active coverage: {second_atlas_cov * 100:.2f}%",
        flush=True,
    )
    print(
        f"Second atlas + cerebellum active coverage: {second_plus_cereb_cov * 100:.2f}%",
        flush=True,
    )
    print(
        f"Final created atlas active coverage: {final_active_cov * 100:.2f}%",
        flush=True,
    )
    print(f"Relative ROIs added: {len(relative_groups)}", flush=True)

    print(f"Saved ROI image: {roi_img_path}", flush=True)
    print(f"Saved ROI figure: {fig_path}", flush=True)
    print(f"Saved ROI table: {csv_path}", flush=True)
    print(f"Saved ROI summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
