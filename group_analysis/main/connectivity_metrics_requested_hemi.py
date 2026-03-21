#!/usr/bin/env python3
"""
Build a hemisphere-specific connectivity ROI atlas for a fixed requested ROI list.

This script does not modify the existing connectivity atlas workflow. It reuses
the selected-voxel analysis volume and beta summaries, but writes all outputs to
its own directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colors as mcolors
from nilearn import plotting

from connectivity_metrics import (
    FSL_CEREBELLUM_MAXPROB_2MM,
    ROIGroup,
    _fetch_ho_cort_sub,
    _load_and_pool_beta_scores,
    _load_cerebellum_label_img,
    _load_selected_ijk,
    _make_mask_from_labels,
    _resample_label_img,
    _select_ids_by_patterns,
)


DEFAULT_ANAT_PATH = Path("results/connectivity/tmp/data/MNI152_T1_2mm_brain.nii.gz")
DEFAULT_OUT_DIR = Path("results/connectivity/atlas_requested_hemi")
MIDLINE_POLICY = "exclude_x_eq_0"


@dataclass(frozen=True)
class RequestedROISpec:
    roi_name: str
    atlas_kind: str
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    cerebellum_labels: tuple[str, ...] = ()


REQUESTED_SPECS: tuple[RequestedROISpec, ...] = (
    RequestedROISpec(
        roi_name="Precentral gyrus (primary motor cortex)",
        atlas_kind="cort",
        include_patterns=("precentral gyrus",),
    ),
    RequestedROISpec(
        roi_name="Putamen (basal ganglia)",
        atlas_kind="sub",
        include_patterns=("putamen",),
    ),
    RequestedROISpec(
        roi_name="Cerebellum (lobules VIIIa, VIIb)",
        atlas_kind="cereb",
        cerebellum_labels=("VIIIa", "VIIb"),
    ),
    RequestedROISpec(
        roi_name="Frontal medial cortex (SMA/pre-SMA)",
        atlas_kind="cort",
        include_patterns=("frontal medial cortex", "juxtapositional lobule cortex"),
    ),
    RequestedROISpec(
        roi_name="Frontal Pole",
        atlas_kind="cort",
        include_patterns=("frontal pole",),
    ),
    RequestedROISpec(
        roi_name="Superior parietal lobule",
        atlas_kind="cort",
        include_patterns=("superior parietal lobule",),
    ),
    RequestedROISpec(
        roi_name="Cerebellar Crus II",
        atlas_kind="cereb",
        cerebellum_labels=("Crus II",),
    ),
    RequestedROISpec(
        roi_name="Insular cortex",
        atlas_kind="cort",
        include_patterns=("insular cortex",),
    ),
    RequestedROISpec(
        roi_name="Hippocampus",
        atlas_kind="sub",
        include_patterns=("hippocampus",),
    ),
    RequestedROISpec(
        roi_name="Parahippocampal gyrus",
        atlas_kind="cort",
        include_patterns=("parahippocampal gyrus",),
    ),
    RequestedROISpec(
        roi_name="Precuneus",
        atlas_kind="cort",
        include_patterns=("precuneous cortex",),
    ),
    RequestedROISpec(
        roi_name="Amygdala",
        atlas_kind="sub",
        include_patterns=("amygdala",),
    ),
    RequestedROISpec(
        roi_name="Temporal cortex",
        atlas_kind="cort",
        include_patterns=(
            "temporal pole",
            "superior temporal gyrus",
            "middle temporal gyrus",
            "inferior temporal gyrus",
            "planum polare",
            "heschl",
            "planum temporale",
        ),
    ),
    RequestedROISpec(
        roi_name="Fusiform cortex",
        atlas_kind="cort",
        include_patterns=(
            "temporal fusiform cortex",
            "temporal occipital fusiform cortex",
            "occipital fusiform gyrus",
        ),
    ),
    RequestedROISpec(
        roi_name="Lateral occipital cortex",
        atlas_kind="cort",
        include_patterns=("lateral occipital cortex",),
    ),
    RequestedROISpec(
        roi_name="Cerebral Cortex",
        atlas_kind="sub",
        include_patterns=("cerebral cortex",),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a hemisphere-specific requested ROI atlas in the selected-voxel "
            "connectivity analysis volume."
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
        default=DEFAULT_ANAT_PATH,
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
        default="selected_beta_trials.npy",
        help="File used to define active voxels. Relative paths are resolved within --data-dir.",
    )
    parser.add_argument(
        "--max-beta-files",
        type=int,
        default=0,
        help="Optional cap on beta files to use (0 means all).",
    )
    parser.add_argument(
        "--chunk-trials",
        type=int,
        default=256,
        help="Chunk size along trials when computing mean|beta|.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for the requested-ROI atlas artifacts.",
    )
    return parser.parse_args()


def _load_cerebellum_value_map() -> dict[str, int]:
    xml_path = Path("/usr/local/fsl/data/atlases/Cerebellum_MNIfnirt.xml")
    if not xml_path.exists():
        raise FileNotFoundError(f"Cerebellum atlas XML not found: {xml_path}")
    root = ET.fromstring(xml_path.read_text(encoding="utf-8"))
    mapping: dict[str, int] = {}
    for label in root.findall(".//label"):
        index = int(label.attrib["index"]) + 1
        name = (label.text or "").strip()
        mapping[name] = index
    return mapping


def _load_requested_active_mask(
    active_beta_file: Path,
    beta_files: list[Path],
    n_voxels: int,
) -> tuple[np.ndarray, str]:
    if active_beta_file.exists():
        try:
            from connectivity_metrics import _load_active_mask

            return _load_active_mask(active_beta_file, n_voxels), active_beta_file.name
        except ValueError:
            pass

    active = np.zeros(n_voxels, dtype=bool)
    used_files: list[str] = []
    for beta_path in beta_files:
        beta = np.load(beta_path, mmap_mode="r")
        if beta.ndim != 2 or beta.shape[0] != n_voxels:
            continue
        active |= np.any(np.isfinite(beta) & (beta != 0), axis=1)
        used_files.append(beta_path.name)

    if not used_files:
        raise ValueError("Could not derive an active mask from any beta file matching the selected voxel count.")
    return active, f"union_nonzero_across_matching_beta_files:{len(used_files)}"


def _hemisphere_planes(anat_img: nib.Nifti1Image) -> dict[str, np.ndarray]:
    nx = anat_img.shape[0]
    x_world = anat_img.affine[0, 0] * np.arange(nx, dtype=np.float32) + anat_img.affine[0, 3]
    return {
        "L": (x_world < 0)[:, None, None],
        "R": (x_world > 0)[:, None, None],
        "midline": (x_world == 0)[:, None, None],
    }


def _mask_from_cerebellum_labels(
    cereb_data: np.ndarray,
    cereb_value_map: dict[str, int],
    hemisphere: str,
    requested_names: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    if hemisphere == "L":
        prefix = "Left "
    elif hemisphere == "R":
        prefix = "Right "
    else:
        raise ValueError(f"Unknown hemisphere: {hemisphere}")
    exact_names = [f"{prefix}{name}" for name in requested_names]
    values = [cereb_value_map[name] for name in exact_names if name in cereb_value_map]
    return _make_mask_from_labels(cereb_data, values, cereb_data.shape), exact_names


def _mask_from_subcortical_labels(
    sub_data: np.ndarray,
    sub_labels: list[str],
    hemisphere: str,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    hemi_word = "left" if hemisphere == "L" else "right"
    include_lower = [item.lower() for item in include_patterns]
    exclude_lower = [item.lower() for item in exclude_patterns]
    ids: list[int] = []
    names: list[str] = []
    for idx, label in enumerate(sub_labels):
        if idx == 0:
            continue
        lname = label.lower()
        if hemi_word not in lname:
            continue
        if not any(pattern in lname for pattern in include_lower):
            continue
        if any(pattern in lname for pattern in exclude_lower):
            continue
        ids.append(idx)
        names.append(label)
    return _make_mask_from_labels(sub_data, ids, sub_data.shape), names


def _mask_from_cortical_labels(
    cort_data: np.ndarray,
    cort_labels: list[str],
    hemi_plane: np.ndarray,
    include_patterns: tuple[str, ...],
    exclude_patterns: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    ids, names = _select_ids_by_patterns(
        cort_labels,
        include=list(include_patterns),
        exclude=list(exclude_patterns),
    )
    mask = _make_mask_from_labels(cort_data, ids, cort_data.shape)
    return mask & hemi_plane, names


def _build_requested_groups(
    anat_img: nib.Nifti1Image,
    cache_dir: Path,
) -> tuple[list[ROIGroup], dict]:
    cort_img, cort_labels, sub_img, sub_labels = _fetch_ho_cort_sub(cache_dir)
    cereb_img = _load_cerebellum_label_img()
    cort_data = _resample_label_img(cort_img, anat_img)
    sub_data = _resample_label_img(sub_img, anat_img)
    cereb_data = _resample_label_img(cereb_img, anat_img)
    cereb_value_map = _load_cerebellum_value_map()
    hemi_planes = _hemisphere_planes(anat_img)

    groups: list[ROIGroup] = []
    mapping_rows: list[dict[str, str]] = []

    for hemisphere in ("L", "R"):
        hemi_plane = hemi_planes[hemisphere]
        for spec in REQUESTED_SPECS:
            if spec.atlas_kind == "cort":
                mask, matched_labels = _mask_from_cortical_labels(
                    cort_data=cort_data,
                    cort_labels=cort_labels,
                    hemi_plane=hemi_plane,
                    include_patterns=spec.include_patterns,
                    exclude_patterns=spec.exclude_patterns,
                )
                source = "Harvard-Oxford Cortical (thr25, 2mm), hemisphere split in MNI space"
            elif spec.atlas_kind == "sub":
                mask, matched_labels = _mask_from_subcortical_labels(
                    sub_data=sub_data,
                    sub_labels=sub_labels,
                    hemisphere=hemisphere,
                    include_patterns=spec.include_patterns,
                    exclude_patterns=spec.exclude_patterns,
                )
                source = "Harvard-Oxford Subcortical (thr25, 2mm)"
            elif spec.atlas_kind == "cereb":
                mask, matched_labels = _mask_from_cerebellum_labels(
                    cereb_data=cereb_data,
                    cereb_value_map=cereb_value_map,
                    hemisphere=hemisphere,
                    requested_names=spec.cerebellum_labels,
                )
                source = "FSL Cerebellum MNIfnirt (maxprob thr25, 2mm)"
            else:
                raise ValueError(f"Unknown atlas kind: {spec.atlas_kind}")

            groups.append(
                ROIGroup(
                    name=f"{hemisphere} {spec.roi_name}",
                    source=source,
                    mask=mask,
                    matched_labels=matched_labels,
                )
            )
            mapping_rows.append(
                {
                    "hemisphere": hemisphere,
                    "roi_name": spec.roi_name,
                    "atlas_kind": spec.atlas_kind,
                    "atlas_source": source,
                    "matched_labels": "; ".join(matched_labels),
                }
            )

    metadata = {
        "hemisphere_rule": {
            "policy": MIDLINE_POLICY,
            "left_definition_mni_x": "< 0",
            "right_definition_mni_x": "> 0",
        },
        "mapping_rows": mapping_rows,
    }
    return groups, metadata


def _assign_disjoint_groups(groups: list[ROIGroup], anat_img: nib.Nifti1Image) -> tuple[list[ROIGroup], nib.Nifti1Image, int]:
    data = np.zeros(anat_img.shape[:3], dtype=np.int16)
    assigned_groups: list[ROIGroup] = []
    n_overlap_voxels = 0
    for idx, group in enumerate(groups, start=1):
        write_mask = group.mask & (data == 0)
        n_overlap_voxels += int(np.count_nonzero(group.mask & (data != 0)))
        data[write_mask] = idx
        assigned_groups.append(
            ROIGroup(
                name=group.name,
                source=group.source,
                mask=write_mask,
                matched_labels=group.matched_labels,
            )
        )
    return assigned_groups, nib.Nifti1Image(data, anat_img.affine, anat_img.header), n_overlap_voxels


def _count_selected_overlap_voxels(groups: list[ROIGroup], selected_ijk: np.ndarray) -> int:
    x, y, z = selected_ijk.T
    membership = np.zeros(selected_ijk.shape[0], dtype=np.int16)
    for group in groups:
        membership += group.mask[x, y, z].astype(np.int16)
    return int(np.count_nonzero(membership > 1))


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
        in_template = group.mask
        in_selected = group.mask[x, y, z]
        in_active = in_selected & active_mask
        rows.append(
            {
                "roi_id": idx,
                "hemisphere": group.name.split(" ", 1)[0],
                "roi_name": group.name.split(" ", 1)[1],
                "full_roi_name": group.name,
                "n_template_voxels": int(np.count_nonzero(in_template)),
                "n_selected_voxels": int(np.count_nonzero(in_selected)),
                "selected_coverage": float(np.count_nonzero(in_selected)) / float(n_selected_total) if n_selected_total else 0.0,
                "n_active_voxels": int(np.count_nonzero(in_active)),
                "active_coverage": float(np.count_nonzero(in_active)) / float(n_active_total) if n_active_total else 0.0,
                "mean_abs_beta_active": float(np.mean(voxel_scores[in_active])) if np.any(in_active) else float("nan"),
                "atlas_source": group.source,
                "matched_labels": "; ".join(group.matched_labels),
            }
        )
    return rows


def _write_roi_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "roi_id",
        "hemisphere",
        "roi_name",
        "full_roi_name",
        "n_template_voxels",
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


def _build_categorical_cmap(n_rois: int) -> mcolors.ListedColormap:
    base = np.vstack(
        [
            plt.get_cmap("tab20")(np.linspace(0, 1, 20)),
            plt.get_cmap("tab20b")(np.linspace(0, 1, 20)),
            plt.get_cmap("tab20c")(np.linspace(0, 1, 20)),
        ]
    )
    colors = base[: max(n_rois, 1), :]
    return mcolors.ListedColormap(colors)


def _plot_atlas(
    roi_img: nib.Nifti1Image,
    anat_img: nib.Nifti1Image,
    rows: list[dict],
    out_png: Path,
    out_pdf: Path,
) -> None:
    n_rois = len(rows)
    cmap = _build_categorical_cmap(n_rois)

    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.25])
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
        alpha=0.72,
        colorbar=False,
        black_bg=False,
        figure=fig,
        axes=ax_brain,
    )
    ax_brain.set_title("Requested Hemisphere-Specific ROI Atlas", fontsize=14)

    ax_text.axis("off")
    n_cols = 2
    rows_per_col = int(np.ceil(n_rois / n_cols))
    line_step = 0.060
    font_size = 8.2
    for idx, row in enumerate(rows):
        col_idx = idx // rows_per_col
        row_idx = idx % rows_per_col
        x0 = 0.02 + col_idx * 0.49
        y0 = 0.98 - row_idx * line_step
        color = cmap.colors[idx]
        ax_text.add_patch(
            plt.Rectangle(
                (x0, y0 - 0.018),
                0.020,
                0.015,
                color=color,
                transform=ax_text.transAxes,
                clip_on=False,
            )
        )
        label = f"{row['full_roi_name']} [{row['n_selected_voxels']} sel / {row['n_active_voxels']} active]"
        ax_text.text(x0 + 0.028, y0, label, fontsize=font_size, va="top", transform=ax_text.transAxes)

    fig.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.04, wspace=0.05)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def _plot_voxel_counts(rows: list[dict], out_png: Path, out_pdf: Path) -> None:
    labels = [row["full_roi_name"] for row in rows]
    selected = np.array([row["n_selected_voxels"] for row in rows], dtype=np.int32)
    active = np.array([row["n_active_voxels"] for row in rows], dtype=np.int32)
    y = np.arange(len(rows), dtype=np.float32)
    colors = ["#1f77b4" if row["hemisphere"] == "L" else "#d62728" for row in rows]

    fig_h = max(10.0, 0.35 * len(rows) + 2.0)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(y + 0.18, selected, height=0.34, color=colors, alpha=0.35, label="Selected voxels")
    ax.barh(y - 0.18, active, height=0.34, color=colors, alpha=0.95, label="Active voxels")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Voxel count")
    ax.set_title("Requested Hemisphere-Specific ROI Voxel Counts")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    anat_path = args.anat_path.expanduser().resolve()
    voxel_indices_path = args.voxel_indices_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
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
    selected_ijk = _load_selected_ijk(voxel_indices_path, anat_img.shape[:3])
    active_mask, active_mask_source = _load_requested_active_mask(
        active_beta_file=active_beta_file,
        beta_files=beta_files,
        n_voxels=selected_ijk.shape[0],
    )
    voxel_scores, used_beta_files = _load_and_pool_beta_scores(
        beta_files=beta_files,
        n_voxels=selected_ijk.shape[0],
        chunk_trials=args.chunk_trials,
    )

    groups, metadata = _build_requested_groups(anat_img=anat_img, cache_dir=cache_dir)
    n_overlap_voxels = _count_selected_overlap_voxels(groups=groups, selected_ijk=selected_ijk)
    groups, roi_img, _ = _assign_disjoint_groups(groups=groups, anat_img=anat_img)
    rows = _compute_roi_rows(
        groups=groups,
        selected_ijk=selected_ijk,
        active_mask=active_mask,
        voxel_scores=voxel_scores,
    )

    x, y, z = selected_ijk.T
    covered_selected = roi_img.get_fdata().astype(np.int16)[x, y, z] > 0
    selected_coverage = float(np.mean(covered_selected)) if covered_selected.size else 0.0
    active_coverage = float(np.mean(covered_selected[active_mask])) if np.any(active_mask) else 0.0

    roi_img_path = out_dir / "requested_hemi_rois_fitted.nii.gz"
    atlas_png_path = out_dir / "requested_hemi_rois_fitted.png"
    atlas_pdf_path = out_dir / "requested_hemi_rois_fitted.pdf"
    counts_png_path = out_dir / "requested_hemi_roi_voxel_counts.png"
    counts_pdf_path = out_dir / "requested_hemi_roi_voxel_counts.pdf"
    csv_path = out_dir / "requested_hemi_roi_stats.csv"
    summary_path = out_dir / "requested_hemi_roi_summary.json"

    nib.save(roi_img, str(roi_img_path))
    _write_roi_csv(csv_path, rows)
    _plot_atlas(roi_img=roi_img, anat_img=anat_img, rows=rows, out_png=atlas_png_path, out_pdf=atlas_pdf_path)
    _plot_voxel_counts(rows=rows, out_png=counts_png_path, out_pdf=counts_pdf_path)

    summary = {
        "anat_path": str(anat_path),
        "voxel_indices_path": str(voxel_indices_path),
        "active_beta_file": str(active_beta_file),
        "active_mask_source": active_mask_source,
        "used_beta_files": used_beta_files,
        "n_requested_rois": len(rows),
        "selected_voxel_count_total": int(selected_ijk.shape[0]),
        "active_voxel_count_total": int(np.count_nonzero(active_mask)),
        "selected_voxel_coverage_by_requested_rois": selected_coverage,
        "active_voxel_coverage_by_requested_rois": active_coverage,
        "overlap_voxels_resolved_by_priority_order": n_overlap_voxels,
        "requested_roi_specifications": [spec.roi_name for spec in REQUESTED_SPECS],
        "metadata": metadata,
        "roi_rows": rows,
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Selected voxels: {selected_ijk.shape[0]}", flush=True)
    print(f"Active voxels: {int(np.count_nonzero(active_mask))}", flush=True)
    print(f"Selected voxel coverage by requested ROIs: {selected_coverage * 100:.2f}%", flush=True)
    print(f"Active voxel coverage by requested ROIs: {active_coverage * 100:.2f}%", flush=True)
    print(f"Resolved overlap voxels: {n_overlap_voxels}", flush=True)
    print(f"Saved ROI image: {roi_img_path}", flush=True)
    print(f"Saved atlas figure: {atlas_png_path}", flush=True)
    print(f"Saved voxel-count figure: {counts_png_path}", flush=True)
    print(f"Saved ROI table: {csv_path}", flush=True)
    print(f"Saved ROI summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
