#!/usr/bin/env python3
"""Build a reportable MNI2mm atlas for selected connectivity voxels.

The atlas uses Schaefer-2018 cortical parcels, FSL MNIfnirt cerebellar lobules,
and Harvard-Oxford subcortical gray nuclei. Voxels overlapping background,
white matter, lateral ventricles, or brainstem masks are left unlabeled.
"""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import datasets, image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANAT_PATH = REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "MNI152_T1_2mm_brain.nii.gz"
DEFAULT_VOXEL_INDICES = REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "selected_voxel_indices.npz"
DEFAULT_ACTIVE_BETA = REPO_ROOT / "results" / "connectivity" / "tmp" / "data" / "selected_beta_trials.npy"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "connectivity_new" / "atlas_schaefer100_cerebellum_subcortical"

FSL_BRAIN_MASK = Path("/usr/local/fsl/data/standard/MNI152_T1_2mm_brain_mask.nii.gz")
FSL_WHITE_PRIOR = Path("/usr/local/fsl/data/standard/tissuepriors/avg152T1_white.img")
FSL_CEREBELLUM_XML = Path("/usr/local/fsl/data/atlases/Cerebellum_MNIfnirt.xml")
FSL_CEREBELLUM_TEMPLATE = "/usr/local/fsl/data/atlases/Cerebellum/Cerebellum-MNIfnirt-maxprob-thr{thr}-2mm.nii.gz"
HO_SUB_TEMPLATE = "/usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr{thr}-2mm.nii.gz"

HO_EXCLUDE_LABELS = {
    1: "Left Cerebral White Matter",
    3: "Left Lateral Ventricle",
    8: "Brain-Stem",
    12: "Right Cerebral White Matter",
    14: "Right Lateral Ventricle",
}

HO_SUBCORTICAL_KEEP = {
    4: "Left Thalamus",
    5: "Left Caudate",
    6: "Left Putamen",
    7: "Left Pallidum",
    9: "Left Hippocampus",
    10: "Left Amygdala",
    11: "Left Accumbens",
    15: "Right Thalamus",
    16: "Right Caudate",
    17: "Right Putamen",
    18: "Right Pallidum",
    19: "Right Hippocampus",
    20: "Right Amygdala",
    21: "Right Accumbens",
}


@dataclass(frozen=True)
class ROIGroup:
    name: str
    source: str
    mask: np.ndarray
    matched_labels: list[str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Schaefer cortical + FSL cerebellar + Harvard-Oxford "
            "subcortical atlas in the selected-voxel MNI2mm analysis space."
        )
    )
    parser.add_argument("--anat-path", type=Path, default=DEFAULT_ANAT_PATH)
    parser.add_argument("--voxel-indices-path", type=Path, default=DEFAULT_VOXEL_INDICES)
    parser.add_argument("--active-beta-file", type=Path, default=DEFAULT_ACTIVE_BETA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--schaefer-n-rois", type=int, default=100)
    parser.add_argument("--schaefer-yeo-networks", type=int, default=7)
    parser.add_argument("--cerebellum-threshold", type=int, default=0)
    parser.add_argument("--harvard-oxford-threshold", type=int, default=25)
    parser.add_argument("--white-prior-threshold", type=float, default=0.5)
    parser.add_argument("--active-row-chunk", type=int, default=2048)
    return parser.parse_args()


def _load_selected_ijk(path: Path, volume_shape: tuple[int, int, int]) -> np.ndarray:
    pack = np.load(path, allow_pickle=True)
    if "selected_ijk" in pack.files:
        ijk = np.asarray(pack["selected_ijk"], dtype=np.int32)
    elif "selected_flat_indices" in pack.files:
        flat = np.asarray(pack["selected_flat_indices"], dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, volume_shape)).astype(np.int32, copy=False)
    else:
        raise KeyError(f"{path} must contain selected_ijk or selected_flat_indices.")

    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected selected_ijk shape (N, 3); got {ijk.shape}")

    valid = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < volume_shape[0])
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < volume_shape[1])
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < volume_shape[2])
    )
    if not np.all(valid):
        dropped = int(np.count_nonzero(~valid))
        print(f"Warning: dropping {dropped} selected voxels outside the reference image.", flush=True)
        ijk = ijk[valid]
    return ijk


def _resample_mask_img(path: Path, ref_img: nib.Nifti1Image, threshold: float = 0.0) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine):
        img = image.resample_to_img(img, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
    data = np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))
    return data > float(threshold)


def _load_label_img(path: Path, ref_img: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine):
        img = image.resample_to_img(img, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
    return np.rint(img.get_fdata()).astype(np.int32)


def _load_active_mask(active_beta_file: Path, n_voxels: int, row_chunk: int) -> tuple[np.ndarray, str]:
    if not active_beta_file.exists():
        return np.ones(n_voxels, dtype=bool), "all_selected_voxels_active_missing_active_beta_file"

    beta = np.load(active_beta_file, mmap_mode="r")
    if beta.ndim != 2 or beta.shape[0] != n_voxels:
        return np.ones(n_voxels, dtype=bool), "all_selected_voxels_active_active_beta_shape_mismatch"

    active = np.zeros(n_voxels, dtype=bool)
    row_chunk = int(max(1, row_chunk))
    for start in range(0, n_voxels, row_chunk):
        stop = min(start + row_chunk, n_voxels)
        block = np.asarray(beta[start:stop, :], dtype=np.float32)
        active[start:stop] = np.any(np.isfinite(block) & (block != 0), axis=1)
    return active, str(active_beta_file)


def _load_cerebellum_labels() -> dict[int, str]:
    if not FSL_CEREBELLUM_XML.exists():
        raise FileNotFoundError(f"FSL cerebellum XML not found: {FSL_CEREBELLUM_XML}")
    root = ET.fromstring(FSL_CEREBELLUM_XML.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    for label in root.findall(".//label"):
        out[int(label.attrib["index"]) + 1] = (label.text or "").strip()
    return out


def _load_schaefer(ref_img: nib.Nifti1Image, n_rois: int, yeo_networks: int, cache_dir: Path) -> tuple[np.ndarray, list[str], str]:
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=int(n_rois),
        yeo_networks=int(yeo_networks),
        resolution_mm=2,
        data_dir=str(cache_dir),
        verbose=0,
    )
    atlas_img = nib.load(str(atlas.maps))
    if atlas_img.shape[:3] != ref_img.shape[:3] or not np.allclose(atlas_img.affine, ref_img.affine):
        atlas_img = image.resample_to_img(
            atlas_img,
            ref_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    labels = [item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item) for item in atlas.labels]
    return np.rint(atlas_img.get_fdata()).astype(np.int32), labels, str(atlas.maps)


def _build_exclusion_masks(
    ref_img: nib.Nifti1Image,
    selected_ijk: np.ndarray,
    white_prior_threshold: float,
    ho_sub_data: np.ndarray,
) -> tuple[np.ndarray, dict]:
    x, y, z = selected_ijk.T
    brain_mask = _resample_mask_img(FSL_BRAIN_MASK, ref_img, threshold=0.0)
    white_mask = _resample_mask_img(FSL_WHITE_PRIOR, ref_img, threshold=float(white_prior_threshold))
    background_mask = ~brain_mask
    ho_exclusion_mask = np.isin(ho_sub_data, list(HO_EXCLUDE_LABELS.keys()))
    exclusion_mask = background_mask | white_mask | ho_exclusion_mask

    category_masks = {
        "background_fsl_brain_mask": background_mask,
        f"white_matter_fsl_prior_ge_{white_prior_threshold:g}": white_mask,
        "white_matter_harvard_oxford": np.isin(ho_sub_data, [1, 12]),
        "brainstem_harvard_oxford": ho_sub_data == 8,
        "lateral_ventricle_harvard_oxford": np.isin(ho_sub_data, [3, 14]),
    }
    rows = []
    for name, mask in category_masks.items():
        at_selected = mask[x, y, z]
        rows.append(
            {
                "category": name,
                "n_selected_voxels": int(np.count_nonzero(at_selected)),
                "pct_selected_voxels": float(np.mean(at_selected) * 100.0) if at_selected.size else 0.0,
            }
        )
    union_at_selected = exclusion_mask[x, y, z]
    qc = {
        "exclusion_categories": rows,
        "n_selected_excluded_union": int(np.count_nonzero(union_at_selected)),
        "pct_selected_excluded_union": float(np.mean(union_at_selected) * 100.0) if union_at_selected.size else 0.0,
        "white_prior_threshold": float(white_prior_threshold),
        "harvard_oxford_excluded_labels": HO_EXCLUDE_LABELS,
    }
    return exclusion_mask, qc


def _groups_from_sources(
    ref_img: nib.Nifti1Image,
    selected_ijk: np.ndarray,
    exclusion_mask: np.ndarray,
    schaefer_n_rois: int,
    schaefer_yeo_networks: int,
    cerebellum_threshold: int,
    ho_threshold: int,
    cache_dir: Path,
) -> tuple[list[ROIGroup], dict]:
    x, y, z = selected_ijk.T
    groups: list[ROIGroup] = []

    cereb_path = Path(FSL_CEREBELLUM_TEMPLATE.format(thr=int(cerebellum_threshold)))
    cereb_data = _load_label_img(cereb_path, ref_img)
    cereb_labels = _load_cerebellum_labels()
    for value, label in cereb_labels.items():
        mask = (cereb_data == int(value)) & ~exclusion_mask
        if np.any(mask[x, y, z]):
            groups.append(
                ROIGroup(
                    name=f"Cerebellum {label}",
                    source=f"FSL Cerebellum MNIfnirt maxprob-thr{int(cerebellum_threshold)} 2mm",
                    mask=mask,
                    matched_labels=[label],
                )
            )

    ho_sub_path = Path(HO_SUB_TEMPLATE.format(thr=int(ho_threshold)))
    ho_sub_data = _load_label_img(ho_sub_path, ref_img)
    for value, label in HO_SUBCORTICAL_KEEP.items():
        mask = (ho_sub_data == int(value)) & ~exclusion_mask
        if np.any(mask[x, y, z]):
            groups.append(
                ROIGroup(
                    name=label,
                    source=f"Harvard-Oxford Subcortical maxprob-thr{int(ho_threshold)} 2mm",
                    mask=mask,
                    matched_labels=[label],
                )
            )

    schaefer_data, schaefer_labels, schaefer_path = _load_schaefer(
        ref_img=ref_img,
        n_rois=schaefer_n_rois,
        yeo_networks=schaefer_yeo_networks,
        cache_dir=cache_dir,
    )
    for value in range(1, len(schaefer_labels)):
        label = schaefer_labels[value]
        mask = (schaefer_data == int(value)) & ~exclusion_mask
        if np.any(mask[x, y, z]):
            groups.append(
                ROIGroup(
                    name=label,
                    source=f"Schaefer 2018 {int(schaefer_n_rois)}-parcel {int(schaefer_yeo_networks)}-network 2mm",
                    mask=mask,
                    matched_labels=[label],
                )
            )

    metadata = {
        "schaefer_path": schaefer_path,
        "schaefer_n_rois": int(schaefer_n_rois),
        "schaefer_yeo_networks": int(schaefer_yeo_networks),
        "cerebellum_path": str(cereb_path),
        "cerebellum_threshold": int(cerebellum_threshold),
        "harvard_oxford_subcortical_path": str(ho_sub_path),
        "harvard_oxford_threshold": int(ho_threshold),
        "assignment_priority": [
            "FSL cerebellar lobules",
            "Harvard-Oxford subcortical gray nuclei",
            "Schaefer cortical parcels",
        ],
    }
    return groups, metadata


def _assign_disjoint(groups: list[ROIGroup], ref_img: nib.Nifti1Image) -> tuple[list[ROIGroup], nib.Nifti1Image, int]:
    data = np.zeros(ref_img.shape[:3], dtype=np.int16)
    out_groups: list[ROIGroup] = []
    n_overlap = 0
    for idx, group in enumerate(groups, start=1):
        overlap = group.mask & (data != 0)
        write_mask = group.mask & (data == 0)
        n_overlap += int(np.count_nonzero(overlap))
        data[write_mask] = idx
        out_groups.append(
            ROIGroup(
                name=group.name,
                source=group.source,
                mask=write_mask,
                matched_labels=group.matched_labels,
            )
        )
    return out_groups, nib.Nifti1Image(data, ref_img.affine, ref_img.header), n_overlap


def _compute_rows(
    groups: list[ROIGroup],
    selected_ijk: np.ndarray,
    active_mask: np.ndarray,
    exclusion_mask: np.ndarray,
) -> tuple[list[dict], dict]:
    x, y, z = selected_ijk.T
    excluded_selected = exclusion_mask[x, y, z]
    active_valid = active_mask & ~excluded_selected
    n_selected_total = int(selected_ijk.shape[0])
    n_active_total = int(np.count_nonzero(active_mask))
    n_active_valid = int(np.count_nonzero(active_valid))

    rows: list[dict] = []
    covered_active = np.zeros(selected_ijk.shape[0], dtype=bool)
    for idx, group in enumerate(groups, start=1):
        selected_members = group.mask[x, y, z]
        active_members = selected_members & active_mask
        active_valid_members = selected_members & active_valid
        covered_active |= active_members
        n_active = int(np.count_nonzero(active_members))
        n_valid_active = int(np.count_nonzero(active_valid_members))
        rows.append(
            {
                "roi_id": idx,
                "roi_name": group.name,
                "n_template_voxels_after_exclusion": int(np.count_nonzero(group.mask)),
                "n_selected_voxels": int(np.count_nonzero(selected_members)),
                "pct_selected_voxels": float(np.count_nonzero(selected_members) / n_selected_total * 100.0)
                if n_selected_total
                else 0.0,
                "n_active_voxels": n_active,
                "pct_active_voxels": float(n_active / n_active_total * 100.0) if n_active_total else 0.0,
                "n_valid_active_voxels": n_valid_active,
                "pct_valid_active_voxels": float(n_valid_active / n_active_valid * 100.0)
                if n_active_valid
                else 0.0,
                "atlas_source": group.source,
                "matched_labels": "; ".join(group.matched_labels),
            }
        )

    covered_valid_active = covered_active & active_valid
    stats = {
        "selected_voxel_count_total": n_selected_total,
        "active_voxel_count_total": n_active_total,
        "active_valid_voxel_count_after_exclusion": n_active_valid,
        "active_valid_voxels_covered_by_atlas": int(np.count_nonzero(covered_valid_active)),
        "active_valid_voxels_unassigned": int(np.count_nonzero(active_valid & ~covered_active)),
        "pct_active_valid_voxels_covered_by_atlas": float(np.mean(covered_valid_active[active_valid]) * 100.0)
        if n_active_valid
        else 0.0,
        "active_voxels_excluded_from_reportable_atlas": int(np.count_nonzero(active_mask & excluded_selected)),
        "pct_active_voxels_excluded_from_reportable_atlas": float(np.mean(excluded_selected[active_mask]) * 100.0)
        if n_active_total
        else 0.0,
    }
    return rows, stats


def _write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "roi_id",
        "roi_name",
        "n_template_voxels_after_exclusion",
        "n_selected_voxels",
        "pct_selected_voxels",
        "n_active_voxels",
        "pct_active_voxels",
        "n_valid_active_voxels",
        "pct_valid_active_voxels",
        "atlas_source",
        "matched_labels",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_counts(rows: list[dict], out_png: Path) -> None:
    plot_rows = [row for row in rows if int(row["n_active_voxels"]) > 0]
    labels = [str(row["roi_name"]) for row in plot_rows]
    counts = np.asarray([int(row["n_active_voxels"]) for row in plot_rows], dtype=np.int64)
    order = np.argsort(counts)
    labels = [labels[idx] for idx in order]
    counts = counts[order]

    fig_h = max(8.0, min(34.0, 0.17 * len(labels) + 2.0))
    fig, ax = plt.subplots(figsize=(12.5, fig_h))
    ax.barh(np.arange(len(labels)), counts, color="#2f6f9f")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Active selected voxels")
    ax.set_title("Schaefer + Cerebellar + Subcortical ROI Active-Voxel Counts")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    anat_path = args.anat_path.expanduser().resolve()
    voxel_indices_path = args.voxel_indices_path.expanduser().resolve()
    active_beta_file = args.active_beta_file.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    cache_dir = out_dir / "atlas_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    ref_img = nib.load(str(anat_path))
    selected_ijk = _load_selected_ijk(voxel_indices_path, ref_img.shape[:3])
    active_mask, active_source = _load_active_mask(active_beta_file, selected_ijk.shape[0], args.active_row_chunk)

    ho_sub_path = Path(HO_SUB_TEMPLATE.format(thr=int(args.harvard_oxford_threshold)))
    ho_sub_data = _load_label_img(ho_sub_path, ref_img)
    exclusion_mask, exclusion_qc = _build_exclusion_masks(
        ref_img=ref_img,
        selected_ijk=selected_ijk,
        white_prior_threshold=float(args.white_prior_threshold),
        ho_sub_data=ho_sub_data,
    )

    groups, source_metadata = _groups_from_sources(
        ref_img=ref_img,
        selected_ijk=selected_ijk,
        exclusion_mask=exclusion_mask,
        schaefer_n_rois=int(args.schaefer_n_rois),
        schaefer_yeo_networks=int(args.schaefer_yeo_networks),
        cerebellum_threshold=int(args.cerebellum_threshold),
        ho_threshold=int(args.harvard_oxford_threshold),
        cache_dir=cache_dir,
    )
    groups, roi_img, n_overlap = _assign_disjoint(groups, ref_img)
    rows, coverage_stats = _compute_rows(
        groups=groups,
        selected_ijk=selected_ijk,
        active_mask=active_mask,
        exclusion_mask=exclusion_mask,
    )

    prefix = f"schaefer{int(args.schaefer_n_rois)}_cereb_subcort"
    roi_img_path = out_dir / f"{prefix}_rois_fitted.nii.gz"
    stats_csv = out_dir / f"{prefix}_roi_stats.csv"
    summary_json = out_dir / f"{prefix}_roi_summary.json"
    qc_json = out_dir / f"{prefix}_voxel_qc.json"
    counts_png = out_dir / f"{prefix}_roi_voxel_counts.png"

    nib.save(roi_img, str(roi_img_path))
    _write_rows(stats_csv, rows)
    _plot_counts(rows, counts_png)

    x, y, z = selected_ijk.T
    roi_data = roi_img.get_fdata().astype(np.int16)
    labeled_selected = roi_data[x, y, z] > 0
    labeled_excluded = exclusion_mask[x, y, z] & labeled_selected
    qc = {
        **exclusion_qc,
        "n_labeled_selected_voxels_overlapping_exclusion_union": int(np.count_nonzero(labeled_excluded)),
        "pct_labeled_selected_voxels_overlapping_exclusion_union": float(np.mean(labeled_excluded) * 100.0)
        if labeled_excluded.size
        else 0.0,
        "n_overlap_voxels_resolved_by_priority_order": int(n_overlap),
    }
    qc_json.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    summary = {
        "atlas_name": f"Schaefer2018-{int(args.schaefer_n_rois)}_FSL-cerebellum_HO-subcortical_MNI2mm",
        "anat_path": str(anat_path),
        "voxel_indices_path": str(voxel_indices_path),
        "active_beta_file": str(active_beta_file),
        "active_mask_source": active_source,
        "roi_img": str(roi_img_path),
        "roi_stats_csv": str(stats_csv),
        "voxel_qc_json": str(qc_json),
        "all_roi_names_in_order": [row["roi_name"] for row in rows],
        "n_requested_rois": len(rows),
        "coverage": coverage_stats,
        "exclusion_qc": qc,
        "source_metadata": source_metadata,
        "roi_rows": rows,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Selected voxels: {coverage_stats['selected_voxel_count_total']}", flush=True)
    print(f"Active voxels: {coverage_stats['active_voxel_count_total']}", flush=True)
    print(
        "Excluded active voxels (WM/brainstem/ventricle/background): "
        f"{coverage_stats['active_voxels_excluded_from_reportable_atlas']} "
        f"({coverage_stats['pct_active_voxels_excluded_from_reportable_atlas']:.2f}%)",
        flush=True,
    )
    print(
        "Atlas coverage of valid active voxels: "
        f"{coverage_stats['active_valid_voxels_covered_by_atlas']}/"
        f"{coverage_stats['active_valid_voxel_count_after_exclusion']} "
        f"({coverage_stats['pct_active_valid_voxels_covered_by_atlas']:.2f}%)",
        flush=True,
    )
    print(f"Labeled selected voxels overlapping exclusion masks: {qc['n_labeled_selected_voxels_overlapping_exclusion_union']}", flush=True)
    print(f"Saved ROI image: {roi_img_path}", flush=True)
    print(f"Saved ROI table: {stats_csv}", flush=True)
    print(f"Saved voxel QC: {qc_json}", flush=True)
    print(f"Saved summary: {summary_json}", flush=True)


if __name__ == "__main__":
    main()
