#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import datasets, image, plotting


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Create atlas-based and anatomy-based motor ROI masks and overlays."
    )
    parser.add_argument(
        "--anat",
        type=Path,
        default=Path("/home/zkavian/Thesis_code_Glm_Opt/GLMsingle/sub-pd009_ses-1_T1w_brain.nii.gz"),
        help="Path to subject anatomy (T1w brain).",
    )
    parser.add_argument(
        "--gray",
        type=Path,
        default=Path("/home/zkavian/Thesis_code_Glm_Opt/GLMsingle/sub-pd009_ses-1_T1w_brain_pve_1.nii.gz"),
        help="Path to gray-matter PVE map.",
    )
    parser.add_argument(
        "--csf",
        type=Path,
        default=Path("/home/zkavian/Thesis_code_Glm_Opt/GLMsingle/sub-pd009_ses-1_T1w_brain_pve_0.nii.gz"),
        help="Path to CSF PVE map (optional for brain+CSF overlay).",
    )
    parser.add_argument(
        "--brain-mask",
        type=Path,
        default=Path("/home/zkavian/Thesis_code_Glm_Opt/GLMsingle/sub-pd009_ses-1_T1w_brain_mask.nii.gz"),
        help="Path to brain mask (optional for anatomy-based ROI).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/home/zkavian/Thesis_code_Glm_Opt/analysis_motor_cortex/roi_labels"),
        help="Output directory for ROI masks and overlays.",
    )
    parser.add_argument(
        "--atlas-threshold",
        type=int,
        default=25,
        help="Harvard-Oxford maxprob atlas threshold (percent).",
    )
    parser.add_argument(
        "--atlas-label-patterns",
        default="Precentral Gyrus",
        help="Comma-separated label patterns to include (case-insensitive).",
    )
    parser.add_argument(
        "--mni-template",
        type=Path,
        default=None,
        help="Optional MNI template to use for atlas registration.",
    )
    parser.add_argument(
        "--assume-mni",
        action="store_true",
        help="Assume anatomy is in MNI space for atlas alignment.",
    )
    parser.add_argument(
        "--left-coord",
        type=float,
        nargs=3,
        default=(-38.0, -20.0, 56.0),
        help="Left motor cortex coord in mm (x y z).",
    )
    parser.add_argument(
        "--right-coord",
        type=float,
        nargs=3,
        default=(38.0, -20.0, 56.0),
        help="Right motor cortex coord in mm (x y z).",
    )
    parser.add_argument(
        "--sphere-radius-mm",
        type=float,
        default=6.0,
        help="Radius of anatomy-based spherical ROI (mm).",
    )
    parser.add_argument(
        "--gm-threshold",
        type=float,
        default=0.3,
        help="Gray-matter threshold for anatomy-based ROI.",
    )
    parser.add_argument(
        "--csf-threshold",
        type=float,
        default=0.5,
        help="CSF threshold for brain+CSF overlay.",
    )
    parser.add_argument(
        "--cut-coords",
        type=float,
        nargs=3,
        default=None,
        help="Optional cut coordinates for overlays (x y z).",
    )
    return parser.parse_args()


def _voxel_sizes(affine):
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def _sphere_mask(shape, affine, center_mm, radius_mm):
    center_mm = np.asarray(center_mm, dtype=float)
    inv_affine = np.linalg.inv(affine)
    center_vox = inv_affine @ np.append(center_mm, 1.0)
    center_vox = center_vox[:3]

    sizes = _voxel_sizes(affine)
    rad_vox = radius_mm / np.maximum(sizes, 1e-6)

    x, y, z = np.indices(shape)
    dx = (x - center_vox[0]) / rad_vox[0]
    dy = (y - center_vox[1]) / rad_vox[1]
    dz = (z - center_vox[2]) / rad_vox[2]
    return (dx * dx + dy * dy + dz * dz) <= 1.0


def _find_flirt():
    return shutil.which("flirt")


def _default_mni_template():
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        return None
    fsl_dir = Path(fsl_dir)
    for name in ("MNI152_T1_2mm_brain.nii.gz", "MNI152_T1_1mm_brain.nii.gz"):
        candidate = fsl_dir / "data" / "standard" / name
        if candidate.exists():
            return candidate
    return None


def _run_flirt(cmd):
    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    subprocess.run(cmd, check=True, env=env)


def _compute_mni_to_anat(mni_template, anat_path, out_dir, flirt_path):
    out_dir = Path(out_dir)
    mat_path = out_dir / "mni_to_anat_flirt.mat"
    warped_path = out_dir / "mni_template_in_anat.nii.gz"
    cmd = [
        flirt_path,
        "-in",
        str(mni_template),
        "-ref",
        str(anat_path),
        "-omat",
        str(mat_path),
        "-out",
        str(warped_path),
        "-dof",
        "12",
    ]
    _run_flirt(cmd)
    return mat_path, warped_path


def _apply_flirt(in_path, ref_path, mat_path, out_path, flirt_path, interp="nearestneighbour"):
    cmd = [
        flirt_path,
        "-in",
        str(in_path),
        "-ref",
        str(ref_path),
        "-applyxfm",
        "-init",
        str(mat_path),
        "-interp",
        interp,
        "-out",
        str(out_path),
    ]
    _run_flirt(cmd)


def _select_label_indices(labels, patterns, side=None):
    indices = []
    for idx, name in enumerate(labels):
        lname = name.lower()
        if side is not None and side not in lname:
            continue
        if any(pattern in lname for pattern in patterns):
            indices.append(idx)
    return indices


def _labels_have_sides(labels):
    for name in labels:
        lname = name.lower()
        if "left" in lname or "right" in lname:
            return True
    return False


def _split_mask_by_hemisphere(mask, affine):
    if not np.any(mask):
        empty = np.zeros(mask.shape, dtype=np.uint8)
        return empty, empty

    ijk = np.column_stack(np.where(mask))
    xyz = nib.affines.apply_affine(affine, ijk)
    left_sel = xyz[:, 0] < 0
    right_sel = xyz[:, 0] > 0
    mid_sel = ~(left_sel | right_sel)

    left_mask = np.zeros(mask.shape, dtype=np.uint8)
    right_mask = np.zeros(mask.shape, dtype=np.uint8)
    left_mask[tuple(ijk[left_sel].T)] = 1
    right_mask[tuple(ijk[right_sel].T)] = 1
    if np.any(mid_sel):
        left_mask[tuple(ijk[mid_sel].T)] = 1
        right_mask[tuple(ijk[mid_sel].T)] = 1

    return left_mask, right_mask


def _save_overlays(mask_img, anat_img, title, out_html, out_png, cut_coords=None):
    view = plotting.view_img(
        mask_img,
        bg_img=anat_img,
        threshold=0.5,
        cmap="autumn",
        colorbar=True,
        title=title,
        cut_coords=cut_coords,
        resampling_interpolation="nearest",
    )
    view.save_as_html(str(out_html))

    display = plotting.plot_roi(
        mask_img,
        bg_img=anat_img,
        cmap="autumn",
        title=title,
        cut_coords=cut_coords,
    )
    display.savefig(str(out_png))
    display.close()


def _save_layered_overlays(
    base_mask_img,
    roi_mask_img,
    anat_img,
    title,
    out_html,
    out_png,
    cut_coords=None,
    base_color="#fff2a8",
    roi_color="#f4a340",
):
    base_data = base_mask_img.get_fdata(dtype=np.float32) > 0.5 if base_mask_img is not None else None
    roi_data = roi_mask_img.get_fdata(dtype=np.float32) > 0.5 if roi_mask_img is not None else None

    if base_data is None and roi_data is None:
        return

    composite = np.zeros(anat_img.shape[:3], dtype=np.uint8)
    if base_data is not None:
        composite[base_data] = 1
    if roi_data is not None:
        composite[roi_data] = 2

    composite_img = nib.Nifti1Image(composite, anat_img.affine, anat_img.header)

    bg_img = anat_img
    if base_data is not None:
        masked_data = anat_img.get_fdata(dtype=np.float32)
        masked_data = masked_data.copy()
        masked_data[~base_data] = 0
        bg_img = nib.Nifti1Image(masked_data, anat_img.affine, anat_img.header)

    cmap = ListedColormap([base_color, roi_color])
    view = plotting.view_img(
        composite_img,
        bg_img=bg_img,
        threshold=0.5,
        cmap=cmap,
        colorbar=True,
        title=title,
        cut_coords=cut_coords,
        resampling_interpolation="nearest",
        vmax=2,
    )
    view.save_as_html(str(out_html))

    display = plotting.plot_anat(
        bg_img,
        title=title,
        cut_coords=cut_coords,
    )
    if base_data is not None:
        base_only = base_data.copy()
        if roi_data is not None:
            base_only &= ~roi_data
        base_only_img = nib.Nifti1Image(base_only.astype(np.uint8), anat_img.affine, anat_img.header)
        display.add_overlay(
            base_only_img,
            cmap=ListedColormap([base_color]),
            alpha=0.35,
            threshold=0.5,
        )
    if roi_data is not None:
        roi_img = nib.Nifti1Image(roi_data.astype(np.uint8), anat_img.affine, anat_img.header)
        display.add_overlay(
            roi_img,
            cmap=ListedColormap([roi_color]),
            alpha=0.8,
            threshold=0.5,
        )
    display.savefig(str(out_png))
    display.close()


def _maybe_resample_mask(mask_img, ref_img, threshold):
    if mask_img is None:
        return None
    if mask_img.shape[:3] != ref_img.shape[:3] or not np.allclose(mask_img.affine, ref_img.affine):
        mask_img = image.resample_to_img(
            mask_img,
            ref_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
    return mask_img.get_fdata(dtype=np.float32) > threshold


def _pct(part, total):
    return float(part) / float(total) * 100 if total else 0.0


def _mask_overlap_stats(mask, brain_mask=None, csf_mask=None):
    total = int(np.count_nonzero(mask))
    stats = {"total_voxels": total}
    if brain_mask is not None:
        in_brain = int(np.count_nonzero(mask & brain_mask))
        stats["in_brain"] = in_brain
        stats["pct_in_brain"] = _pct(in_brain, total)
        stats["outside_brain"] = total - in_brain
        stats["pct_outside_brain"] = _pct(total - in_brain, total)
    if csf_mask is not None:
        in_csf = int(np.count_nonzero(mask & csf_mask))
        stats["in_csf"] = in_csf
        stats["pct_in_csf"] = _pct(in_csf, total)
    if brain_mask is not None and csf_mask is not None:
        in_brain_not_csf = int(np.count_nonzero(mask & brain_mask & ~csf_mask))
        stats["in_brain_not_csf"] = in_brain_not_csf
        stats["pct_in_brain_not_csf"] = _pct(in_brain_not_csf, total)
    return stats


def main():
    args = _parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    anat_img = nib.load(str(args.anat))
    gray_img = nib.load(str(args.gray)) if args.gray else None
    brain_img = nib.load(str(args.brain_mask)) if args.brain_mask else None
    csf_img = nib.load(str(args.csf)) if args.csf else None

    flirt_path = None
    use_flirt = False
    mat_path = None
    warped_template_path = None
    mni_template = None
    mni_template_img = None
    registration_info = {"method": "resample_to_img", "assume_mni": bool(args.assume_mni)}

    if not args.assume_mni:
        flirt_path = _find_flirt()
        mni_template = args.mni_template.expanduser().resolve() if args.mni_template else _default_mni_template()
        if args.mni_template and (mni_template is None or not mni_template.exists()):
            print(
                f"Warning: MNI template not found at {args.mni_template}; "
                "falling back to header-based resampling.",
                flush=True,
            )
            mni_template = None
        if flirt_path and mni_template and mni_template.exists():
            try:
                mat_path, warped_template_path = _compute_mni_to_anat(
                    mni_template, args.anat, out_dir, flirt_path
                )
                use_flirt = True
                registration_info = {
                    "method": "flirt",
                    "assume_mni": False,
                    "flirt_path": flirt_path,
                    "mni_template": str(mni_template),
                    "matrix": str(mat_path),
                    "warped_template": str(warped_template_path),
                }
                print("Registered MNI template to anatomy with FLIRT.", flush=True)
            except subprocess.CalledProcessError:
                print(
                    "Warning: FLIRT registration failed; falling back to header-based resampling.",
                    flush=True,
                )
        else:
            print(
                "Warning: FLIRT or MNI template not available; falling back to header-based resampling.",
                flush=True,
            )

    atlas = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr{thr}-2mm".format(thr=args.atlas_threshold)
    )
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    labels = list(atlas.labels)

    patterns = [p.strip().lower() for p in args.atlas_label_patterns.split(",") if p.strip()]
    atlas_data = atlas_img.get_fdata().astype(int)
    if _labels_have_sides(labels):
        left_indices = _select_label_indices(labels, patterns, side="left")
        right_indices = _select_label_indices(labels, patterns, side="right")
        combined_indices = sorted(set(left_indices + right_indices))
        atlas_left = np.isin(atlas_data, left_indices)
        atlas_right = np.isin(atlas_data, right_indices)
        atlas_combined = np.isin(atlas_data, combined_indices)
        hemisphere_split = "label"
    else:
        combined_indices = _select_label_indices(labels, patterns, side=None)
        atlas_combined = np.isin(atlas_data, combined_indices)
        atlas_left, atlas_right = _split_mask_by_hemisphere(atlas_combined, atlas_img.affine)
        left_indices = combined_indices
        right_indices = combined_indices
        hemisphere_split = "mni-x"

    atlas_left = atlas_left.astype(np.uint8)
    atlas_right = atlas_right.astype(np.uint8)
    atlas_combined = atlas_combined.astype(np.uint8)

    atlas_left_img = nib.Nifti1Image(atlas_left, atlas_img.affine, atlas_img.header)
    atlas_right_img = nib.Nifti1Image(atlas_right, atlas_img.affine, atlas_img.header)
    atlas_combined_img = nib.Nifti1Image(atlas_combined, atlas_img.affine, atlas_img.header)

    atlas_left_path = out_dir / "atlas_motor_left.nii.gz"
    atlas_right_path = out_dir / "atlas_motor_right.nii.gz"
    atlas_combined_path = out_dir / "atlas_motor_combined.nii.gz"
    if use_flirt:
        if mni_template_img is None:
            mni_template_img = nib.load(str(mni_template))
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            tmpdir = Path(tmpdir)
            atlas_left_mni = tmpdir / "atlas_left_mni.nii.gz"
            atlas_right_mni = tmpdir / "atlas_right_mni.nii.gz"
            atlas_combined_mni = tmpdir / "atlas_combined_mni.nii.gz"
            atlas_left_mni_img = image.resample_to_img(
                atlas_left_img,
                mni_template_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            atlas_right_mni_img = image.resample_to_img(
                atlas_right_img,
                mni_template_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            atlas_combined_mni_img = image.resample_to_img(
                atlas_combined_img,
                mni_template_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            atlas_left_mni_img.to_filename(str(atlas_left_mni))
            atlas_right_mni_img.to_filename(str(atlas_right_mni))
            atlas_combined_mni_img.to_filename(str(atlas_combined_mni))

            _apply_flirt(atlas_left_mni, args.anat, mat_path, atlas_left_path, flirt_path)
            _apply_flirt(atlas_right_mni, args.anat, mat_path, atlas_right_path, flirt_path)
            _apply_flirt(atlas_combined_mni, args.anat, mat_path, atlas_combined_path, flirt_path)
        atlas_left_resampled = nib.load(str(atlas_left_path))
        atlas_right_resampled = nib.load(str(atlas_right_path))
        atlas_combined_resampled = nib.load(str(atlas_combined_path))
    else:
        atlas_left_resampled = image.resample_to_img(
            atlas_left_img,
            anat_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_right_resampled = image.resample_to_img(
            atlas_right_img,
            anat_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_combined_resampled = image.resample_to_img(
            atlas_combined_img,
            anat_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        atlas_left_resampled.to_filename(str(atlas_left_path))
        atlas_right_resampled.to_filename(str(atlas_right_path))
        atlas_combined_resampled.to_filename(str(atlas_combined_path))

    atlas_left_mask = atlas_left_resampled.get_fdata(dtype=np.float32) > 0.5
    atlas_right_mask = atlas_right_resampled.get_fdata(dtype=np.float32) > 0.5
    atlas_combined_mask = atlas_combined_resampled.get_fdata(dtype=np.float32) > 0.5

    # Anatomy-based spherical ROIs restricted to gray matter (and brain mask if provided).
    shape = anat_img.shape[:3]
    gm_mask = None
    if gray_img is not None:
        gm_data = gray_img.get_fdata(dtype=np.float32)
        gm_mask = gm_data > args.gm_threshold
    brain_mask = _maybe_resample_mask(brain_img, anat_img, threshold=0) if brain_img is not None else None
    csf_mask = (
        _maybe_resample_mask(csf_img, anat_img, threshold=args.csf_threshold) if csf_img is not None else None
    )

    if use_flirt:
        if mni_template_img is None:
            mni_template_img = nib.load(str(mni_template))
        left_sphere_mni = _sphere_mask(
            mni_template_img.shape[:3], mni_template_img.affine, args.left_coord, args.sphere_radius_mm
        )
        right_sphere_mni = _sphere_mask(
            mni_template_img.shape[:3], mni_template_img.affine, args.right_coord, args.sphere_radius_mm
        )
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            tmpdir = Path(tmpdir)
            left_sphere_mni_path = tmpdir / "left_sphere_mni.nii.gz"
            right_sphere_mni_path = tmpdir / "right_sphere_mni.nii.gz"
            left_sphere_path = tmpdir / "left_sphere_in_anat.nii.gz"
            right_sphere_path = tmpdir / "right_sphere_in_anat.nii.gz"
            nib.Nifti1Image(
                left_sphere_mni.astype(np.uint8),
                mni_template_img.affine,
                mni_template_img.header,
            ).to_filename(str(left_sphere_mni_path))
            nib.Nifti1Image(
                right_sphere_mni.astype(np.uint8),
                mni_template_img.affine,
                mni_template_img.header,
            ).to_filename(str(right_sphere_mni_path))
            _apply_flirt(left_sphere_mni_path, args.anat, mat_path, left_sphere_path, flirt_path)
            _apply_flirt(right_sphere_mni_path, args.anat, mat_path, right_sphere_path, flirt_path)
            left_sphere = nib.load(str(left_sphere_path)).get_fdata() > 0.5
            right_sphere = nib.load(str(right_sphere_path)).get_fdata() > 0.5
    else:
        left_sphere = _sphere_mask(shape, anat_img.affine, args.left_coord, args.sphere_radius_mm)
        right_sphere = _sphere_mask(shape, anat_img.affine, args.right_coord, args.sphere_radius_mm)

    if gm_mask is not None:
        left_sphere &= gm_mask
        right_sphere &= gm_mask
    if brain_mask is not None:
        left_sphere &= brain_mask
        right_sphere &= brain_mask

    anatomy_left = left_sphere.astype(np.uint8)
    anatomy_right = right_sphere.astype(np.uint8)
    anatomy_combined = np.clip(anatomy_left + anatomy_right, 0, 1).astype(np.uint8)

    anatomy_left_img = nib.Nifti1Image(anatomy_left, anat_img.affine, anat_img.header)
    anatomy_right_img = nib.Nifti1Image(anatomy_right, anat_img.affine, anat_img.header)
    anatomy_combined_img = nib.Nifti1Image(anatomy_combined, anat_img.affine, anat_img.header)

    anatomy_left_path = out_dir / "anatomy_motor_left.nii.gz"
    anatomy_right_path = out_dir / "anatomy_motor_right.nii.gz"
    anatomy_combined_path = out_dir / "anatomy_motor_combined.nii.gz"
    anatomy_left_img.to_filename(str(anatomy_left_path))
    anatomy_right_img.to_filename(str(anatomy_right_path))
    anatomy_combined_img.to_filename(str(anatomy_combined_path))

    cut_coords = tuple(args.cut_coords) if args.cut_coords else None
    brain_csf_mask_img = None
    brain_mask_img = None
    brain_not_csf_img = None
    atlas_in_brain_img = None
    atlas_in_brain_no_csf_img = None

    if brain_mask is not None:
        brain_mask_img = nib.Nifti1Image(brain_mask.astype(np.uint8), anat_img.affine, anat_img.header)
        atlas_in_brain = atlas_combined_mask & brain_mask
        atlas_in_brain_img = nib.Nifti1Image(atlas_in_brain.astype(np.uint8), anat_img.affine, anat_img.header)
        atlas_in_brain_path = out_dir / "atlas_motor_combined_in_brain.nii.gz"
        atlas_in_brain_img.to_filename(str(atlas_in_brain_path))

    if brain_mask is not None and csf_mask is not None:
        brain_not_csf = brain_mask & ~csf_mask
        brain_not_csf_img = nib.Nifti1Image(brain_not_csf.astype(np.uint8), anat_img.affine, anat_img.header)
        atlas_in_brain_no_csf = atlas_combined_mask & brain_not_csf
        atlas_in_brain_no_csf_img = nib.Nifti1Image(
            atlas_in_brain_no_csf.astype(np.uint8),
            anat_img.affine,
            anat_img.header,
        )
        atlas_in_brain_no_csf_path = out_dir / "atlas_motor_combined_in_brain_no_csf.nii.gz"
        atlas_in_brain_no_csf_img.to_filename(str(atlas_in_brain_no_csf_path))

    if brain_mask is not None or csf_mask is not None:
        brain_csf_mask = np.zeros(shape, dtype=bool)
        if brain_mask is not None:
            brain_csf_mask |= brain_mask
        if csf_mask is not None:
            brain_csf_mask |= csf_mask
        brain_csf_mask_img = nib.Nifti1Image(
            brain_csf_mask.astype(np.uint8),
            anat_img.affine,
            anat_img.header,
        )

    if atlas_in_brain_no_csf_img is not None:
        atlas_overlay_base = brain_not_csf_img
        atlas_overlay_img = atlas_in_brain_no_csf_img
        atlas_overlay_title = "Atlas-based Motor ROI (Brain Masked, CSF Removed)"
    elif atlas_in_brain_img is not None:
        atlas_overlay_base = brain_mask_img
        atlas_overlay_img = atlas_in_brain_img
        atlas_overlay_title = "Atlas-based Motor ROI (Brain Masked)"
    else:
        atlas_overlay_base = brain_csf_mask_img
        atlas_overlay_img = atlas_combined_resampled
        atlas_overlay_title = "Atlas-based Motor ROI (Precentral)"

    _save_layered_overlays(
        atlas_overlay_base,
        atlas_overlay_img,
        anat_img,
        atlas_overlay_title,
        out_dir / "atlas_motor_overlay.html",
        out_dir / "atlas_motor_overlay.png",
        cut_coords=cut_coords,
    )
    if brain_mask is not None:
        _save_layered_overlays(
            brain_mask_img,
            atlas_in_brain_img,
            anat_img,
            "Atlas-based Motor ROI (Brain Masked)",
            out_dir / "atlas_motor_overlay_in_brain.html",
            out_dir / "atlas_motor_overlay_in_brain.png",
            cut_coords=cut_coords,
        )
        if csf_mask is not None:
            _save_layered_overlays(
                brain_not_csf_img,
                atlas_in_brain_no_csf_img,
                anat_img,
                "Atlas-based Motor ROI (Brain Masked, CSF Removed)",
                out_dir / "atlas_motor_overlay_in_brain_no_csf.html",
                out_dir / "atlas_motor_overlay_in_brain_no_csf.png",
                cut_coords=cut_coords,
            )
    _save_overlays(
        anatomy_combined_img,
        anat_img,
        "Anatomy-based Motor ROI (Spheres)",
        out_dir / "anatomy_motor_overlay.html",
        out_dir / "anatomy_motor_overlay.png",
        cut_coords=cut_coords,
    )

    # display = plotting.plot_anat(
    #     anat_img,
    #     title="Anatomy + Motor Cortex ROI (Atlas)",
    #     cut_coords=cut_coords,
    # )
    # display.add_overlay(
    #     atlas_combined_resampled,
    #     cmap="Oranges",
    #     alpha=0.6,
    #     threshold=0.5,
    # )
    # display.savefig(str(out_dir / "anatomy_motor_atlas_overlay.png"))
    # display.close()

    summary = {
        "atlas": {
            "label_patterns": patterns,
            "left_indices": left_indices,
            "right_indices": right_indices,
            "left_labels": [labels[i] for i in left_indices],
            "right_labels": [labels[i] for i in right_indices],
            "hemisphere_split": hemisphere_split,
            "voxel_counts": {
                "left": int(np.count_nonzero(atlas_left_resampled.get_fdata())),
                "right": int(np.count_nonzero(atlas_right_resampled.get_fdata())),
                "combined": int(np.count_nonzero(atlas_combined_resampled.get_fdata())),
            },
        },
        "anatomy": {
            "left_coord_mm": list(args.left_coord),
            "right_coord_mm": list(args.right_coord),
            "sphere_radius_mm": float(args.sphere_radius_mm),
            "gm_threshold": float(args.gm_threshold),
            "voxel_counts": {
                "left": int(np.count_nonzero(anatomy_left)),
                "right": int(np.count_nonzero(anatomy_right)),
                "combined": int(np.count_nonzero(anatomy_combined)),
            },
        },
    }
    summary["registration"] = registration_info
    summary["masking_effects"] = {
        "brain_mask_available": brain_mask is not None,
        "csf_mask_available": csf_mask is not None,
        "atlas": {
            "left": _mask_overlap_stats(atlas_left_mask, brain_mask, csf_mask),
            "right": _mask_overlap_stats(atlas_right_mask, brain_mask, csf_mask),
            "combined": _mask_overlap_stats(atlas_combined_mask, brain_mask, csf_mask),
        },
    }
    with open(out_dir / "motor_roi_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved ROI outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
