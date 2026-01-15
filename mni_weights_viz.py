#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")

from nilearn import image, plotting
from scipy import ndimage


def _find_flirt():
    flirt_path = shutil.which("flirt")
    if flirt_path:
        return flirt_path
    fsl_dir = os.environ.get("FSLDIR")
    if not fsl_dir:
        return None
    candidate = Path(fsl_dir) / "bin" / "flirt"
    if candidate.exists():
        return str(candidate)
    return None


def _resolve_fsl_dir(flirt_path=None):
    if flirt_path:
        try:
            flirt_path = Path(flirt_path).resolve()
        except OSError:
            flirt_path = None
    if flirt_path:
        candidate = flirt_path.parent.parent
        if (candidate / "data" / "standard").exists():
            return candidate
    fsl_dir = os.environ.get("FSLDIR")
    if fsl_dir:
        return Path(fsl_dir)
    return None


def _default_mni_template(flirt_path=None):
    fsl_dir = _resolve_fsl_dir(flirt_path)
    if not fsl_dir:
        return None
    for name in ("MNI152_T1_2mm_brain.nii.gz", "MNI152_T1_1mm_brain.nii.gz"):
        candidate = fsl_dir / "data" / "standard" / name
        if candidate.exists():
            return candidate
    return None


def _run_flirt(cmd):
    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    subprocess.run(cmd, check=True, env=env)


def _compute_anat_to_mni(anat_path, mni_template, out_dir, flirt_path):
    out_dir = Path(out_dir)
    mat_path = out_dir / "anat_to_mni_flirt.mat"
    warped_path = out_dir / "anat_in_mni.nii.gz"
    cmd = [
        flirt_path,
        "-in",
        str(anat_path),
        "-ref",
        str(mni_template),
        "-omat",
        str(mat_path),
        "-out",
        str(warped_path),
        "-dof",
        "12",
        "-cost",
        "normmi",
        "-searchrx",
        "-90",
        "90",
        "-searchry",
        "-90",
        "90",
        "-searchrz",
        "-90",
        "90",
    ]
    _run_flirt(cmd)
    return mat_path, warped_path


def _apply_flirt(in_path, ref_path, mat_path, out_path, flirt_path, interp="trilinear"):
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


def _parse_list(value, item_type=float):
    items = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        items.append(item_type(item))
    return items


def _cluster_filter(data, threshold, min_cluster_size):
    thresholded = data > threshold
    labeled, num_features = ndimage.label(thresholded)
    if num_features == 0:
        return data * 0, 0, 0
    cluster_sizes = np.bincount(labeled.ravel())
    small_clusters = cluster_sizes < min_cluster_size
    keep_mask = thresholded & ~small_clusters[labeled]
    result = np.zeros_like(data)
    result[keep_mask] = data[keep_mask]
    kept_clusters = num_features - int(np.sum(small_clusters[1:]))
    total_kept_voxels = int(np.sum(keep_mask))
    return result, kept_clusters, total_kept_voxels


def _ensure_exists(path, label):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _sanitize_nifti(path, out_dir, label):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    nonfinite = ~np.isfinite(data)
    if np.any(nonfinite):
        cleaned = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        out_path = Path(out_dir) / f"{label}_finite.nii.gz"
        out_img = nib.Nifti1Image(cleaned.astype(np.float32, copy=False), img.affine, img.header)
        out_img.to_filename(str(out_path))
        return out_path, out_img, int(np.sum(nonfinite))
    return path, img, 0


def _build_output_dir(weights_path, out_dir):
    if out_dir:
        return Path(out_dir)
    weights_base = weights_path.name
    if weights_base.endswith(".nii.gz"):
        weights_base = weights_base[:-7]
    return weights_path.parent / f"mni_viz_{weights_base}"


def main():
    parser = argparse.ArgumentParser(
        description="Register weights and anatomy to MNI space and generate thresholded visualizations."
    )
    parser.add_argument("--weights", required=True, type=Path, help="Weights NIfTI in anatomical space.")
    parser.add_argument("--anat", required=True, type=Path, help="Anatomical T1w brain NIfTI.")
    parser.add_argument("--out-dir", type=Path, help="Output directory (default: mni_viz_<weights>).")
    parser.add_argument("--mni-template", type=Path, help="MNI template NIfTI (optional).")
    parser.add_argument("--assume-mni", action="store_true", help="Skip registration and assume inputs are already in MNI space.")
    parser.add_argument(
        "--threshold-space",
        choices=("mni", "native"),
        default="mni",
        help="Compute thresholds/cluster filtering in MNI space (default) or native space before warping.",
    )
    parser.add_argument("--fwhm", type=float, default=0.0, help="Gaussian smoothing FWHM in mm (0 disables smoothing).")
    parser.add_argument("--percentiles", default="90,95,99", help="Comma-separated percentiles for thresholds (default: 90,95,99).")
    parser.add_argument("--cluster-sizes", default="100,75,50", help="Comma-separated minimum cluster sizes per threshold (default: 100,75,50).")
    parser.add_argument("--vmax-percentile", type=float, default=99.9, help="Percentile for vmax scaling (default: 99.9).")
    args = parser.parse_args()

    weights_path = args.weights.expanduser().resolve()
    anat_path = args.anat.expanduser().resolve()
    _ensure_exists(weights_path, "Weights file")
    _ensure_exists(anat_path, "Anatomy file")

    out_dir = _build_output_dir(weights_path, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path, weights_img, nonfinite_count = _sanitize_nifti(weights_path, out_dir, "weights_input")
    if nonfinite_count:
        print(f"Replaced {nonfinite_count} non-finite voxels with 0 in weights (saved to {weights_path}).")

    percentiles = _parse_list(args.percentiles, float)
    cluster_sizes = _parse_list(args.cluster_sizes, int)
    if not percentiles:
        raise ValueError("No percentiles specified.")
    if len(cluster_sizes) != len(percentiles):
        raise ValueError("cluster-sizes must have the same length as percentiles.")

    mni_template = None
    if args.mni_template:
        mni_template = args.mni_template.expanduser().resolve()
        _ensure_exists(mni_template, "MNI template")

    threshold_space = args.threshold_space.lower()
    if args.assume_mni and threshold_space == "native":
        threshold_space = "mni"
    flirt_path = None
    mat_path = None
    registration_method = "assume-mni"
    if args.assume_mni:
        weights_mni_img = weights_img
        anat_mni_img = nib.load(str(anat_path))
        if mni_template:
            mni_img = nib.load(str(mni_template))
            weights_mni_img = image.resample_to_img(weights_mni_img, mni_img, interpolation="continuous", force_resample=True, copy_header=True)
            anat_mni_img = image.resample_to_img(anat_mni_img, mni_img, interpolation="continuous", force_resample=True, copy_header=True)
            weights_mni_path = out_dir / "weights_in_mni_resampled.nii.gz"
            anat_mni_path = out_dir / "anat_in_mni_resampled.nii.gz"
            weights_mni_img.to_filename(str(weights_mni_path))
            anat_mni_img.to_filename(str(anat_mni_path))
            registration_method = "assume-mni-resampled"
    else:
        flirt_path = _find_flirt()
        if not flirt_path:
            raise RuntimeError("FSL flirt not found in PATH. Install FSL or use --assume-mni.")
        if mni_template is None:
            mni_template = _default_mni_template(flirt_path)
        if mni_template is None or not mni_template.exists():
            raise RuntimeError("Could not resolve MNI template. Provide --mni-template or set FSLDIR.")
        mat_path, anat_mni_path = _compute_anat_to_mni(anat_path, mni_template, out_dir, flirt_path)
        weights_mni_path = out_dir / "weights_in_mni.nii.gz"
        _apply_flirt(weights_path, mni_template, mat_path, weights_mni_path, flirt_path, interp="trilinear")
        anat_mni_img = nib.load(str(anat_mni_path))
        weights_mni_img = nib.load(str(weights_mni_path))
        registration_method = "flirt"

    print(f"Registration method: {registration_method}")
    print(f"Output directory: {out_dir}")
    print(f"Threshold space: {threshold_space}")

    if threshold_space == "native":
        processed_img = weights_img
        if args.fwhm and args.fwhm > 0:
            processed_img = image.smooth_img(weights_img, fwhm=args.fwhm)
            smoothed_path = out_dir / f"weights_native_smoothed_fwhm{args.fwhm:g}.nii.gz"
            processed_img.to_filename(str(smoothed_path))
            print(f"Saved smoothed native weights: {smoothed_path}")
        if not args.assume_mni:
            print("Warping native-thresholded maps to MNI with nearest-neighbour interpolation.")
    else:
        processed_img = weights_mni_img
        if args.fwhm and args.fwhm > 0:
            processed_img = image.smooth_img(weights_mni_img, fwhm=args.fwhm)
            smoothed_path = out_dir / f"weights_in_mni_smoothed_fwhm{args.fwhm:g}.nii.gz"
            processed_img.to_filename(str(smoothed_path))
            print(f"Saved smoothed weights: {smoothed_path}")

    data = processed_img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    finite_vals = data[data > 0]
    if finite_vals.size == 0:
        raise RuntimeError("No finite positive values found in weights data.")

    vmax = float(np.percentile(finite_vals, args.vmax_percentile))
    thresholds = [float(np.percentile(finite_vals, pct)) for pct in percentiles]

    print("Thresholds:")
    for pct, thr, min_size in zip(percentiles, thresholds, cluster_sizes):
        print(f"  {pct:g}th percentile: {thr:.6g} (min_cluster={min_size})")

    for pct, thr, min_size in zip(percentiles, thresholds, cluster_sizes):
        filtered, clusters, voxels = _cluster_filter(data, thr, min_size)
        print(f"  {pct:g}th percentile: {clusters} clusters, {voxels} voxels kept")

        out_base = out_dir / f"weights_mni_thr{int(pct)}"
        if threshold_space == "native":
            native_base = out_dir / f"weights_native_thr{int(pct)}"
            native_img = nib.Nifti1Image(filtered.astype(np.float32), processed_img.affine, processed_img.header)
            native_img.to_filename(str(native_base.with_suffix(".nii.gz")))
            if args.assume_mni:
                native_img.to_filename(str(out_base.with_suffix(".nii.gz")))
                filtered_img = native_img
            else:
                if flirt_path is None or mat_path is None:
                    raise RuntimeError("Missing FLIRT registration for native thresholding.")
                _apply_flirt(native_base.with_suffix(".nii.gz"), mni_template, mat_path, out_base.with_suffix(".nii.gz"), flirt_path, interp="nearestneighbour")
                filtered_img = nib.load(str(out_base.with_suffix(".nii.gz")))
        else:
            filtered_img = nib.Nifti1Image(filtered.astype(np.float32), processed_img.affine, processed_img.header)
            filtered_img.to_filename(str(out_base.with_suffix(".nii.gz")))

        view = plotting.view_img(filtered_img, bg_img=anat_mni_img, cmap="hot", symmetric_cmap=False, threshold=thr, vmax=vmax, colorbar=True, title=f"Weights in MNI (p{int(pct)}, clusters={clusters}, min_size={min_size})")
        view.save_as_html(str(out_base.with_suffix(".html")))

        display = plotting.plot_stat_map(filtered_img, bg_img=anat_mni_img, cmap="hot", symmetric_cbar=False, threshold=thr, vmax=vmax, colorbar=True,
            title=f"Weights in MNI (p{int(pct)}, clusters={clusters})", display_mode="ortho")
        display.savefig(str(out_base.with_suffix(".png")), dpi=150)
        display.close()

    print("Done.")


if __name__ == "__main__":
    main()
