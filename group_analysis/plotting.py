import argparse
import os

import nibabel as nib
import numpy as np
from nilearn import plotting

DEFAULT_BETA_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/cleaned_beta_volume_group.npy"
DEFAULT_MASK_PATH = "/Data/zahra/results_beta_preprocessed/group_concat/common_mask_group.npy"
DEFAULT_ANAT_PATH = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz"
DEFAULT_GRAY_MASK_PATH = "/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain_seg_gm.nii.gz"


def _compute_activation(beta, metric):
    if metric == "median_abs":
        activation = np.nanmedian(np.abs(beta), axis=1)
        label = "median(|beta|)"
    elif metric == "mean_abs":
        activation = np.abs(np.nanmean(beta, axis=1))
        label = "|mean(beta)|"
    elif metric == "effect_size_abs":
        mean_beta = np.nanmean(beta, axis=1)
        std_beta = np.nanstd(beta, axis=1)
        activation = np.divide(
            np.abs(mean_beta),
            np.maximum(std_beta, 1e-6),
            out=np.zeros_like(mean_beta, dtype=np.float32),
            where=np.isfinite(std_beta),
        )
        label = "|mean(beta)| / std(beta)"
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return activation.astype(np.float32, copy=False), label


def _motor_like_roi_mask(coords_mm):
    x = coords_mm[:, 0]
    y = coords_mm[:, 1]
    z = coords_mm[:, 2]
    return (
        (np.abs(x) <= 45.0)
        & (y >= -40.0)
        & (y <= -5.0)
        & (z >= 30.0)
        & (z <= 75.0)
    )


def _top_peak_report(vol, anat_affine, n=10):
    flat = np.argsort(vol.ravel())[::-1]
    rows = []
    for flat_idx in flat:
        value = float(vol.ravel()[flat_idx])
        if value <= 0:
            break
        ijk = np.unravel_index(flat_idx, vol.shape)
        xyz = nib.affines.apply_affine(anat_affine, np.array(ijk))
        rows.append((ijk, xyz, value))
        if len(rows) >= n:
            break
    return rows


def main():
    parser = argparse.ArgumentParser(description="Plot group pre-optimization activation map.")
    parser.add_argument("--beta-path", default=DEFAULT_BETA_PATH)
    parser.add_argument("--mask-path", default=DEFAULT_MASK_PATH)
    parser.add_argument("--anat-path", default=DEFAULT_ANAT_PATH)
    parser.add_argument("--gray-mask-path", default=DEFAULT_GRAY_MASK_PATH)
    parser.add_argument("--min-trial-coverage", type=float, default=0.0)
    parser.add_argument("--vmax-percentile", type=float, default=99.9)
    parser.add_argument("--top-percentile", type=float, default=50.0)
    parser.add_argument(
        "--metric",
        choices=["effect_size_abs", "median_abs", "mean_abs"],
        default=os.environ.get("FMRI_PLOT_METRIC", "effect_size_abs"),
    )
    args = parser.parse_args()

    beta = np.load(args.beta_path, mmap_mode="r")
    print(beta.shape)
    mask = np.load(args.mask_path).astype(bool)
    anat_img = nib.load(args.anat_path)
    gray_mask = nib.load(args.gray_mask_path).get_fdata() > 0.5

    if beta.shape[0] != int(mask.sum()):
        raise ValueError(
            f"beta voxel count ({beta.shape[0]}) does not match mask voxels ({int(mask.sum())})."
        )

    coverage = np.mean(np.isfinite(beta), axis=1)
    coverage_keep = coverage > args.min_trial_coverage

    activation, metric_label = _compute_activation(beta, args.metric)
    activation[~coverage_keep] = np.nan
    activation = np.nan_to_num(activation, nan=0.0, posinf=0.0, neginf=0.0)

    vol = np.zeros(mask.shape, dtype=np.float32)
    vol[mask] = activation
    vol[~gray_mask] = 0.0

    finite_vals = activation[np.isfinite(activation) & (activation > 0)]
    if finite_vals.size == 0:
        raise RuntimeError("No finite activation values found after coverage filtering.")
    if not 0.0 <= args.top_percentile <= 100.0:
        raise ValueError(f"--top-percentile must be in [0, 100], got {args.top_percentile}.")
    top_threshold = float(np.percentile(finite_vals, args.top_percentile))
    vmax = float(np.percentile(finite_vals, args.vmax_percentile))

    mask_coords = np.argwhere(mask)
    mask_coords_mm = nib.affines.apply_affine(anat_img.affine, mask_coords)
    motor_mask = _motor_like_roi_mask(mask_coords_mm)
    if np.any(motor_mask):
        motor_vals = activation[motor_mask]
        motor_positive = motor_vals[motor_vals > 0]
        if motor_positive.size > 0:
            motor_max = float(np.max(motor_positive))
            motor_median = float(np.median(motor_positive))
            print(
                f"Top voxels only (>=p{args.top_percentile:.1f}, threshold={top_threshold:.4f}); "
                f"motor-like ROI median={motor_median:.4f}, max={motor_max:.4f}",
                flush=True,
            )

    peaks = _top_peak_report(vol, anat_img.affine, n=5)
    print("Top 5 peaks (ijk -> MNI xyz, value):", flush=True)
    for idx, (ijk, xyz, value) in enumerate(peaks, start=1):
        print(
            f"  {idx}. {ijk} -> ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}), {value:.4f}",
            flush=True,
        )

    activation_img = nib.Nifti1Image(vol, anat_img.affine, anat_img.header)
    view = plotting.view_img(
        activation_img,
        bg_img=anat_img,
        cmap="jet",
        symmetric_cmap=False,
        threshold=top_threshold,
        vmax=vmax,
        colorbar=True,
        title=(
            f"Pre-optimization activation: {metric_label} "
            f"(coverage>{args.min_trial_coverage:.2f}, top >= p{args.top_percentile:.1f})"
        ),
    )

    out_dir = os.path.abspath(
        os.path.expanduser(os.environ.get("FMRI_PLOT_OUT_DIR", os.getcwd()))
    )
    os.makedirs(out_dir, exist_ok=True)
    pct_tag = str(args.vmax_percentile).replace(".", "p")
    top_tag = str(args.top_percentile).replace(".", "p")
    out_html = os.path.join(
        out_dir,
        f"preopt_activation_{args.metric}_cov{args.min_trial_coverage:.2f}_topvox_p{top_tag}_vmaxp{pct_tag}.html",
    )
    view.save_as_html(out_html)
    print(f"Saved: {out_html}", flush=True)

    return view


if __name__ == "__main__":
    main()
