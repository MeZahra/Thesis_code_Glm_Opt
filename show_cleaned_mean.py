import argparse
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.image import resample_to_img


BASE_PATH = Path(
    "/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/"
    "PRECISIONSTIM_PD_Data_Results/fMRI_preprocessed_data/"
    "Rev_pipeline/derivatives"
)


def load_mean_volume(sub: str, ses: int, run: int) -> tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray]:
    """
    Rebuild the mean beta volume from the saved 2-D filtered data and the mask.

    Returns
    -------
    mean_img : nib.Nifti1Image
        Reconstructed 3-D mean beta image in BOLD space.
    anat_img : nib.Nifti1Image
        Anatomical image used as background.
    mean_vals : np.ndarray
        Mean beta values for voxels that passed filtering.
    """
    cleaned_path = Path(f"cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy")
    mask_path = Path(f"mask_all_nan_sub{sub}_ses{ses}_run{run}.npy")

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Missing cleaned beta file: {cleaned_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask file: {mask_path}")

    beta_2d = np.load(cleaned_path)  # shape: (n_valid_voxels, n_trials)
    mask_flat = np.load(mask_path)   # shape: (n_voxels,), True iff voxel was all-NaN

    data_name = f"sub-pd0{sub}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz"
    bold_path = BASE_PATH / f"sub-pd0{sub}" / f"ses-{ses}" / "func" / data_name
    if not bold_path.exists():
        raise FileNotFoundError(f"Missing BOLD reference image: {bold_path}")

    anat_path = (
        BASE_PATH
        / f"sub-pd0{sub}"
        / f"ses-{ses}"
        / "anat"
        / f"sub-pd0{sub}_ses-{ses}_T1w_brain.nii.gz"
    )
    if not anat_path.exists():
        raise FileNotFoundError(f"Missing anatomical image: {anat_path}")

    bold_img = nib.load(str(bold_path))
    anat_img = nib.load(str(anat_path))

    if mask_flat.ndim != 1 or mask_flat.shape[0] != np.prod(bold_img.shape[:3]):
        raise ValueError("Mask size does not match the spatial dimensions of the BOLD image.")

    mean_vals = np.nanmean(beta_2d, axis=1).astype(np.float32)
    mean_volume_flat = np.full(mask_flat.shape[0], np.nan, dtype=np.float32)
    mean_volume_flat[~mask_flat] = mean_vals
    mean_volume = mean_volume_flat.reshape(bold_img.shape[:3])

    mean_img = nib.Nifti1Image(mean_volume, bold_img.affine, bold_img.header)
    mean_img = resample_to_img(mean_img, anat_img, interpolation="linear")
    return mean_img, anat_img, mean_vals


def show_mean(sub: str, ses: int, run: int, output: Optional[Path]) -> None:
    mean_img, anat_img, mean_vals = load_mean_volume(sub, ses, run)

    finite_vals = mean_vals[np.isfinite(mean_vals)]
    if finite_vals.size == 0:
        raise ValueError("No finite mean beta values found in the cleaned volume.")

    vmax = np.percentile(np.abs(finite_vals), 99)
    vmax = float(max(vmax, 1e-6))

    view = plotting.view_img(
        mean_img,
        bg_img=anat_img,
        cmap="cold_hot",
        symmetric_cmap=True,
        vmax=vmax,
        # threshold="auto",
        colorbar=True,
        title=f"Mean cleaned beta | sub{sub} ses{ses} run{run}",
    )

    if output is None:
        # display in an interactive window (falls back to temp HTML in notebooks)
        view.open_in_browser()
    else:
        output = output.with_suffix(".html")
        view.save_as_html(str(output))
        print(f"Saved interactive viewer to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the mean cleaned beta volume on the anatomical image."
    )
    parser.add_argument("--sub", required=True, help="Subject identifier, e.g. 04")
    parser.add_argument("--ses", required=True, type=int, help="Session number, e.g. 1")
    parser.add_argument("--run", required=True, type=int, help="Run number, e.g. 1")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output HTML file for the interactive viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    show_mean(args.sub, args.ses, args.run, args.output)


if __name__ == "__main__":
    main()
