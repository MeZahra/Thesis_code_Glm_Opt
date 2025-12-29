import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting

import Beta_preprocessing as bp


def _mask_cmap(rgb, alpha=0.6):
    colors = [
        (0.0, 0.0, 0.0, 0.0),
        (rgb[0], rgb[1], rgb[2], alpha),
    ]
    return ListedColormap(colors)


def main():
    data_root = (Path.cwd() / bp.DATA_DIRNAME).resolve()
    sub_label = f"sub-pd0{bp.sub}"
    data_paths = {
        "anat": data_root / f"{sub_label}_ses-{bp.ses}_T1w_brain.nii.gz",
        "brain": data_root / f"{sub_label}_ses-{bp.ses}_T1w_brain_mask.nii.gz",
        "csf": data_root / f"{sub_label}_ses-{bp.ses}_T1w_brain_pve_0.nii.gz",
        "gray": data_root / f"{sub_label}_ses-{bp.ses}_T1w_brain_pve_1.nii.gz",
    }

    anat_img = nib.load(str(data_paths["anat"]))
    back_mask = nib.load(str(data_paths["brain"])).get_fdata(dtype=np.float32)
    csf_mask = nib.load(str(data_paths["csf"])).get_fdata(dtype=np.float32)
    gray_mask = nib.load(str(data_paths["gray"])).get_fdata(dtype=np.float32)

    back_img = nib.Nifti1Image((back_mask > 0).astype(np.float32), anat_img.affine, anat_img.header)
    csf_img = nib.Nifti1Image((csf_mask > 0).astype(np.float32), anat_img.affine, anat_img.header)
    gray_img = nib.Nifti1Image((gray_mask > 0).astype(np.float32), anat_img.affine, anat_img.header)

    display = plotting.plot_anat(anat_img, title="Anatomy with CSF/Brain/Gray masks", dim=-1)
    display.add_overlay(back_img, cmap=_mask_cmap((1.0, 0.35, 0.25)), threshold=0.5)
    display.add_overlay(csf_img, cmap=_mask_cmap((0.2, 0.6, 1.0)), threshold=0.5)
    display.add_overlay(gray_img, cmap=_mask_cmap((0.2, 0.8, 0.4)), threshold=0.5)

    out_png = Path.cwd() / f"mask_overlay_sub{bp.sub}_ses{bp.ses}_run{bp.run}.png"
    display.savefig(out_png)
    display.close()
    print(f"Saved mask overlay PNG: {out_png}", flush=True)


if __name__ == "__main__":
    main()
