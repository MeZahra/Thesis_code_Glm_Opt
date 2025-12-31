import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap
from nilearn import plotting

import Beta_preprocessing as bp


def _label_cmap(colors, alpha=0.6):
    rgba = [(0.0, 0.0, 0.0, 0.0)]
    rgba.extend((r, g, b, alpha) for r, g, b in colors)
    return ListedColormap(rgba)


def _robust_range(data, lower=2.0, upper=98.0):
    finite = data[np.isfinite(data)]
    finite = finite[finite > 0]
    if finite.size == 0:
        return float(np.nanmin(data)), float(np.nanmax(data))
    vmin, vmax = np.percentile(finite, [lower, upper])
    if vmin == vmax:
        vmax = vmin + np.finfo(np.float32).eps
    return float(vmin), float(vmax)


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

    regions = [
        ("brain", back_mask, (1.0, 0.35, 0.25)),
        ("csf", csf_mask, (0.2, 0.6, 1.0)),
        ("gray", gray_mask, (0.2, 0.8, 0.4)),
    ]

    label_data = np.zeros(back_mask.shape, dtype=np.int16)
    for label_idx, (_, mask_data, _) in enumerate(regions, start=1):
        label_data[mask_data > 0] = label_idx

    label_img = nib.Nifti1Image(label_data, anat_img.affine, anat_img.header)
    label_cmap = _label_cmap([color for _, _, color in regions], alpha=0.6)

    if not hasattr(plotting, "view_image"):
        plotting.view_image = plotting.view_img

    view = plotting.view_image(
        label_img,
        bg_img=anat_img,
        cmap=label_cmap,
        symmetric_cmap=False,
        vmin=0,
        vmax=len(regions),
        threshold=0.5,
        resampling_interpolation="nearest",
        opacity=0.6,
        colorbar=True,
        title="Anatomy with CSF/Brain/Gray masks",
    )

    out_html = Path.cwd() / f"mask_overlay_sub{bp.sub}_ses{bp.ses}_run{bp.run}.html"
    view.save_as_html(out_html)
    print(f"Saved mask overlay HTML: {out_html}", flush=True)

    gray_value_img = nib.Nifti1Image(gray_mask, anat_img.affine, anat_img.header)
    vmin, vmax = _robust_range(gray_mask)
    gray_view = plotting.view_image(
        gray_value_img,
        bg_img=anat_img,
        cmap="viridis",
        symmetric_cmap=False,
        vmin=vmin,
        vmax=vmax,
        threshold=0.0,
        resampling_interpolation="continuous",
        opacity=0.7,
        colorbar=True,
        title="Anatomy with Gray mask values",
    )

    out_gray_html = Path.cwd() / f"gray_overlay_sub{bp.sub}_ses{bp.ses}_run{bp.run}.html"
    gray_view.save_as_html(out_gray_html)
    print(f"Saved gray-value overlay HTML: {out_gray_html}", flush=True)
    return view


if __name__ == "__main__":
    main()
