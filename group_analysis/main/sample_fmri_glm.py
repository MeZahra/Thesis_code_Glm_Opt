"""Minimal fMRI-style first-level GLM example with synthetic data.

This script uses Nilearn's GLM implementation to fit a task regressor
convolved with an SPM HRF to a small synthetic 4D fMRI image.
"""

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from pathlib import Path


def make_sample_data(seed=7):
    """Create a small synthetic fMRI image with one active voxel cluster."""
    rng = np.random.default_rng(seed)

    tr = 1.0
    n_scans = 120
    frame_times = np.arange(n_scans) * tr
    volume_shape = (8, 8, 8)

    events = pd.DataFrame(
        {
            "onset": [10, 30, 50, 70, 90],
            "duration": [5, 5, 5, 5, 5],
            "trial_type": ["squeeze"] * 5,
        }
    )

    design = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
    )
    task_signal = design["squeeze"].to_numpy()
    task_signal = (task_signal - task_signal.mean()) / task_signal.std()

    data = rng.normal(loc=0.0, scale=1.0, size=(*volume_shape, n_scans))
    active_mask = np.zeros(volume_shape, dtype=bool)
    active_mask[3:5, 3:5, 3:5] = True
    data[active_mask, :] += 2.5 * task_signal

    affine = np.eye(4)
    fmri_img = nib.Nifti1Image(data.astype(np.float32), affine)
    mask_img = nib.Nifti1Image(np.ones(volume_shape, dtype=np.uint8), affine)
    return fmri_img, mask_img, events, active_mask, tr


def save_beta_slice_image(beta_values, active_mask, max_voxel, output_path):
    """Save one axial slice showing beta/effect-size values across voxels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    z_index = max_voxel[2]
    beta_slice = beta_values[:, :, z_index]
    active_slice = active_mask[:, :, z_index]
    vmax = np.nanmax(np.abs(beta_slice))

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(
        beta_slice.T,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.contour(active_slice.T, levels=[0.5], colors="black", linewidths=1.5)
    ax.scatter(max_voxel[0], max_voxel[1], marker="x", color="yellow", s=80)
    ax.set_title(f"Squeeze contrast beta values, z-slice {z_index}")
    ax.set_xlabel("Voxel x")
    ax.set_ylabel("Voxel y")
    fig.colorbar(image, ax=ax, label="Beta / effect size")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_glm():
    fmri_img, mask_img, events, active_mask, tr = make_sample_data()

    model = FirstLevelModel(
        t_r=tr,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=0.01,
        noise_model="ols",
        mask_img=mask_img,
        signal_scaling=False,
        minimize_memory=False,
    )
    model = model.fit(fmri_img, events=events)

    z_map = model.compute_contrast("squeeze", output_type="z_score")
    effect_map = model.compute_contrast("squeeze", output_type="effect_size")

    z_values = z_map.get_fdata()
    effect_values = effect_map.get_fdata()
    background_mask = ~active_mask
    max_voxel = tuple(
        int(index) for index in np.unravel_index(np.nanargmax(z_values), z_values.shape)
    )
    output_path = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "sample_fmri_glm_beta_values.png"
    )
    save_beta_slice_image(effect_values, active_mask, max_voxel, output_path)

    return {
        "n_scans": fmri_img.shape[-1],
        "n_voxels": int(np.prod(fmri_img.shape[:3])),
        "n_active_voxels": int(active_mask.sum()),
        "max_z": float(z_values[max_voxel]),
        "max_z_voxel": max_voxel,
        "mean_z_active_cluster": float(np.mean(z_values[active_mask])),
        "mean_z_background": float(np.mean(z_values[background_mask])),
        "mean_effect_active_cluster": float(np.mean(effect_values[active_mask])),
        "mean_effect_background": float(np.mean(effect_values[background_mask])),
        "beta_image_path": str(output_path),
    }


if __name__ == "__main__":
    result = run_glm()
    print("Synthetic fMRI GLM result")
    for key, value in result.items():
        print(f"{key}: {value}")
