import warnings
import nibabel as nib
import numpy as np
from nilearn import plotting
from nilearn.plotting import view_img
from nilearn.image import resample_to_img
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

def save_beta_html(beta_3d, anat_img, bold_img, A, title, fname):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*symmetric_cmap=False.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=".*vmin cannot be chosen when cmap is symmetric.*", category=UserWarning)

    finite = np.isfinite(beta_3d)
    beta_filled = np.where(finite, beta_3d, 0.0)
    
    finite = np.isfinite(beta_3d)
    vmax = np.nanpercentile(beta_filled[finite], 99)  
    vmin = np.nanpercentile(beta_filled[finite], 1)
    thr = 1e-6 

    beta_on_anat = nib.Nifti1Image(beta_3d, A, bold_img.header)
    
    view = view_img(beta_on_anat, bg_img=anat_img, cmap='seismic', symmetric_cmap=False, vmax=vmax, vmin=vmin, threshold=thr, 
                    colorbar=True, title=title)
    view.save_as_html(file_name=fname)


def cfs_brain_mask_plot(nonzero_mask, mask, bold_img, anat_img):
    # plot the selected voxels on the anatomical image
    voxel_coords = np.column_stack(nonzero_mask)
    overlay_data = np.zeros(mask.shape, dtype=np.float32)
    overlay_data[tuple(voxel_coords.T)] = 1.0

    overlay_img = nib.Nifti1Image(overlay_data, bold_img.affine, bold_img.header)

    overlay_view = plotting.view_img(overlay_img, bg_img=anat_img, cmap='autumn', symmetric_cmap=False, 
                                    threshold=0, vmax=1, opacity=0.9, colorbar=False, title='nonzero_mask voxels')

    return overlay_view


def mean_beta_outlier_voxels(beta, bold_data, bold_img, anat_img, nonzero_mask, sub, ses, run): 
    finite_beta = beta[np.isfinite(beta)]
    lower_thr, upper_thr = np.nanpercentile(finite_beta, [1, 99])
    print(f'low_thr: {lower_thr:.2f}, high_thr: {upper_thr:.2f}') #low_thr: -4.64, high_thr: 4.60

    beta_extreme_mask = np.logical_or(beta < lower_thr, beta > upper_thr)
    voxels_with_extreme_beta = np.any(beta_extreme_mask, axis=1)
    print(f'number of voxels with at least one extreme beta: {voxels_with_extreme_beta.sum()}')
    # low_thr: -4.64, high_thr: 4.60

    beta_mean = np.nanmean(beta, axis=1)
    beta_mean_extreme = np.full(beta_mean.shape, np.nan, dtype=np.float32)
    beta_mean_extreme[voxels_with_extreme_beta] = beta_mean[voxels_with_extreme_beta]

    outlier_volume = np.full(bold_data.shape[:3], np.nan, dtype=np.float32)
    outlier_volume[nonzero_mask] = beta_mean_extreme

    outlier_img = nib.Nifti1Image(outlier_volume, bold_img.affine, bold_img.header)
    outlier_img = resample_to_img(outlier_img, anat_img, interpolation='linear')

    outlier_view = plotting.view_img(
        outlier_img,
        bg_img=anat_img,
        cmap='jet',
        symmetric_cmap=False,
        colorbar=True,
        title='Mean beta for voxels with outlier responses'
    )
    outlier_view.save_as_html(file_name=f'outlier_betas_sub{sub}_ses{ses}_run{run}.html')
    return 


def csf_mask_with_outlier(csf_mask_data, voxels_with_extreme_beta, bold_data, bold_img, anat_img, nonzero_mask, sub, ses, run):
    extreme_voxel_volume = np.zeros(bold_data.shape[:3], dtype=bool)
    extreme_voxel_volume[nonzero_mask] = voxels_with_extreme_beta

    csf_overlap_volume = np.logical_and(extreme_voxel_volume, csf_mask_data)
    n_overlap_voxels = int(csf_overlap_volume.sum())
    print(f"Extreme-beta voxels overlapping CSF: {n_overlap_voxels}") #Extreme-beta voxels overlapping CSF: 0

    overlay_volume = np.zeros(bold_data.shape[:3], dtype=np.int8)
    overlay_volume[csf_mask_data] = 10
    overlay_volume[extreme_voxel_volume] = 20
    overlay_volume[csf_overlap_volume] = 30

    overlay_img = nib.Nifti1Image(overlay_volume.astype(np.float32), bold_img.affine, bold_img.header)
    overlay_img = resample_to_img(overlay_img, anat_img, interpolation='nearest')
    cmap = mcolors.ListedColormap(['blue', 'green', 'red'])

    overlay_view = plotting.view_img(
        overlay_img,
        bg_img=anat_img,
        cmap=cmap,
        symmetric_cmap=False,
        threshold=0.5,
        opacity=0.8,
        vmax=20,
        vmin=0,
        title='CSF vs. Extreme Beta Voxels (labels: 1=CSF, 2=Extreme, 3=Overlap)'
    )
    overlay_view.save_as_html(file_name=f'csf_extreme_overlap_sub{sub}_ses{ses}_run{run}.html')
    return  overlay_img

def check_beta_range_and_outliers(beta, bold_data, bold_img, anat_img, nonzero_mask):
    # beta histogram and plot on the brain
    outlier_threshold = 200.0

    beta_min = np.nanmin(beta)
    beta_max = np.nanmax(beta)
    print(f"Beta values range: [{beta_min:.2f}, {beta_max:.2f}]")

    plt.figure()
    plt.hist(beta.flatten())
    plt.title('GLM beta (all trials) histogram')
    plt.xlabel('Beta value')
    plt.ylabel('Voxel count')
    plt.show()

    outlier_mask = np.abs(beta) > outlier_threshold
    outlier_voxel_mask = np.any(outlier_mask, axis=1)
    outlier_voxel_coords = np.column_stack(nonzero_mask)[outlier_voxel_mask]

    n_outlier_voxels = int(outlier_voxel_mask.sum())
    n_total_voxels = beta.shape[0]
    print(f"{(n_outlier_voxels / n_total_voxels) * 100:.2f}% of voxels are outliers ({n_outlier_voxels} of {n_total_voxels}).")

    outlier_volume = np.zeros(bold_data.shape[:3], dtype=np.float32)
    outlier_volume[nonzero_mask] = outlier_voxel_mask.astype(np.float32)

    outlier_img = nib.Nifti1Image(outlier_volume, bold_img.affine, bold_img.header)
    outlier_img = resample_to_img(outlier_img, anat_img, interpolation='nearest')

    outlier_view = plotting.view_img(
        outlier_img,
        bg_img=anat_img,
        cmap='autumn',
        symmetric_cmap=False,
        threshold=0.5,
        vmax=1,
        opacity=0.8,
        title=f'Outlier voxels |beta| > {outlier_threshold:.0f}'
    )
    return outlier_view

def check_avg_beta_range(beta, bold_data, bold_img, anat_img, nonzero_mask):
    outlier_volume = np.zeros(bold_data.shape[:3], dtype=np.float32)
    beta_mean = np.nanmean(beta, axis=1)

    # Clip to percentile bounds to suppress extreme outliers for visualization
    lower_clip, upper_clip = np.nanpercentile(beta_mean, [2, 98])
    beta_mean_clipped = np.clip(beta_mean, lower_clip, upper_clip)

    outlier_volume[nonzero_mask] = beta_mean_clipped

    fig, axs = plt.subplots(1,2)
    axs[0].hist(beta_mean, bins=100, color='blue', alpha=0.35)
    axs[0].set_title('Mean Beta, all Voxels')
    axs[1].hist(beta_mean_clipped, bins=100, color='orange', alpha=0.7)
    axs[1].set_title('Mean Beta, all Voxels, clipped')

    plt.tight_layout()
    plt.show()

    outlier_img = nib.Nifti1Image(outlier_volume, bold_img.affine, bold_img.header)
    outlier_img = resample_to_img(outlier_img, anat_img, interpolation='nearest')

    outlier_view = plotting.view_img(
        outlier_img,
        bg_img=anat_img,
        cmap='jet',
        opacity=0.8,
        title='Mean Beta of all Voxels (clipped)'
    )
    return outlier_view


def active_voxel_plot(clean_active_beta, voxels_with_extreme_beta, reject, bold_img, anat_img, nonzero_mask, sub, ses, run):
    plt.figure()
    plt.hist(clean_active_beta.ravel())
    plt.title("Active Voxels Beta Values")

    plt.figure()
    plt.hist(np.nanmean(clean_active_beta, axis=1))
    plt.title("Active Voxels Mean Beta Values")

    clean_mask = ~np.asarray(voxels_with_extreme_beta, dtype=bool)
    clean_indices = np.nonzero(clean_mask)[0]
    reject_active = np.asarray(reject, dtype=bool)
    voxel_indices = clean_indices[reject_active]

    stat_values = np.nanmean(clean_active_beta, axis=1)
    stat_values = np.asarray(stat_values, dtype=np.float32)
    voxel_indices = np.asarray(voxel_indices, dtype=np.int64)
    finite = np.isfinite(stat_values)

    volume = np.full(bold_img.shape[:3], np.nan, dtype=np.float32)
    coords = tuple(axis[voxel_indices] for axis in nonzero_mask)
    volume[coords] = stat_values

    stat_img = nib.Nifti1Image(volume, bold_img.affine, bold_img.header)
    stat_img = resample_to_img(stat_img, anat_img, interpolation='linear')

    view = plotting.view_img(
        stat_img,
        bg_img=anat_img,
        cmap='jet',
        colorbar=True,
        title='Mean beta for active voxels'
    )
    view.save_as_html(f'sub{sub}_ses{ses}_run{run}_meanBeta_activeVoxels.html')
    return
    
