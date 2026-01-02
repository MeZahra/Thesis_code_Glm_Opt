# %%
# In this file, the brain activation is always reported as absolute value.
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import scipy.ndimage as ndimage
from nilearn import datasets, image, plotting

DATA_DIRNAME = 'sub09_ses1'
sub = '09'
ses = 1
RUNS = (1, 2)
RUN_TAG = "_".join(str(r) for r in RUNS)
TRIAL_LEN = 9
TRIALS_PER_RUN = 90
TOTAL_TRIALS = TRIALS_PER_RUN * len(RUNS)

# %%
def _extract_trial_segments(masked_bold, trial_len, num_trials, rest_every = 30, rest_len = 20):
    """Split masked BOLD time series into trial-length segments.

    Parameters
    ----------
    masked_bold : ndarray, shape (n_voxels, n_timepoints)
        Masked BOLD time series data.
    trial_len : int
        Number of timepoints per trial.
    num_trials : int
        Number of trials to extract.
    rest_every : int or None, optional
        Insert a rest gap after every this many trials; set to 0/None to disable.
    rest_len : int, optional
        Number of timepoints to skip for each rest period.

    Returns
    -------
    segments : ndarray, shape (n_voxels, num_trials, trial_len)
        Trial segments; values without data remain NaN (float32).
    """
    num_voxels, num_timepoints = masked_bold.shape
    segments = np.full((num_voxels, num_trials, trial_len), np.nan, dtype=np.float32)
    start = 0
    for i in range(num_trials):
        end = start + trial_len
        segments[:, i, :] = masked_bold[:, start:end]
        start = end
        if rest_every and (i + 1) % rest_every == 0:
            start += rest_len
    return segments

def _save_beta_overlay(mean_abs_beta, anat_img, out_html, threshold_pct, vmax_pct, cut_coords, snapshot_path):
    """Save HTML (and optional PNG) overlay for mean |beta| volume.

    Parameters
    ----------
    mean_abs_beta : ndarray, shape (X, Y, Z)
        Volume of mean absolute beta values.
    anat_img : nib.Nifti1Image
        Anatomical image used for affine/header and background.
    out_html : str or Path
        Output HTML file path.
    threshold_pct : float
        Percentile for display threshold.
    vmax_pct : float
        Percentile for display vmax.
    cut_coords : sequence[float] or None
        Slice coordinates for the overlay view.
    snapshot_path : str or Path or None
        Optional PNG snapshot path; if None, no snapshot is saved.

    """
    finite = mean_abs_beta[np.isfinite(mean_abs_beta)]
    thr = float(np.percentile(finite, threshold_pct))
    vmax = float(np.percentile(finite, vmax_pct))
    img = nib.Nifti1Image(mean_abs_beta.astype(np.float32), anat_img.affine, anat_img.header)
    view = plotting.view_img(img, bg_img=anat_img, cmap='jet', symmetric_cmap=False, threshold=thr, vmax=vmax, colorbar=True,
                             title=f'Mean |beta| (thr p{threshold_pct}={thr:.2f}, vmax p{vmax_pct}={vmax:.2f})', cut_coords=cut_coords)
    view.save_as_html(out_html)
    print(f'Saved overlay: {out_html}', flush=True)

    if snapshot_path:
        display = plotting.plot_stat_map(img, bg_img=anat_img, cmap='jet', symmetric_cbar=False, threshold=thr, vmax=vmax, colorbar=True, 
                                         title=f'Mean |beta| (thr p{threshold_pct}, vmax p{vmax_pct})',cut_coords=cut_coords)
        display.savefig(snapshot_path)
        display.close()
        print(f'Saved snapshot: {snapshot_path}', flush=True)

def _save_overlay_html(data, anat_img, out_html, title, threshold_pct, vmax_pct, cmap='jet', symmetric_cmap = False, cut_coords=None):
    """Save an HTML overlay for a 3D volume.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        Volume to display.
    anat_img : nib.Nifti1Image
        Anatomical image used for affine/header and background.
    out_html : str or Path
        Output HTML file path.
    title : str
        Title for the overlay.
    threshold_pct : float or None
        Percentile for threshold when threshold is None.
    vmax_pct : float or None
        Percentile for vmax when vmax is None.
    cmap : str, optional
        Matplotlib colormap name.
    symmetric_cmap : bool, optional
        Whether to use a symmetric colormap.
    cut_coords : sequence[float] or None, optional
        Slice coordinates for the overlay view.
    """
    img = nib.Nifti1Image(data.astype(np.float32), anat_img.affine, anat_img.header)
    finite = data[np.isfinite(data)]
    threshold = float(np.percentile(finite, threshold_pct))
    vmax = float(np.percentile(finite, vmax_pct))

    view = plotting.view_img(img, bg_img=anat_img, cmap=cmap, symmetric_cmap=symmetric_cmap, threshold=threshold, vmax=vmax, colorbar=True, title=title, cut_coords=cut_coords)
    view.save_as_html(out_html)
    print(f'Saved overlay: {out_html}', flush=True)

def _with_tag(path, tag):
    safe_tag = str(tag).strip().replace(' ', '_')
    if path.name.endswith(".nii.gz"):
        base = path.name[:-7]
        return path.with_name(f"{base}_{safe_tag}.nii.gz")
    return path.with_name(f"{path.stem}_{safe_tag}{path.suffix}")

def _mean_abs_beta_volume(beta, coords, volume_shape):
    """Project per-voxel beta values into a 3D volume by mean |beta|.

    Parameters
    ----------
    beta : ndarray, shape (n_voxels, n_trials)
        Beta values per voxel and trial.
    coords : tuple of ndarray
        Tuple of (x_idx, y_idx, z_idx), each shape (n_voxels,).
    volume_shape : tuple of int
        Target 3D volume shape (X, Y, Z).

    Returns
    -------
    volume : ndarray, shape volume_shape
        3D volume filled with mean absolute beta; NaN elsewhere.
    """
    with np.errstate(invalid='ignore'):
        mean_abs = np.nanmean(np.abs(beta), axis=1)
    volume = np.full(volume_shape, np.nan, dtype=np.float32)
    volume[coords] = mean_abs.astype(np.float32)
    return volume

def _load_mask_indices(path):
    data = np.load(str(path), allow_pickle=True)
    if isinstance(data, np.ndarray):
        data = tuple(data)
    axes = []
    for ax in data:
        ax_arr = np.asarray(ax)
        if ax_arr.dtype.kind in ("i", "u"):
            axes.append(ax_arr.astype(np.intp, copy=False))
            continue
        ax_float = ax_arr.astype(np.float64)
        ax_round = np.rint(ax_float)
        axes.append(ax_round.astype(np.intp))
    return tuple(axes)

def _resolve_fsl_dir(flirt_path=None):
    """Resolve FSLDIR from a flirt path or environment."""
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
    """Return an MNI template path from FSL if available.

    Returns
    -------
    Path or None
        Existing template path, or None if not found.
    """
    fsl_dir = _resolve_fsl_dir(flirt_path)
    if not fsl_dir:
        return None
    for name in ("MNI152_T1_2mm_brain.nii.gz", "MNI152_T1_1mm_brain.nii.gz"):
        candidate = fsl_dir / "data" / "standard" / name
        if candidate.exists():
            return candidate
    return None

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

def _run_flirt(cmd):
    """Run FSL FLIRT with NIFTI_GZ output.

    Parameters
    ----------
    cmd : list[str]
        Command argument list for FLIRT.

    Returns
    -------
    None
    """
    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    subprocess.run(cmd, check=True, env=env)

def _compute_mni_to_anat(mni_template, anat_path, out_dir, flirt_path):
    """Compute FLIRT transform from MNI template to anatomy.

    Parameters
    ----------
    mni_template : str or Path
        MNI template image path.
    anat_path : str or Path
        Anatomical reference image path.
    out_dir : str or Path
        Output directory for transform files.
    flirt_path : str or Path
        Path to the FLIRT executable.

    Returns
    -------
    mat_path : Path
        Transform matrix file path.
    warped_path : Path
        Warped MNI template in anatomical space.
    """
    out_dir = Path(out_dir)
    mat_path = out_dir / "mni_to_anat_flirt.mat"
    warped_path = out_dir / "mni_template_in_anat.nii.gz"
    cmd = [flirt_path, "-in", str(mni_template), "-ref", str(anat_path), "-omat", str(mat_path), "-out", str(warped_path), "-dof", "12"]
    _run_flirt(cmd)
    return mat_path, warped_path

def _apply_flirt(in_path, ref_path, mat_path, out_path, flirt_path, interp="nearestneighbour"):
    """Apply an existing FLIRT transform to an image.

    Parameters
    ----------
    in_path : str or Path
        Input image path.
    ref_path : str or Path
        Reference image path.
    mat_path : str or Path
        Transform matrix path.
    out_path : str or Path
        Output image path.
    flirt_path : str or Path
        Path to the FLIRT executable.
    interp : str, optional
        FLIRT interpolation mode.

    Returns
    -------
    None
    """
    cmd = [flirt_path, "-in", str(in_path), "-ref", str(ref_path), "-applyxfm", "-init", str(mat_path), "-interp", interp,"-out", str(out_path)]
    _run_flirt(cmd)

def _save_atlas_qc_plots(atlas_in_ref, labels, ref_img, brain_mask_path, out_dir, atlas_name, registration_method):
    """Generate QC plots for atlas registration quality.

    Parameters
    ----------
    atlas_in_ref : nib.Nifti1Image
        Registered atlas in reference space.
    labels : list of str
        Atlas label names.
    ref_img : nib.Nifti1Image
        Reference anatomical image.
    brain_mask_path : str or Path
        Path to brain mask for overlap statistics.
    out_dir : Path
        Output directory for QC plots.
    atlas_name : str
        Atlas identifier for filenames.
    registration_method : str
        'flirt' or 'resample' for documentation.

    Returns
    -------
    dict
        QC statistics including overlap percentages.
    """
    from nilearn import plotting
    import json

    out_dir = Path(out_dir)
    atlas_data = np.rint(atlas_in_ref.get_fdata(dtype=np.float32)).astype(int)

    # Load brain mask for overlap stats
    brain_mask = nib.load(str(brain_mask_path)).get_fdata() > 0

    # Compute overlap statistics
    atlas_voxels = atlas_data > 0
    total_atlas = int(np.count_nonzero(atlas_voxels))
    in_brain = int(np.count_nonzero(atlas_voxels & brain_mask))
    outside_brain = total_atlas - in_brain

    qc_stats = {
        'registration_method': registration_method,
        'total_atlas_voxels': total_atlas,
        'voxels_in_brain': in_brain,
        'voxels_outside_brain': outside_brain,
        'pct_in_brain': float(in_brain) / float(total_atlas) * 100 if total_atlas else 0.0,
        'pct_outside_brain': float(outside_brain) / float(total_atlas) * 100 if total_atlas else 0.0,
    }

    # Generate overlay showing atlas boundaries on anatomy
    qc_html = out_dir / f'atlas_qc_{atlas_name}_{registration_method}.html'
    qc_png = out_dir / f'atlas_qc_{atlas_name}_{registration_method}.png'

    # Create edge map of atlas regions for visualization
    from scipy import ndimage
    atlas_binary = (atlas_data > 0).astype(np.uint8)
    atlas_edges = ndimage.sobel(atlas_binary.astype(float))
    atlas_edges = (atlas_edges > 0).astype(np.float32)
    edges_img = nib.Nifti1Image(atlas_edges, atlas_in_ref.affine, atlas_in_ref.header)

    view = plotting.view_img(edges_img, bg_img=ref_img, threshold=0.5, cmap='hot',
                            title=f'Atlas QC: {registration_method} ({qc_stats["pct_in_brain"]:.1f}% in brain)',
                            colorbar=True)
    view.save_as_html(str(qc_html))

    display = plotting.plot_roi(atlas_in_ref, bg_img=ref_img, cmap='tab20',
                               title=f'Atlas Regions: {registration_method}')
    display.savefig(str(qc_png))
    display.close()

    # Save QC statistics to JSON
    qc_json = out_dir / f'atlas_qc_{atlas_name}_{registration_method}.json'
    with open(qc_json, 'w') as f:
        json.dump(qc_stats, f, indent=2)

    print(f"Atlas QC: {in_brain}/{total_atlas} voxels in brain ({qc_stats['pct_in_brain']:.1f}%)", flush=True)
    print(f"Saved atlas QC plots: {qc_html.name}, {qc_png.name}, {qc_json.name}", flush=True)

    return qc_stats

def _select_roi_indices(labels, label_patterns):
    """Select atlas label indices matching requested patterns.

    Parameters
    ----------
    labels : sequence of str
        Atlas label names; index 0 is assumed to be background.
    label_patterns : str or None
        Comma-separated substrings to match (case-insensitive); None selects all.

    Returns
    -------
    indices : list of int
        Label indices that match requested patterns.
    """
    patterns = []
    if label_patterns:
        patterns = [p.strip().lower() for p in label_patterns.split(",") if p.strip()]
    indices = []
    for idx, name in enumerate(labels):
        if idx == 0 or name.strip().lower() == "background":
            continue
        if not patterns:
            indices.append(idx)
            continue
        lname = name.lower()
        if any(pattern in lname for pattern in patterns):
            indices.append(idx)
    return indices

def _align_atlas_to_reference(atlas_img, anat_img, anat_path, ref_img, out_dir, assume_mni=False, return_method=False):
    """Resample atlas to the reference image space.

    Parameters
    ----------
    atlas_img : nib.Nifti1Image
        Atlas image in MNI or anatomical space.
    anat_img : nib.Nifti1Image
        Anatomical image for header-based resampling.
    anat_path : str or Path
        Anatomical image path for FLIRT registration.
    ref_img : nib.Nifti1Image
        Reference image defining target shape/affine.
    out_dir : str or Path
        Directory for FLIRT outputs and temporary files.
    assume_mni : bool, optional
        If True, skip MNI-to-anat registration and only resample.
    return_method : bool, optional
        If True, also return the registration method string ('flirt' or 'resample').

    Returns
    -------
    atlas_in_ref : nib.Nifti1Image
        Atlas resampled to ref_img space (shape ref_img.shape[:3]).
    registration_method : str, optional
        Returned only when return_method is True.
    """
    use_flirt = False
    flirt_path = None
    mni_template_img = None
    registration_method = "resample"

    if not assume_mni:
        flirt_path = _find_flirt()
        mni_template = _default_mni_template(flirt_path)
        if flirt_path and mni_template and mni_template.exists():
            mni_template_img = nib.load(str(mni_template))
            _compute_mni_to_anat(mni_template, anat_path, out_dir, flirt_path)
            use_flirt = True
            registration_method = "flirt"
            print("Registered MNI template to anatomy with FLIRT.", flush=True)
        else:
            print("Warning: FLIRT or MNI template not available; falling back to header-based resampling.", flush=True)

    if use_flirt:
        if mni_template_img is None:
            mni_template_img = nib.load(str(mni_template))
        mat_path = Path(out_dir) / "mni_to_anat_flirt.mat"
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            tmpdir = Path(tmpdir)
            atlas_mni_path = tmpdir / "atlas_mni.nii.gz"
            atlas_anat_path = tmpdir / "atlas_in_anat.nii.gz"
            atlas_mni_img = image.resample_to_img(atlas_img, mni_template_img, interpolation="nearest", force_resample=True, copy_header=True)
            atlas_mni_img.to_filename(str(atlas_mni_path))
            _apply_flirt(atlas_mni_path, anat_path, mat_path, atlas_anat_path, flirt_path)
            atlas_in_anat_img = nib.load(str(atlas_anat_path))
            # Detach from temp file path before the temp dir is removed.
            atlas_in_anat = nib.Nifti1Image(atlas_in_anat_img.get_fdata(dtype=np.float32), atlas_in_anat_img.affine, atlas_in_anat_img.header)
    else:
        atlas_in_anat = image.resample_to_img(atlas_img, anat_img, interpolation="nearest", force_resample=True, copy_header=True)

    # Diagnostic output
    method_label = 'FLIRT registration' if use_flirt else 'header-based resampling'
    print(f"Atlas alignment method: {method_label}", flush=True)
    if not use_flirt:
        print("WARNING: Using fallback resampling - atlas alignment may be suboptimal", flush=True)

    if (atlas_in_anat.shape[:3] != ref_img.shape[:3] or not np.allclose(atlas_in_anat.affine, ref_img.affine)):
        atlas_in_ref = image.resample_to_img(atlas_in_anat, ref_img, interpolation="nearest", force_resample=True, copy_header=True)
        if return_method:
            return atlas_in_ref, registration_method
        return atlas_in_ref

    if return_method:
        return atlas_in_anat, registration_method
    return atlas_in_anat

def _rank_rois_by_beta(summary, anat_img, anat_path, ref_img, out_path, atlas_threshold, label_patterns, assume_mni, summary_stat='percentile_95'):
    """Rank atlas ROIs by summary statistic and write a CSV report.

    Parameters
    ----------
    summary : ndarray, shape (X, Y, Z)
        Summary volume; must match ref_img.shape[:3].
    anat_img : nib.Nifti1Image
        Anatomical image used for resampling.
    anat_path : str or Path
        Anatomical image path for FLIRT registration.
    ref_img : nib.Nifti1Image
        Reference image defining target shape/affine.
    out_path : str or Path
        Output CSV file path.
    atlas_threshold : int
        Harvard-Oxford atlas threshold value.
    label_patterns : str or None
        Comma-separated substrings to match ROI labels.
    assume_mni : bool
        If True, assume atlas is already in MNI space.
    summary_stat : str
        ROI summary statistic: 'mean', 'mean_abs', 'percentile_95', 'percentile_90', or 'peak'.

    Returns
    -------
    None
    """
    atlas = datasets.fetch_atlas_harvard_oxford(f"cort-maxprob-thr{atlas_threshold}-2mm")
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    labels = [lbl.decode("utf-8", errors="replace") if isinstance(lbl, bytes) else str(lbl) for lbl in atlas.labels]
    if summary.shape != ref_img.shape[:3]:
        raise ValueError(f"Summary shape does not match ROI reference image; summary shape={summary.shape}, ref shape={ref_img.shape[:3]}.")

    atlas_in_ref, registration_method = _align_atlas_to_reference(
        atlas_img,
        anat_img,
        anat_path,
        ref_img,
        out_path.parent,
        assume_mni=assume_mni,
        return_method=True,
    )

    # Generate atlas QC diagnostics
    brain_mask_path = str(anat_path).replace('T1w_brain.nii.gz', 'T1w_brain_mask.nii.gz')
    if Path(brain_mask_path).exists():
        _save_atlas_qc_plots(atlas_in_ref, labels, ref_img, brain_mask_path, out_path.parent,
                            f'thr{atlas_threshold}', registration_method)

    atlas_out_path = out_path.parent / f'atlas_thr{atlas_threshold}_{registration_method}.nii.gz'
    atlas_in_ref.to_filename(str(atlas_out_path))
    print(f"Saved aligned atlas: {atlas_out_path}", flush=True)

    atlas_data = np.rint(atlas_in_ref.get_fdata(dtype=np.float32)).astype(int)
    indices = _select_roi_indices(labels, label_patterns)
    if not indices:
        raise ValueError("No atlas labels matched the requested ROI label patterns.")

    results = []
    for idx in indices:
        mask = atlas_data == idx
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            roi_stat = np.nan
            valid_voxels = 0
        else:
            values = summary[mask]
            finite = values[np.isfinite(values)]
            valid_voxels = int(finite.size)

            if finite.size == 0:
                roi_stat = np.nan
            elif summary_stat == 'mean':
                roi_stat = float(np.nanmean(finite))
            elif summary_stat == 'mean_abs':
                roi_stat = float(np.nanmean(np.abs(finite)))
            elif summary_stat == 'percentile_95':
                roi_stat = float(np.nanpercentile(finite, 95))
            elif summary_stat == 'percentile_90':
                roi_stat = float(np.nanpercentile(finite, 90))
            elif summary_stat == 'peak':
                roi_stat = float(np.nanmax(finite))
            elif summary_stat == 'robust_mean':
                # Trim top/bottom 5% then take mean - resistant to outliers
                abs_finite = np.abs(finite)
                lower = np.percentile(abs_finite, 5)
                upper = np.percentile(abs_finite, 95)
                trimmed = abs_finite[(abs_finite >= lower) & (abs_finite <= upper)]
                roi_stat = float(np.mean(trimmed)) if trimmed.size > 0 else np.nan
            elif summary_stat == 'total_activation':
                # Total integrated activation: mean × valid_voxel_count
                # Rewards spatially extensive activation
                roi_stat = float(np.nanmean(np.abs(finite)) * valid_voxels)
            else:
                raise ValueError(f"Unknown summary_stat: {summary_stat}")

        results.append({"label_index": idx, "label": labels[idx], "roi_stat": roi_stat, "stat_type": summary_stat, "voxel_count": voxel_count, "valid_voxel_count": valid_voxels})

    def _sort_key(row):
        roi_stat = row["roi_stat"]
        if roi_stat is None or np.isnan(roi_stat):
            return (1, 0.0)
        return (0, -roi_stat)

    results = sorted(results, key=_sort_key)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    fieldnames = ["rank", "label_index", "label", "roi_stat", "stat_type", "voxel_count", "valid_voxel_count"]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved ROI ranking: {out_path}", flush=True)

# %%
def hampel_filter_image(image, window_size, threshold_factor, return_stats=False):
    """Apply a 3D Hampel filter to each volume in a 4D image.

    Parameters
    ----------
    image : ndarray, shape (X, Y, Z, T)
        4D image to filter; modified in place.
    window_size : int
        Size of the cubic neighborhood window.
    threshold_factor : float
        MAD multiplier used to flag outliers.
    return_stats : bool, optional
        If True, also return a dict of summary counts.

    Returns
    -------
    image : ndarray, shape (X, Y, Z, T)
        Filtered image.
    stats : dict, optional
        Only returned when return_stats is True. Keys: insufficient_total, corrected_total.
    """
    footprint = np.ones((window_size,) * 3, dtype=bool)
    insufficient_any = np.zeros(image.shape[:3], dtype=bool)
    corrected_any = np.zeros(image.shape[:3], dtype=bool)

    for t in range(image.shape[3]):
        print(f"Trial Number: {t}", flush=True)
        vol = image[..., t]
        valid = np.isfinite(vol)

        med = ndimage.generic_filter(vol, np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        mad = ndimage.generic_filter(np.abs(vol - med), np.nanmedian, footprint=footprint, mode='constant', cval=np.nan).astype(np.float32)
        counts = ndimage.generic_filter(np.isfinite(vol).astype(np.float32), np.sum, footprint=footprint, mode='constant', cval=0).astype(np.float32)
        neighbor_count = counts - valid.astype(np.float32)

        scaled_mad = 1.4826 * mad
        insufficient = valid & (neighbor_count < 3)
        insufficient_any |= insufficient
        image[..., t][insufficient] = np.nan

        enough_data = (neighbor_count >= 3) & valid
        outliers = enough_data & (np.abs(vol - med) > threshold_factor * scaled_mad)

        corrected_any |= outliers
        image[..., t][outliers] = med[outliers]

    if return_stats:
        stats = {
            'insufficient_total': int(np.count_nonzero(insufficient_any)),
            'corrected_total': int(np.count_nonzero(corrected_any)),
        }
        return image, stats

    return image

def _parse_args():
    parser = argparse.ArgumentParser(description='Beta preprocessing for GLMsingle outputs.')
    parser.add_argument('--gray-threshold', type=float, default=0.7)  # Increased from 0.5
    parser.add_argument('--skip-ttest', action='store_true', default=True)  # Skip FDR by default for motor tasks
    parser.add_argument('--skip-hampel', action='store_true')
    parser.add_argument('--glmsingle-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--output-tag', type=str, default=None)
    parser.add_argument('--mask-indices', type=str, default=None)
    parser.add_argument('--runs', type=str, default=None, help='Comma-separated run numbers to use (e.g., "1" or "1,2"); default uses RUNS.')
    parser.add_argument('--roi-stat', type=str, default='mean_abs', choices=['mean', 'mean_abs', 'percentile_95', 'percentile_90', 'peak', 'robust_mean', 'total_activation'], help='ROI summary statistic (default: mean_abs)')
    return parser.parse_args()

def _parse_runs_arg(runs_arg, available_runs):
    parts = [p.strip() for p in str(runs_arg).split(",") if p.strip()]
    runs = []
    for part in parts:
        runs.append(int(part))
    unknown = [r for r in runs if r not in available_runs]
    requested = set(runs)
    return [r for r in available_runs if r in requested]

def main():
    """Run the beta preprocessing pipeline.
    Inputs are loaded from disk based on hard-coded subject/session/run values.
    Outputs are written to the current working directory.
    """
    args = _parse_args()
    selected_runs = _parse_runs_arg(args.runs, RUNS)
    run_tag = "_".join(str(r) for r in selected_runs)
    data_root = (Path.cwd() / DATA_DIRNAME).resolve()
    
    print("Define Parameters ....")
    fdr_alpha = 0.2   # FDR alpha for voxelwise t-tests. Relaxed to 0.2 for motor discovery
    hampel_window = 5      # Hampel filter window size (voxels).
    hampel_threshold = 3.0   # Hampel MAD multiplier for outliers.
    outlier_percentile = 99.0      # Percentile cutoff for beta outliers. Relaxed from 99.9 to 99.0
    max_outlier_fraction = 0.3     # Max outlier fraction per voxel. 0.5
    overlay_threshold_pct = 60      # Overlay threshold percentile.
    overlay_vmax_pct = 99.9      # Overlay vmax percentile.
    cut_coords = None      # Slice coords for overlay; None uses default cuts.
    cut_coords = tuple(cut_coords) if cut_coords else None
    roi_tag = 'mean_abs'
    skip_roi_ranking = False      # Skip ROI ranking output.
    roi_atlas_threshold = 25      # Harvard-Oxford atlas threshold.
    roi_label_patterns = None      # ROI label filter patterns.
    
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_tag = args.output_tag
    beta_overlay_html_path = _with_tag(output_dir / f'beta_overlay_sub{sub}_ses{ses}_run{run_tag}.html', output_tag)
    clean_beta_overlay_html_path = _with_tag(output_dir / f'clean_beta_overlay_sub{sub}_ses{ses}_run{run_tag}.html', output_tag)
    ttest_beta_overlay_html_path = _with_tag(output_dir / f'ttest_beta_overlay_sub{sub}_ses{ses}_run{run_tag}.html', output_tag)
    overlay_html_path = _with_tag(output_dir / f'clean_active_beta_overlay_sub{sub}_ses{ses}_run{run_tag}.html', output_tag)
    overlay_snapshot_path = overlay_html_path.with_suffix(".png")
    cached_beta_path = _with_tag(output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run_tag}.npy', output_tag)

    print("Loading Files ...")
    sub_label = f'sub-pd0{sub}'
    bold_paths = [data_root / f'{sub_label}_ses-{ses}_run-{r}_task-mv_bold_corrected_smoothed_reg.nii.gz' for r in selected_runs]
    data_paths = {'bold': bold_paths[0], 'anat': data_root / f'{sub_label}_ses-{ses}_T1w_brain.nii.gz',
        'brain': data_root / f'{sub_label}_ses-{ses}_T1w_brain_mask.nii.gz', 'csf': data_root / f'{sub_label}_ses-{ses}_T1w_brain_pve_0.nii.gz', 'gray': data_root / f'{sub_label}_ses-{ses}_T1w_brain_pve_1.nii.gz'}
    anat_img = nib.load(str(data_paths['anat']))
    # bold_imgs = [nib.load(str(path)) for path in bold_paths]
    # bold_img = bold_imgs[0]
    # bold_data = np.concatenate([img.get_fdata() for img in bold_imgs], axis=3)
    volume_shape = anat_img.shape[:3]
    glmsingle_file = Path(args.glmsingle_file) if args.glmsingle_file else (data_root / 'TYPED_FITHRF_GLMDENOISE_RR.npy')
    if not glmsingle_file.exists():
        raise FileNotFoundError(f"GLMsingle output not found: {glmsingle_file}")
    glm_dict = np.load(str(glmsingle_file), allow_pickle=True).item()
    beta_glm = glm_dict['betasmd']
    back_mask = nib.load(str(data_paths['brain'])).get_fdata(dtype=np.float32)
    csf_mask = nib.load(str(data_paths['csf'])).get_fdata(dtype=np.float32)
    gray_mask = nib.load(str(data_paths['gray'])).get_fdata(dtype=np.float32)

    if cached_beta_path.exists():
        beta_volume_filter = np.load(cached_beta_path)
        mean_clean_active = np.nanmean(np.abs(beta_volume_filter), axis=-1)
        mean_clean_active_path = _with_tag(output_dir / f'mean_clean_active_sub{sub}_ses{ses}_run{run_tag}.nii.gz', output_tag)
        mean_clean_active_img = nib.Nifti1Image(mean_clean_active.astype(np.float32), anat_img.affine, anat_img.header)
        mean_clean_active_img.to_filename(str(mean_clean_active_path))
        print(f'Saved mean_clean_active: {mean_clean_active_path}', flush=True)
        _save_beta_overlay(mean_clean_active, anat_img=anat_img, out_html=str(overlay_html_path),threshold_pct=overlay_threshold_pct,
                            vmax_pct=overlay_vmax_pct, cut_coords=cut_coords, snapshot_path=str(overlay_snapshot_path))
        if not skip_roi_ranking:
            roi_rank_path = _with_tag(output_dir / f'roi_{roi_tag}_sub{sub}_ses{ses}_run{run_tag}.csv', output_tag)
            _rank_rois_by_beta(mean_clean_active, anat_img=anat_img, anat_path=data_paths['anat'], ref_img=anat_img, out_path=roi_rank_path,
                                atlas_threshold=roi_atlas_threshold, label_patterns=roi_label_patterns, assume_mni=False, summary_stat=args.roi_stat)
        return

    print("Apply Masking on Bold data...")
    csf_mask_data = csf_mask > 0
    back_mask_data = back_mask > 0
    if args.mask_indices:
        nonzero_mask = _load_mask_indices(args.mask_indices)
        keep_voxels = np.ones(nonzero_mask[0].shape[0], dtype=bool)
        print(f"Using mask indices from {args.mask_indices}.", flush=True)
        if args.gray_threshold is not None:
            print("Note: --gray-threshold is ignored when --mask-indices is set.", flush=True)

        # Create mapping from beta_glm voxels to mask_indices voxels
        # beta_glm was created with a gray threshold of 0.5, but mask_indices may be a subset
        # We need to find which voxels in beta_glm correspond to mask_indices
        gray_mask_data = gray_mask > 0.5  # The threshold used when creating beta_glm
        beta_mask = np.logical_and(back_mask_data, np.logical_and(gray_mask_data, ~csf_mask_data))
        beta_indices = np.where(beta_mask)

        # Create a 3D boolean mask from mask_indices
        mask_3d = np.zeros(volume_shape, dtype=bool)
        mask_3d[tuple(nonzero_mask)] = True

        # Extract this mask at beta_indices locations to get which beta voxels to keep
        beta_subset_mask = mask_3d[beta_indices]
    else:
        mask = np.logical_and(back_mask_data, ~csf_mask_data)
        nonzero_mask = np.where(mask)
        gray_mask_data = gray_mask > args.gray_threshold
        keep_voxels = gray_mask_data[nonzero_mask]
        beta_subset_mask = None
    # bold_flat = bold_data[nonzero_mask]
    # masked_bold = bold_flat[keep_voxels].astype(np.float32)
    masked_coords = tuple(ax[keep_voxels] for ax in nonzero_mask)

    print("Reshape Bold and Beta datasets...")
    start_idx = 0
    end_idx = beta_glm.shape[-1]
    beta = beta_glm[:, 0, 0, start_idx:end_idx]
    if args.runs:
        run_to_pos = {run: idx for idx, run in enumerate(RUNS)}
        run_slices = []
        for run in selected_runs:
            pos = run_to_pos[run]
            run_start = pos * TRIALS_PER_RUN
            run_end = run_start + TRIALS_PER_RUN
            run_slices.append(slice(run_start, run_end))
        if len(run_slices) == 1:
            beta = beta[:, run_slices[0]]
        else:
            beta = np.concatenate([beta[:, s] for s in run_slices], axis=1)
    total_trials = beta.shape[1]

    # Apply masking to beta to match masked_coords
    if beta_subset_mask is not None:
        # Using mask_indices: beta has all voxels from the GLM mask (gray>0.5),
        # but we only want the subset in mask_indices
        beta = beta[beta_subset_mask]
    else:
        # In the non-mask_indices case, we need to apply the keep_voxels mask
        keep_voxels_count = int(np.count_nonzero(keep_voxels))
        if beta.shape[0] == keep_voxels.shape[0]:
            beta = beta[keep_voxels]
        elif beta.shape[0] == keep_voxels_count:
            # Beta is already filtered somehow
            pass
    # if beta.shape[0] != masked_bold.shape[0]:
    #     raise ValueError(
    #         f"Beta voxels ({beta.shape[0]}) do not match masked BOLD ({masked_bold.shape[0]})."
    #     )
    # bold_data_reshape = _extract_trial_segments(masked_bold, trial_len=TRIAL_LEN, num_trials=total_trials, rest_every=30, rest_len=20)
    print("Remove NaN voxels...")
    nan_voxels = np.isnan(beta).all(axis=1)
    if np.any(nan_voxels):
        beta = beta[~nan_voxels]
        masked_coords = tuple(coord[~nan_voxels] for coord in masked_coords)
    # print(f"Beta Shape: {beta.shape}, Bold shape: {bold_data_reshape.shape}")
    beta_overlay = _mean_abs_beta_volume(beta, masked_coords, volume_shape)
    _save_overlay_html(beta_overlay, anat_img=anat_img, out_html=str(beta_overlay_html_path), title='', 
                       threshold_pct=overlay_threshold_pct, vmax_pct=overlay_vmax_pct, cut_coords=cut_coords)

    print("Remove Outlier Beta Values...")
    # My code
    # med = np.nanmedian(beta, keepdims=True)
    # mad = np.nanmedian(np.abs(beta - med), keepdims=True)
    # scale = 1.4826 * np.maximum(mad, 1e-9)
    # beta_norm = (beta - med) / scale
    # thr = np.nanpercentile(np.abs(beta_norm), outlier_percentile)
    # outlier_mask = np.abs(beta_norm) > thr
    # clean_beta = beta.copy()
    # voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
    # valid_voxels = voxel_outlier_fraction <= max_outlier_fraction
    # clean_beta[~valid_voxels] = np.nan
    # clean_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
    # keeped_mask = ~np.all(np.isnan(clean_beta), axis=1)

    # GPT code
    # per-voxel robust z
    med = np.nanmedian(beta, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(beta - med), axis=1, keepdims=True)
    scale = 1.4826 * np.maximum(mad, 1e-9)
    beta_norm = (beta - med) / scale
    # per-voxel threshold or fixed z
    thr = np.nanpercentile(np.abs(beta_norm), outlier_percentile, axis=1, keepdims=True)
    outlier_mask = np.abs(beta_norm) > thr
    clean_beta = beta.copy()
    clean_beta[outlier_mask] = np.nan
    min_valid_trials = int(0.7 * total_trials)
    valid_voxels = np.sum(np.isfinite(clean_beta), axis=1) >= min_valid_trials
    clean_beta[~valid_voxels] = np.nan
    keeped_mask = valid_voxels
    # GPT code finish


    clean_beta = clean_beta[keeped_mask]
    keeped_indices = np.flatnonzero(keeped_mask)
    # bold_data_reshape[~valid_voxels, :, :] = np.nan
    trial_outliers = np.logical_and(outlier_mask, valid_voxels[:, None])
    # bold_data_reshape = np.where(trial_outliers[:, :, None], np.nan, bold_data_reshape)
    # bold_data_reshape = bold_data_reshape[keeped_mask]
    removed_voxels = beta.shape[0] - int(np.sum(keeped_mask))
    print(f"Outlier filter removed {removed_voxels}/{beta.shape[0]}.")
    # print(f"Clean Beta Shape: {clean_beta.shape}, Bold shape: {bold_data_reshape.shape}")
    clean_beta_coords = tuple(coord[keeped_mask] for coord in masked_coords)
    clean_beta_overlay = _mean_abs_beta_volume(clean_beta, clean_beta_coords, volume_shape)
    _save_overlay_html(clean_beta_overlay, anat_img=anat_img, out_html=str(clean_beta_overlay_html_path), title='',
                       threshold_pct=overlay_threshold_pct, vmax_pct=overlay_vmax_pct, cut_coords=cut_coords)
    
    print(f"Apply ttest?: {args.skip_ttest}")
    if args.skip_ttest:
        clean_active_beta = clean_beta
        clean_active_idx = keeped_indices
        # clean_active_bold = bold_data_reshape
    else:
        tvals, pvals = ttest_1samp(clean_beta, popmean=0, axis=1, nan_policy='omit')
        tested = np.isfinite(pvals)
        rej, q, _, _ = multipletests(pvals[tested], alpha=fdr_alpha, method='fdr_bh')

        n_voxel = clean_beta.shape[0]
        qvals = np.full(n_voxel, np.nan)
        reject = np.zeros(n_voxel, dtype=bool)
        reject[tested] = rej
        qvals[tested] = q

        clean_active_beta = clean_beta[reject]
        clean_active_idx = keeped_indices[reject]
        # clean_active_bold = bold_data_reshape[reject]
        # print(f"After ttest: Beta Shape: {clean_active_beta.shape}, Bold shape: {clean_active_bold.shape}")
        ttest_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
        ttest_beta_overlay = _mean_abs_beta_volume(clean_active_beta, ttest_coords, volume_shape)
        _save_overlay_html(ttest_beta_overlay, anat_img=anat_img, out_html=str(ttest_beta_overlay_html_path), title='', 
                           threshold_pct=overlay_threshold_pct, vmax_pct=overlay_vmax_pct, cut_coords=cut_coords)

    num_trials = clean_active_beta.shape[1]
    clean_active_volume = np.full(volume_shape + (num_trials,), np.nan)
    active_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
    clean_active_volume[active_coords[0], active_coords[1], active_coords[2], :] = clean_active_beta

    print(f"Apply spatial filtering?: {args.skip_hampel}")
    if args.skip_hampel:
        beta_volume_filter = clean_active_volume
    else:
        beta_volume_filter, hampel_stats = hampel_filter_image(clean_active_volume.astype(np.float32), window_size=hampel_window, threshold_factor=hampel_threshold, return_stats=True)
        print('Total voxels with <3 neighbours:', hampel_stats['insufficient_total'], flush=True)
        print('Total corrected voxels:', hampel_stats['corrected_total'], flush=True)

    print("Saving Beta preprocessing files ....")
    nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1)
    mask_2d = nan_voxels.reshape(-1)

    np.save(output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy', beta_volume_filter)
    np.save(output_dir / f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy', mask_2d)

    beta_volume_filter = np.load(output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy')
    nan_voxels = np.all(np.isnan(beta_volume_filter), axis=-1)
    mask_2d = np.load(output_dir / f'mask_all_nan_sub{sub}_ses{ses}_run{run}.npy')
    np.save(output_dir / f"nan_mask_flat_sub{sub}_ses{ses}_run{run}.npy", mask_2d)
    beta_volume_clean_2d = beta_volume_filter[~nan_voxels]
    np.save(output_dir / f"beta_volume_filter_sub{sub}_ses{ses}_run{run}.npy", beta_volume_clean_2d)

    active_flat_idx = np.ravel_multi_index(active_coords, nan_voxels.shape)
    np.save(output_dir / f"active_flat_indices__sub{sub}_ses{ses}_run{run}.npy", active_flat_idx)
    keep_mask = ~mask_2d[active_flat_idx]
    clean_active_bold = clean_active_bold[keep_mask, ...]
    np.save(output_dir / f"active_bold_sub{sub}_ses{ses}_run{run}.npy", clean_active_bold)
    clean_active_beta = clean_active_beta[keep_mask, ...]
    print(f"After filtering: Beta Shape: {beta_volume_clean_2d.shape}, Bold shape: {clean_active_bold.shape}")

    clean_active_idx = clean_active_idx[keep_mask]
    active_coords = tuple(coord[keep_mask] for coord in active_coords)
    np.save(output_dir / f"active_coords_sub{sub}_ses{ses}_run{run}.npy", active_coords)

    mean_clean_active = np.nanmean(np.abs(beta_volume_filter), axis=-1)
    mean_clean_active_path = _with_tag(output_dir / f'mean_clean_active_sub{sub}_ses{ses}_run{run_tag}.nii.gz', output_tag)
    mean_clean_active_img = nib.Nifti1Image(mean_clean_active.astype(np.float32), anat_img.affine, anat_img.header)
    mean_clean_active_img.to_filename(str(mean_clean_active_path))
    print(f'Saved mean_clean_active: {mean_clean_active_path}', flush=True)
    _save_beta_overlay(mean_clean_active, anat_img=anat_img, out_html=str(overlay_html_path), threshold_pct=overlay_threshold_pct, 
                       vmax_pct=overlay_vmax_pct, cut_coords=cut_coords, snapshot_path=str(overlay_snapshot_path))
    if not skip_roi_ranking:
        roi_rank_path = _with_tag(output_dir / f'roi_{roi_tag}_sub{sub}_ses{ses}_run{run_tag}.csv', output_tag)
        _rank_rois_by_beta(mean_clean_active, anat_img=anat_img, anat_path=data_paths['anat'], ref_img=anat_img, out_path=roi_rank_path,
                           atlas_threshold=roi_atlas_threshold, label_patterns=roi_label_patterns, assume_mni=False, summary_stat=args.roi_stat)


if __name__ == '__main__':
    main()
