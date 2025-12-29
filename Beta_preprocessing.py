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
run = 1
TRIAL_LEN = 9

# %%
def _load_trial_keep(root, run):
    if root is None:
        return None
    for pattern in (f'trial_keep_run{run}.npy', f'trial_keep_run{run:02d}.npy'):
        path = root / pattern
        if path.exists():
            return np.load(path)
    return None

def _trial_counts(total_trials, num_runs, trial_keep):
    counts = []
    for run_idx in range(num_runs):
        keep = trial_keep[run_idx] if run_idx < len(trial_keep) else None
        if keep is not None:
            counts.append(int(np.count_nonzero(keep)))
            continue
        counts.append(None)

    if all(count is not None for count in counts):
        if sum(counts) == total_trials:
            return counts

    base = total_trials // num_runs
    counts = [base] * num_runs
    counts[-1] = total_trials - base * (num_runs - 1)
    return counts

def _extract_trial_segments(masked_bold, trial_len, num_trials, rest_every = 30, rest_len = 20):
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

def _save_beta_overlay(
    mean_abs_beta: np.ndarray,
    anat_img: nib.Nifti1Image,
    out_html: str,
    threshold_pct: float,
    vmax_pct: float,
    cut_coords=None,
    snapshot_path: str | None = None,
):
    finite = mean_abs_beta[np.isfinite(mean_abs_beta)]
    if finite.size == 0:
        raise ValueError('No finite beta values to visualize.')

    thr = float(np.percentile(finite, threshold_pct))
    vmax = float(np.percentile(finite, vmax_pct))
    img = nib.Nifti1Image(mean_abs_beta.astype(np.float32), anat_img.affine, anat_img.header)
    view = plotting.view_img(
        img,
        bg_img=anat_img,
        cmap='inferno',
        symmetric_cmap=False,
        threshold=thr,
        vmax=vmax,
        colorbar=True,
        title=f'Mean |beta| (thr p{threshold_pct}={thr:.2f}, vmax p{vmax_pct}={vmax:.2f})',
        cut_coords=cut_coords,
    )
    view.save_as_html(out_html)
    print(f'Saved overlay: {out_html}', flush=True)

    if snapshot_path:
        try:
            display = plotting.plot_stat_map(
                img,
                bg_img=anat_img,
                cmap='inferno',
                symmetric_cmap=False,
                threshold=thr,
                vmax=vmax,
                colorbar=True,
                title=f'Mean |beta| (thr p{threshold_pct}, vmax p{vmax_pct})',
                cut_coords=cut_coords,
            )
        except (TypeError, AttributeError) as exc:
            if 'symmetric_cmap' not in str(exc):
                raise
            display = plotting.plot_stat_map(
                img,
                bg_img=anat_img,
                cmap='inferno',
                threshold=thr,
                vmax=vmax,
                colorbar=True,
                title=f'Mean |beta| (thr p{threshold_pct}, vmax p{vmax_pct})',
                cut_coords=cut_coords,
            )
        display.savefig(snapshot_path)
        display.close()
        print(f'Saved snapshot: {snapshot_path}', flush=True)


def _compute_beta_summary(beta_volume_filter):
    with np.errstate(invalid='ignore'):
        summary = np.nanmean(np.abs(beta_volume_filter), axis=-1)

    return summary


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


def _select_roi_indices(labels, label_patterns):
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


def _align_atlas_to_reference(
    atlas_img,
    anat_img,
    anat_path,
    ref_img,
    out_dir,
    assume_mni=False,
):
    use_flirt = False
    flirt_path = None
    mni_template_img = None
    if not assume_mni:
        flirt_path = _find_flirt()
        mni_template = _default_mni_template()
        if flirt_path and mni_template and mni_template.exists():
            try:
                _compute_mni_to_anat(mni_template, anat_path, out_dir, flirt_path)
                use_flirt = True
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

    if use_flirt:
        if mni_template_img is None:
            mni_template_img = nib.load(str(mni_template))
        mat_path = Path(out_dir) / "mni_to_anat_flirt.mat"
        with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
            tmpdir = Path(tmpdir)
            atlas_mni_path = tmpdir / "atlas_mni.nii.gz"
            atlas_anat_path = tmpdir / "atlas_in_anat.nii.gz"
            atlas_mni_img = image.resample_to_img(
                atlas_img,
                mni_template_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
            atlas_mni_img.to_filename(str(atlas_mni_path))
            _apply_flirt(atlas_mni_path, anat_path, mat_path, atlas_anat_path, flirt_path)
            atlas_in_anat_img = nib.load(str(atlas_anat_path))
            # Detach from temp file path before the temp dir is removed.
            atlas_in_anat = nib.Nifti1Image(
                atlas_in_anat_img.get_fdata(dtype=np.float32),
                atlas_in_anat_img.affine,
                atlas_in_anat_img.header,
            )
    else:
        atlas_in_anat = image.resample_to_img(
            atlas_img,
            anat_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )

    if (
        atlas_in_anat.shape[:3] != ref_img.shape[:3]
        or not np.allclose(atlas_in_anat.affine, ref_img.affine)
    ):
        atlas_in_ref = image.resample_to_img(
            atlas_in_anat,
            ref_img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True,
        )
        return atlas_in_ref

    return atlas_in_anat


def _rank_rois_by_beta(
    summary: np.ndarray,
    anat_img: nib.Nifti1Image,
    anat_path: Path,
    ref_img: nib.Nifti1Image,
    out_path: Path,
    atlas_threshold: int,
    label_patterns: str | None,
    assume_mni: bool,
):
    atlas = datasets.fetch_atlas_harvard_oxford(f"cort-maxprob-thr{atlas_threshold}-2mm")
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    labels = [
        lbl.decode("utf-8", errors="replace") if isinstance(lbl, bytes) else str(lbl)
        for lbl in atlas.labels
    ]

    if summary.shape != ref_img.shape[:3]:
        raise ValueError(
            "Summary shape does not match ROI reference image; "
            f"summary shape={summary.shape}, ref shape={ref_img.shape[:3]}."
        )

    atlas_in_ref = _align_atlas_to_reference(
        atlas_img,
        anat_img,
        anat_path,
        ref_img,
        out_path.parent,
        assume_mni=assume_mni,
    )
    atlas_data = np.rint(atlas_in_ref.get_fdata(dtype=np.float32)).astype(int)

    indices = _select_roi_indices(labels, label_patterns)
    if not indices:
        raise ValueError("No atlas labels matched the requested ROI label patterns.")

    results = []
    for idx in indices:
        mask = atlas_data == idx
        voxel_count = int(np.count_nonzero(mask))
        if voxel_count == 0:
            mean_beta = np.nan
            valid_voxels = 0
        else:
            values = summary[mask]
            finite = values[np.isfinite(values)]
            valid_voxels = int(finite.size)
            mean_beta = float(np.nanmean(finite)) if finite.size else np.nan

        results.append(
            {
                "label_index": idx,
                "label": labels[idx],
                "mean_beta": mean_beta,
                "voxel_count": voxel_count,
                "valid_voxel_count": valid_voxels,
            }
        )

    def _sort_key(row):
        mean_beta = row["mean_beta"]
        if mean_beta is None or np.isnan(mean_beta):
            return (1, 0.0)
        return (0, -mean_beta)

    results = sorted(results, key=_sort_key)
    for rank, row in enumerate(results, start=1):
        row["rank"] = rank

    fieldnames = ["rank", "label_index", "label", "mean_beta", "voxel_count", "valid_voxel_count"]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved ROI ranking: {out_path}", flush=True)

# %%
def hampel_filter_image(image, window_size, threshold_factor, return_stats=False):
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
    parser.add_argument('--gray-threshold', type=float, default=0.5)
    parser.add_argument('--skip-ttest', action='store_true')
    parser.add_argument('--skip-hampel', action='store_true')
    return parser.parse_args()

def main():
    args = _parse_args()
    data_root = (Path.cwd() / DATA_DIRNAME).resolve()
    
    print("Define Parameters ....")
    fdr_alpha = 0.05   # FDR alpha for voxelwise t-tests.
    hampel_window = 5      # Hampel filter window size (voxels).
    hampel_threshold = 3.0   # Hampel MAD multiplier for outliers.
    outlier_percentile = 99.9      # Percentile cutoff for beta outliers.
    max_outlier_fraction = 0.5     # Max outlier fraction per voxel.
    overlay_threshold_pct = 90.0      # Overlay threshold percentile.
    overlay_vmax_pct = 99.0      # Overlay vmax percentile.
    cut_coords = None      # Slice coords for overlay; None uses default cuts.
    cut_coords = tuple(cut_coords) if cut_coords else None
    roi_tag = 'mean_abs'
    skip_roi_ranking = False      # Skip ROI ranking output.
    roi_atlas_threshold = 25      # Harvard-Oxford atlas threshold.
    roi_label_patterns = None      # ROI label filter patterns.
    
    output_dir = Path.cwd()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_html_path = output_dir / f'clean_active_beta_overlay_sub{sub}_ses{ses}_run{run}.html'
    overlay_snapshot_path = overlay_html_path.with_suffix(".png")
    cached_beta_path = output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy'
    if cached_beta_path.exists():
        beta_volume_filter = np.load(cached_beta_path)
        mean_clean_active = _compute_beta_summary(beta_volume_filter)
        _save_beta_overlay(mean_clean_active, anat_img=anat_img, out_html=str(overlay_html_path),threshold_pct=overlay_threshold_pct,
                            vmax_pct=overlay_vmax_pct, cut_coords=cut_coords, snapshot_path=str(overlay_snapshot_path))
        if not skip_roi_ranking:
            roi_rank_path = output_dir / f'roi_{roi_tag}_sub{sub}_ses{ses}_run{run}.csv'
            _rank_rois_by_beta(mean_clean_active, anat_img=anat_img, anat_path=data_paths['anat'], ref_img=anat_img, out_path=roi_rank_path, 
                                atlas_threshold=roi_atlas_threshold, label_patterns=roi_label_patterns, assume_mni=False)
        return

    print("Loading Files ...")
    sub_label = f'sub-pd0{sub}'
    data_paths = {'bold': data_root / f'{sub_label}_ses-{ses}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz',
        'brain': data_root / f'{sub_label}_ses-{ses}_T1w_brain_mask.nii.gz',
        'csf': data_root / f'{sub_label}_ses-{ses}_T1w_brain_pve_0.nii.gz',
        'gray': data_root / f'{sub_label}_ses-{ses}_T1w_brain_pve_1.nii.gz',
        'anat': data_root / f'{sub_label}_ses-{ses}_T1w_brain.nii.gz'}
    anat_img = nib.load(str(data_paths['anat']))
    bold_img = nib.load(str(data_paths['bold']))
    bold_data = bold_img.get_fdata()
    glmsingle_file = data_root / 'TYPED_FITHRF_GLMDENOISE_RR.npy'
    glm_dict = np.load(str(glmsingle_file), allow_pickle=True).item()
    beta_glm = glm_dict['betasmd']
    back_mask = nib.load(str(data_paths['brain'])).get_fdata(dtype=np.float32)
    csf_mask = nib.load(str(data_paths['csf'])).get_fdata(dtype=np.float32)
    gray_mask = nib.load(str(data_paths['gray'])).get_fdata(dtype=np.float32)

    print("Apply Masking on Bold data")
    csf_mask_data = csf_mask > 0
    back_mask_data = back_mask > 0
    mask = np.logical_and(back_mask_data, ~csf_mask_data)
    nonzero_mask = np.where(mask)
    gray_mask_data = gray_mask > args.gray_threshold
    keep_voxels = gray_mask_data[nonzero_mask]
    bold_flat = bold_data[nonzero_mask]
    masked_bold = bold_flat[keep_voxels].astype(np.float32)
    masked_coords = tuple(ax[keep_voxels] for ax in nonzero_mask)

    num_runs = 2

    trial_keep = []
    for run_idx in range(num_runs):
        trial_keep.append(_load_trial_keep(data_root, run_idx + 1))

    counts = _trial_counts(beta_glm.shape[-1], num_runs, trial_keep)
    run_index = max(0, run - 1)
    run_index = min(run_index, num_runs - 1)

    start_idx = sum(counts[:run_index])
    end_idx = start_idx + counts[run_index]
    end_idx = min(end_idx, beta_glm.shape[-1])

    beta = beta_glm[:, 0, 0, start_idx:end_idx]

    keep = trial_keep[run_index]
    num_trials = int(keep.shape[0]) if keep is not None else beta.shape[-1]
    bold_data_reshape = _extract_trial_segments(
        masked_bold,
        trial_len=TRIAL_LEN,
        num_trials=num_trials,
        rest_every=30,
        rest_len=20
    )
    if bold_data_reshape.shape[1] > beta.shape[-1]:
        bold_data_reshape = bold_data_reshape[:, : beta.shape[-1], :]

    print(2, flush=True)

    beta = beta[keep_voxels]

    nan_voxels = np.isnan(beta).all(axis=1)
    if np.any(nan_voxels):
        beta = beta[~nan_voxels]
        bold_data_reshape = bold_data_reshape[~nan_voxels]
        masked_coords = tuple(coord[~nan_voxels] for coord in masked_coords)

    print(3, flush=True)

    med = np.nanmedian(beta, keepdims=True)
    mad = np.nanmedian(np.abs(beta - med), keepdims=True)
    scale = 1.4826 * np.maximum(mad, 1e-9)
    beta_norm = (beta - med) / scale
    thr = np.nanpercentile(np.abs(beta_norm), outlier_percentile)
    outlier_mask = np.abs(beta_norm) > thr

    clean_beta = beta.copy()
    voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
    valid_voxels = voxel_outlier_fraction <= max_outlier_fraction
    clean_beta[~valid_voxels] = np.nan
    clean_beta[np.logical_and(outlier_mask, valid_voxels[:, None])] = np.nan
    keeped_mask = ~np.all(np.isnan(clean_beta), axis=1)
    clean_beta = clean_beta[keeped_mask]
    keeped_indices = np.flatnonzero(keeped_mask)

    bold_data_reshape[~valid_voxels, :, :] = np.nan
    trial_outliers = np.logical_and(outlier_mask, valid_voxels[:, None])
    bold_data_reshape = np.where(trial_outliers[:, :, None], np.nan, bold_data_reshape)
    bold_data_reshape = bold_data_reshape[keeped_mask]

    print(4, flush=True)

    if args.skip_ttest:
        clean_active_beta = clean_beta
        clean_active_idx = keeped_indices
        clean_active_bold = bold_data_reshape
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
        clean_active_bold = bold_data_reshape[reject]

    num_trials = clean_active_beta.shape[1]
    clean_active_volume = np.full(bold_data.shape[:3] + (num_trials,), np.nan)
    active_coords = tuple(coord[clean_active_idx] for coord in masked_coords)
    clean_active_volume[active_coords[0], active_coords[1], active_coords[2], :] = clean_active_beta

    print(5, flush=True)

    if args.skip_hampel:
        beta_volume_filter = clean_active_volume.astype(np.float32)
        hampel_stats = {'insufficient_total': 0, 'corrected_total': 0}
    else:
        beta_volume_filter, hampel_stats = hampel_filter_image(
            clean_active_volume.astype(np.float32),
            window_size=hampel_window,
            threshold_factor=hampel_threshold,
            return_stats=True,
        )
    print('Total voxels with <3 neighbours:', hampel_stats['insufficient_total'], flush=True)
    print('Total corrected voxels:', hampel_stats['corrected_total'], flush=True)

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

    clean_active_idx = clean_active_idx[keep_mask]
    active_coords = tuple(coord[keep_mask] for coord in active_coords)
    np.save(output_dir / f"active_coords_sub{sub}_ses{ses}_run{run}.npy", active_coords)

    mean_clean_active = _compute_beta_summary(beta_volume_filter)

    _save_beta_overlay(
        mean_clean_active,
        anat_img=anat_img,
        out_html=str(overlay_html_path),
        threshold_pct=overlay_threshold_pct,
        vmax_pct=overlay_vmax_pct,
        cut_coords=cut_coords,
        snapshot_path=str(overlay_snapshot_path),
    )
    if not skip_roi_ranking:
        roi_rank_path = output_dir / f'roi_{roi_tag}_sub{sub}_ses{ses}_run{run}.csv'
        _rank_rois_by_beta(
            mean_clean_active,
            anat_img=anat_img,
            anat_path=data_paths['anat'],
            ref_img=anat_img,
            out_path=roi_rank_path,
            atlas_threshold=roi_atlas_threshold,
            label_patterns=roi_label_patterns,
            assume_mni=False,
        )


if __name__ == '__main__':
    main()
