# %%
import argparse
import csv
import os
import shutil
import subprocess
import tempfile
from os.path import join
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests
import scipy.ndimage as ndimage
from nilearn import datasets, image, plotting


DEFAULT_DATA_ROOT = Path(os.environ.get('GLM_DATA_ROOT', Path(__file__).resolve().parent / 'GLMsingle'))
DEFAULT_GLMSINGLE_ROOT = Path(os.environ.get('GLM_OUTPUT_ROOT', DEFAULT_DATA_ROOT))
DEFAULT_SUB = os.environ.get('GLM_SUB', '09')
DEFAULT_SES = int(os.environ.get('GLM_SES', '1'))
DEFAULT_RUN = int(os.environ.get('GLM_RUN', '1'))
DEFAULT_TRIAL_LEN = int(os.environ.get('GLM_TRIAL_LEN', '9'))


# %%
def _find_latest_glmsingle_typed_file(root: Path, subject: str, session: int) -> Path:
    ses_int = int(session)
    candidates = [root / f'GLMOutputs-sub{subject}-ses{ses_int}']
    patterns = [f'TYPED_FITHRF_GLMDENOISE_RR.npy']

    matches: list[Path] = []
    for directory in candidates:
        if not directory.exists():
            continue
        for pattern in patterns:
            matches.extend(directory.glob(pattern))

    if not matches:
        raise FileNotFoundError(f'Could not find GLMsingle TYPE-D output under {root} for sub={subject}, ses={session}.')

    return max(matches, key=lambda p: p.stat().st_mtime)

def _infer_csf_exclusion_threshold(brain_mask: np.ndarray, csf_pve: np.ndarray, target_voxels: int) -> float:
    candidates = [0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    counts = {thr: int(np.count_nonzero(brain_mask & ~(csf_pve > thr))) for thr in candidates}

    for thr, count in counts.items():
        if count == target_voxels:
            return float(thr)

    best_thr = min(candidates, key=lambda thr: abs(counts[thr] - target_voxels))
    print(
        f'Warning: could not exactly match GLMsingle voxel count ({target_voxels}). '
        f'Using csf_threshold={best_thr} with mask voxels={counts[best_thr]}.',
        flush=True,
    )
    return float(best_thr)

def _precentral_cut_coords(data_root: Path, subject: str, session: int):
    roi_path = (
        data_root
        / f'sub-pd0{subject}'
        / f'ses-{session}'
        / 'func'
        / 'atlas_files'
        / 'rois'
        / 'precentral.nii.gz'
    )
    if not roi_path.exists():
        return None

    roi_img = nib.load(str(roi_path))
    roi_data = roi_img.get_fdata() > 0
    if not np.any(roi_data):
        return None

    coords_vox = np.array(np.where(roi_data)).T
    centroid_vox = coords_vox.mean(axis=0)
    centroid_world = (roi_img.affine @ np.r_[centroid_vox, 1])[:3]
    return tuple(float(x) for x in centroid_world)

def _resolve_first_match(root: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f'No files matched patterns under {root}: {patterns}')

def _resolve_data_paths(data_root: Path, subject: str, session: int, run: int) -> dict:
    sub_label = f'sub-pd0{subject}'
    ses_label = f'ses-{session}'
    derivatives_dir = data_root / sub_label / ses_label

    bold_name = f'{sub_label}_ses-{session}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'
    brain_mask_name = f'{sub_label}_ses-{session}_T1w_brain_mask.nii.gz'
    csf_name = f'{sub_label}_ses-{session}_T1w_brain_pve_0.nii.gz'
    gray_name = f'{sub_label}_ses-{session}_T1w_brain_pve_1.nii.gz'
    anat_name = f'{sub_label}_ses-{session}_T1w_brain.nii.gz'

    if derivatives_dir.exists():
        anat_dir = derivatives_dir / 'anat'
        func_dir = derivatives_dir / 'func'
        bold_path = func_dir / bold_name
        if bold_path.exists():
            return {
                'bold': bold_path,
                'brain': anat_dir / brain_mask_name,
                'csf': anat_dir / csf_name,
                'gray': anat_dir / gray_name,
                'anat': anat_dir / anat_name,
            }

    patterns = {
        'bold': [bold_name, f'{sub_label}_ses-{session:02d}_run-{run}_task-mv_bold_corrected_smoothed_reg.nii.gz'],
        'brain': [brain_mask_name, f'{sub_label}_ses-{session:02d}_T1w_brain_mask.nii.gz'],
        'csf': [csf_name, f'{sub_label}_ses-{session:02d}_T1w_brain_pve_0.nii.gz'],
        'gray': [gray_name, f'{sub_label}_ses-{session:02d}_T1w_brain_pve_1.nii.gz'],
        'anat': [anat_name, f'{sub_label}_ses-{session:02d}_T1w_brain.nii.gz'],
    }

    return {key: _resolve_first_match(data_root, pats) for key, pats in patterns.items()}

def _resolve_go_times(data_root: Path, subject: str, session: int) -> Path | None:
    candidates = [
        data_root / f'PSPD0{subject}-ses-{session}-go-times.txt',
        data_root / f'PSPD0{subject}-ses-{session:02d}-go-times.txt',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None

def _load_go_times(path: Path | None) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    arr = np.loadtxt(path, dtype=int)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    return arr

def _load_trial_keep(root: Path, run: int) -> np.ndarray | None:
    if root is None:
        return None
    for pattern in (f'trial_keep_run{run}.npy', f'trial_keep_run{run:02d}.npy'):
        path = root / pattern
        if path.exists():
            return np.load(path)
    return None

def _trial_counts(total_trials: int, num_runs: int, trial_keep: list[np.ndarray | None], go_times: np.ndarray | None):
    counts = []
    for run_idx in range(num_runs):
        keep = trial_keep[run_idx] if run_idx < len(trial_keep) else None
        if keep is not None:
            counts.append(int(np.count_nonzero(keep)))
            continue
        if go_times is not None and run_idx < go_times.shape[0]:
            counts.append(int(go_times.shape[1]))
            continue
        counts.append(None)

    if all(count is not None for count in counts):
        if sum(counts) == total_trials:
            return counts

    base = total_trials // num_runs
    counts = [base] * num_runs
    counts[-1] = total_trials - base * (num_runs - 1)
    return counts

def _extract_trial_segments(masked_bold: np.ndarray, onsets: np.ndarray, trial_len: int) -> np.ndarray:
    num_voxels, num_timepoints = masked_bold.shape
    segments = np.full((num_voxels, len(onsets), trial_len), np.nan, dtype=np.float32)
    truncated = 0
    for i, onset in enumerate(onsets):
        start = int(onset) - 1
        end = start + trial_len
        if end > num_timepoints:
            truncated += 1
            if start < num_timepoints:
                available = masked_bold[:, start:num_timepoints]
                segments[:, i, : available.shape[1]] = available
            continue
        segments[:, i, :] = masked_bold[:, start:end]
    if truncated:
        print(f'Warning: truncated {truncated} trial segments beyond BOLD length.', flush=True)
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


def _compute_beta_summary(
    beta_volume_filter: np.ndarray,
    overlay_stat: str,
    overlay_positive_only: bool,
) -> np.ndarray:
    with np.errstate(invalid='ignore'):
        if overlay_stat == 'mean_abs':
            summary = np.nanmean(np.abs(beta_volume_filter), axis=-1)
        elif overlay_stat == 'mean':
            summary = np.nanmean(beta_volume_filter, axis=-1)
        else:
            mean_beta = np.nanmean(beta_volume_filter, axis=-1)
            finite = mean_beta[np.isfinite(mean_beta)]
            scale = float(np.nanstd(finite)) if finite.size else 0.0
            if scale <= 0:
                scale = 1.0
            summary = (mean_beta - float(np.nanmean(finite)) if finite.size else mean_beta) / scale

        if overlay_positive_only:
            summary = np.where(summary > 0, summary, np.nan)

    return summary.astype(np.float32)


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
    mni_template=None,
):
    use_flirt = False
    flirt_path = None
    mni_template_img = None
    if not assume_mni:
        flirt_path = _find_flirt()
        mni_template = mni_template or _default_mni_template()
        if flirt_path and mni_template and Path(mni_template).exists():
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
    mni_template: Path | None,
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
        mni_template=mni_template,
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
    parser.add_argument('--subject', default=DEFAULT_SUB)
    parser.add_argument('--session', type=int, default=DEFAULT_SES)
    parser.add_argument('--run', type=int, default=DEFAULT_RUN)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--glmsingle-root', type=Path, default=DEFAULT_GLMSINGLE_ROOT)
    parser.add_argument('--typed-path', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--trial-len', type=int, default=DEFAULT_TRIAL_LEN)
    parser.add_argument('--gray-threshold', type=float, default=0.5)
    parser.add_argument('--apply-gray-mask', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--mask-mode', default='brain_minus_csf', choices=('brain_minus_csf', 'brain_only', 'gray_only', 'gray_minus_csf'))
    parser.add_argument('--csf-threshold', type=float, default=None)
    parser.add_argument('--fdr-alpha', type=float, default=0.05)
    parser.add_argument('--skip-ttest', action='store_true')
    parser.add_argument('--outlier-percentile', type=float, default=99.9)
    parser.add_argument('--max-outlier-fraction', type=float, default=0.5)
    parser.add_argument('--skip-hampel', action='store_true')
    parser.add_argument('--hampel-window', type=int, default=5)
    parser.add_argument('--hampel-threshold', type=float, default=3.0)
    parser.add_argument('--overlay-threshold-pct', type=float, default=90.0)
    parser.add_argument('--overlay-vmax-pct', type=float, default=99.0)
    parser.add_argument('--overlay-stat', default='mean_abs', choices=('mean_abs', 'mean', 'mean_z'), help='Summary statistic for overlay (default: mean_abs).')
    parser.add_argument('--overlay-positive-only', action='store_true', help='Zero out non-positive overlay values before thresholding.')
    parser.add_argument('--cut-coords', type=float, nargs=3, default=None)
    parser.add_argument('--snapshot-path', type=Path, default=None)
    parser.add_argument('--skip-roi-ranking', action='store_true', help='Skip atlas-based ROI ranking.')
    parser.add_argument('--roi-atlas-threshold', type=int, default=25, help='Harvard-Oxford atlas threshold (percent).')
    parser.add_argument('--roi-label-patterns', default=None, help='Comma-separated label patterns to include (optional).')
    parser.add_argument('--roi-assume-mni', action='store_true', help='Assume anatomy is in MNI space for atlas alignment.')
    parser.add_argument('--roi-mni-template', type=Path, default=None, help='Optional MNI template for atlas registration.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--go-times', type=Path, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    sub = args.subject
    ses = args.session
    run = args.run

    data_root = args.data_root.expanduser().resolve()
    glmsingle_root = args.glmsingle_root.expanduser().resolve()

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(f'sub{sub}')
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cached_beta_path = output_dir / f'cleaned_beta_volume_sub{sub}_ses{ses}_run{run}.npy'

    data_paths = _resolve_data_paths(data_root, sub, ses, run)
    anat_img = nib.load(str(data_paths['anat']))
    bold_img = nib.load(str(data_paths['bold']))

    roi_ref_img = anat_img
    if bold_img.shape[:3] != anat_img.shape[:3] or not np.allclose(bold_img.affine, anat_img.affine):
        roi_ref_img = bold_img
        print("Warning: anatomy and BOLD grids differ; ROI ranking will use BOLD grid.", flush=True)

    cut_coords = tuple(args.cut_coords) if args.cut_coords else _precentral_cut_coords(data_root, sub, ses)
    roi_tag = args.overlay_stat + ("_pos" if args.overlay_positive_only else "")

    if cached_beta_path.exists() and not args.force:
        beta_volume_filter = np.load(cached_beta_path)
        mean_clean_active = _compute_beta_summary(
            beta_volume_filter,
            overlay_stat=args.overlay_stat,
            overlay_positive_only=args.overlay_positive_only,
        )
        _save_beta_overlay(
            mean_clean_active,
            anat_img=anat_img,
            out_html=str(output_dir / f'clean_active_beta_overlay_sub{sub}_ses{ses}_run{run}.html'),
            threshold_pct=args.overlay_threshold_pct,
            vmax_pct=args.overlay_vmax_pct,
            cut_coords=cut_coords,
            snapshot_path=str(args.snapshot_path) if args.snapshot_path else None)
        if not args.skip_roi_ranking:
            roi_rank_path = output_dir / f'roi_{roi_tag}_sub{sub}_ses{ses}_run{run}.csv'
            _rank_rois_by_beta(
                mean_clean_active,
                anat_img=anat_img,
                anat_path=data_paths['anat'],
                ref_img=roi_ref_img,
                out_path=roi_rank_path,
                atlas_threshold=args.roi_atlas_threshold,
                label_patterns=args.roi_label_patterns,
                assume_mni=args.roi_assume_mni,
                mni_template=args.roi_mni_template,
            )
        return

    bold_data = bold_img.get_fdata()

    back_mask = nib.load(str(data_paths['brain'])).get_fdata(dtype=np.float32)
    csf_mask = nib.load(str(data_paths['csf'])).get_fdata(dtype=np.float32)
    gray_mask = nib.load(str(data_paths['gray'])).get_fdata(dtype=np.float32)

    print(1, flush=True)

    back_mask_data = back_mask > 0
    gray_mask_data = gray_mask > args.gray_threshold

    typed_path = args.typed_path
    if typed_path is None:
        typed_path = _find_latest_glmsingle_typed_file(glmsingle_root, sub, ses)
    print(f'Loading GLMsingle output: {typed_path}', flush=True)
    glm_dict = np.load(str(typed_path), allow_pickle=True).item()
    beta_glm = glm_dict['betasmd']
    target_voxels = int(beta_glm.shape[0])

    if args.csf_threshold is None:
        csf_thr = _infer_csf_exclusion_threshold(back_mask_data, csf_mask, target_voxels)
    else:
        csf_thr = float(args.csf_threshold)
    print(f'Using csf_threshold={csf_thr} (target voxels={target_voxels})', flush=True)

    if args.mask_mode == 'brain_minus_csf':
        mask = np.logical_and(back_mask_data, ~(csf_mask > csf_thr))
    elif args.mask_mode == 'brain_only':
        mask = back_mask_data
    elif args.mask_mode == 'gray_only':
        mask = gray_mask_data
    else:
        mask = np.logical_and(gray_mask_data, ~(csf_mask > csf_thr))

    nonzero_mask = np.where(mask)
    if args.apply_gray_mask and args.mask_mode not in ('gray_only', 'gray_minus_csf'):
        keep_voxels = gray_mask_data[nonzero_mask]
    else:
        keep_voxels = np.ones(nonzero_mask[0].shape[0], dtype=bool)

    bold_flat = bold_data[nonzero_mask]
    masked_bold = bold_flat[keep_voxels]
    masked_coords = tuple(ax[keep_voxels] for ax in nonzero_mask)

    masked_bold = masked_bold.astype(np.float32)

    go_times_path = args.go_times
    if go_times_path is None:
        go_times_path = _resolve_go_times(data_root, sub, ses)
    go_times = _load_go_times(go_times_path)
    num_runs = go_times.shape[0] if go_times is not None else 2

    trial_keep = []
    for run_idx in range(num_runs):
        trial_keep.append(_load_trial_keep(Path(typed_path).parent, run_idx + 1))

    counts = _trial_counts(beta_glm.shape[-1], num_runs, trial_keep, go_times)
    run_index = max(0, run - 1)
    run_index = min(run_index, num_runs - 1)

    start_idx = sum(counts[:run_index])
    end_idx = start_idx + counts[run_index]
    end_idx = min(end_idx, beta_glm.shape[-1])

    beta = beta_glm[:, 0, 0, start_idx:end_idx]

    if go_times is not None:
        run_onsets = go_times[run_index]
        keep = trial_keep[run_index]
        if keep is not None:
            run_onsets = run_onsets[keep]
        if len(run_onsets) > beta.shape[-1]:
            run_onsets = run_onsets[: beta.shape[-1]]
        bold_data_reshape = _extract_trial_segments(masked_bold, run_onsets, args.trial_len)
    else:
        num_trials = beta.shape[-1]
        num_voxels, num_timepoints = masked_bold.shape
        bold_data_reshape = np.full((num_voxels, num_trials, args.trial_len), np.nan, dtype=np.float32)

        start = 0
        for i in range(num_trials):
            end = start + args.trial_len
            if end > num_timepoints:
                raise ValueError("Masked BOLD data does not contain enough timepoints for all trials")
            bold_data_reshape[:, i, :] = masked_bold[:, start:end]
            start += args.trial_len
            if start in (270, 560):
                start += 20

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
    thr = np.nanpercentile(np.abs(beta_norm), args.outlier_percentile)
    outlier_mask = np.abs(beta_norm) > thr

    clean_beta = beta.copy()
    voxel_outlier_fraction = np.mean(outlier_mask, axis=1)
    valid_voxels = voxel_outlier_fraction <= args.max_outlier_fraction
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
        rej, q, _, _ = multipletests(pvals[tested], alpha=args.fdr_alpha, method='fdr_bh')

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
            window_size=args.hampel_window,
            threshold_factor=args.hampel_threshold,
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

    mean_clean_active = _compute_beta_summary(
        beta_volume_filter,
        overlay_stat=args.overlay_stat,
        overlay_positive_only=args.overlay_positive_only,
    )

    _save_beta_overlay(
        mean_clean_active,
        anat_img=anat_img,
        out_html=str(output_dir / f'clean_active_beta_overlay_sub{sub}_ses{ses}_run{run}.html'),
        threshold_pct=args.overlay_threshold_pct,
        vmax_pct=args.overlay_vmax_pct,
        cut_coords=cut_coords,
        snapshot_path=str(args.snapshot_path) if args.snapshot_path else None,
    )
    if not args.skip_roi_ranking:
        roi_rank_path = output_dir / f'roi_{roi_tag}_sub{sub}_ses{ses}_run{run}.csv'
        _rank_rois_by_beta(
            mean_clean_active,
            anat_img=anat_img,
            anat_path=data_paths['anat'],
            ref_img=roi_ref_img,
            out_path=roi_rank_path,
            atlas_threshold=args.roi_atlas_threshold,
            label_patterns=args.roi_label_patterns,
            assume_mni=args.roi_assume_mni,
            mni_template=args.roi_mni_template,
        )


if __name__ == '__main__':
    main()
