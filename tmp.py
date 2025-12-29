#!/usr/bin/env python3
import csv
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from nilearn import datasets, plotting

from Beta_preprocessing import TRIALS_PER_RUN, _align_atlas_to_reference, _rank_rois_by_beta


def _load_roi_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                label_index = int(row["label_index"])
            except (KeyError, TypeError, ValueError):
                continue
            if label_index <= 0:
                continue
            label = row.get("label") or f"ROI_{label_index}"
            rank_raw = row.get("rank", "")
            rank = None
            if rank_raw and str(rank_raw).strip().isdigit():
                rank = int(rank_raw)
            rows.append({"rank": rank, "label_index": label_index, "label": label})

    if not rows:
        return rows

    if any(row["rank"] is not None for row in rows):
        rows = sorted(rows, key=lambda r: (r["rank"] is None, r["rank"] or 0))
        next_rank = 1
        for row in rows:
            if row["rank"] is None:
                row["rank"] = next_rank
            next_rank = max(next_rank + 1, row["rank"] + 1)
    else:
        for idx, row in enumerate(rows, start=1):
            row["rank"] = idx
    return rows


def _make_colormap(max_rank: int):
    if max_rank <= 20:
        base = plt.get_cmap("tab20", max_rank)
        colors = [base(i) for i in range(max_rank)]
    else:
        base = plt.get_cmap("gist_ncar", max_rank)
        colors = [base(i) for i in range(max_rank)]
    colors = [(0.0, 0.0, 0.0, 0.0)] + colors
    return ListedColormap(colors, name="roi_rank_cmap")


def _run_fsl(cmd):
    env = os.environ.copy()
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")
    subprocess.run(cmd, check=True, env=env)


def _atlas_overlap_ratio(atlas_data, brain_mask):
    atlas_mask = atlas_data > 0
    total = int(np.count_nonzero(atlas_mask))
    if total == 0:
        return 0.0
    inside = int(np.count_nonzero(atlas_mask & brain_mask))
    return inside / total


def _find_mni_resources():
    fsl_dir = Path(os.environ.get("FSLDIR", "/usr/local/fsl")).expanduser().resolve()
    template = fsl_dir / "data" / "standard" / "MNI152_T1_2mm_brain.nii.gz"
    if not template.exists():
        template = fsl_dir / "data" / "standard" / "MNI152_T1_2mm.nii.gz"
    ref_mask = fsl_dir / "data" / "standard" / "MNI152_T1_2mm_brain_mask_dil.nii.gz"
    config = fsl_dir / "etc" / "flirtsch" / "T1_2_MNI152_2mm.cnf"
    if not template.exists() or not config.exists():
        return None, None, None
    if not ref_mask.exists():
        ref_mask = None
    return template, ref_mask, config


def _align_atlas_with_fnirt(atlas_path, anat_path, brain_mask_path, out_dir):
    flirt = shutil.which("flirt")
    fnirt = shutil.which("fnirt")
    invwarp = shutil.which("invwarp")
    applywarp = shutil.which("applywarp")
    if not all([flirt, fnirt, invwarp, applywarp]):
        return None

    mni_template, ref_mask, config = _find_mni_resources()
    if not mni_template:
        return None

    out_dir = Path(out_dir)
    anat_to_mni_mat = out_dir / "anat_to_mni_affine.mat"
    anat_in_mni = out_dir / "anat_in_mni_affine.nii.gz"
    anat_to_mni_warp = out_dir / "anat_to_mni_fnirt.nii.gz"
    mni_to_anat_warp = out_dir / "mni_to_anat_fnirt.nii.gz"
    atlas_in_anat = out_dir / "atlas_in_anat_fnirt.nii.gz"

    cmd = [
        flirt,
        "-in",
        str(anat_path),
        "-ref",
        str(mni_template),
        "-omat",
        str(anat_to_mni_mat),
        "-out",
        str(anat_in_mni),
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
    _run_fsl(cmd)

    cmd = [
        fnirt,
        f"--in={anat_path}",
        f"--ref={mni_template}",
        f"--aff={anat_to_mni_mat}",
        f"--cout={anat_to_mni_warp}",
        f"--config={config}",
    ]
    if ref_mask is not None:
        cmd.append(f"--refmask={ref_mask}")
    if brain_mask_path and Path(brain_mask_path).exists():
        cmd.append(f"--inmask={brain_mask_path}")
    _run_fsl(cmd)

    cmd = [invwarp, f"--warp={anat_to_mni_warp}", f"--ref={anat_path}", f"--out={mni_to_anat_warp}"]
    _run_fsl(cmd)

    cmd = [
        applywarp,
        f"--in={atlas_path}",
        f"--ref={anat_path}",
        f"--warp={mni_to_anat_warp}",
        "--interp=nn",
        f"--out={atlas_in_anat}",
    ]
    _run_fsl(cmd)

    return nib.load(str(atlas_in_anat))


def _compute_mean_abs_summary(data_root: Path, sub: str, ses: str, run: int, gray_threshold: float):
    sub_label = f"sub-pd0{sub}"
    glm_path = data_root / "TYPED_FITHRF_GLMDENOISE_RR.npy"
    if not glm_path.exists():
        raise FileNotFoundError(f"Missing GLMsingle file: {glm_path}")

    brain_path = data_root / f"{sub_label}_ses-{ses}_T1w_brain_mask.nii.gz"
    csf_path = data_root / f"{sub_label}_ses-{ses}_T1w_brain_pve_0.nii.gz"
    gray_path = data_root / f"{sub_label}_ses-{ses}_T1w_brain_pve_1.nii.gz"

    brain_mask = nib.load(str(brain_path)).get_fdata(dtype=np.float32) > 0
    csf_mask = nib.load(str(csf_path)).get_fdata(dtype=np.float32) > 0
    gray_mask = nib.load(str(gray_path)).get_fdata(dtype=np.float32)

    mask = np.logical_and(brain_mask, ~csf_mask)
    nonzero_mask = np.where(mask)
    gray_keep = gray_mask[nonzero_mask] > gray_threshold

    glm_dict = np.load(str(glm_path), allow_pickle=True).item()
    beta_glm = glm_dict["betasmd"]
    start_idx = (run - 1) * TRIALS_PER_RUN
    end_idx = start_idx + TRIALS_PER_RUN
    beta = beta_glm[:, 0, 0, start_idx:end_idx]
    beta = beta[gray_keep]

    with np.errstate(invalid="ignore"):
        mean_abs = np.nanmean(np.abs(beta), axis=1).astype(np.float32)

    volume_shape = brain_mask.shape
    summary = np.full(volume_shape, np.nan, dtype=np.float32)
    coords = tuple(ax[gray_keep] for ax in nonzero_mask)
    summary[coords] = mean_abs
    return summary


def main():
    fsl_dir = Path(os.environ.get("FSLDIR", "/usr/local/fsl")).expanduser().resolve()
    os.environ.setdefault("NILEARN_DATA", str(fsl_dir.parent))

    root = Path.cwd()
    data_root = root / "sub09_ses1"
    sub = "09"
    ses = "1"
    run = 1
    gray_threshold = 0.5
    atlas_threshold = 25

    csv_path = root / "roi_mean_abs_sub09_ses1_run1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing ROI CSV file: {csv_path}")

    out_dir = root / "tmp_roi_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_label = f"sub-pd0{sub}"
    anat_path = data_root / f"{sub_label}_ses-{ses}_T1w_brain.nii.gz"
    anat_img = nib.load(str(anat_path))
    brain_mask_path = data_root / f"{sub_label}_ses-{ses}_T1w_brain_mask.nii.gz"
    brain_mask = nib.load(str(brain_mask_path)).get_fdata(dtype=np.float32) > 0

    print("Computing mean |beta| summary for ROI ranking...", flush=True)
    summary = _compute_mean_abs_summary(data_root, sub=sub, ses=ses, run=run, gray_threshold=gray_threshold)
    rank_csv_path = out_dir / f"roi_rank_sub{sub}_ses{ses}_run{run}.csv"
    _rank_rois_by_beta(
        summary,
        anat_img=anat_img,
        anat_path=anat_path,
        ref_img=anat_img,
        out_path=rank_csv_path,
        atlas_threshold=atlas_threshold,
        label_patterns=None,
        assume_mni=False,
    )

    print("Building ROI overlay from CSV labels...", flush=True)
    roi_rows = _load_roi_rows(csv_path)
    if not roi_rows:
        raise RuntimeError(f"No ROI labels found in {csv_path}")

    atlas = datasets.fetch_atlas_harvard_oxford(
        f"cort-maxprob-thr{atlas_threshold}-2mm",
        data_dir=str(fsl_dir.parent),
    )
    atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
    atlas_candidates = []
    atlas_in_ref = _align_atlas_to_reference(
        atlas_img,
        anat_img=anat_img,
        anat_path=anat_path,
        ref_img=anat_img,
        out_dir=out_dir,
        assume_mni=False,
    )
    atlas_data = np.rint(atlas_in_ref.get_fdata(dtype=np.float32)).astype(int)
    atlas_candidates.append(("flirt", atlas_in_ref, _atlas_overlap_ratio(atlas_data, brain_mask)))

    try:
        fnirt_img = _align_atlas_with_fnirt(
            atlas_path=atlas.filename,
            anat_path=anat_path,
            brain_mask_path=brain_mask_path,
            out_dir=out_dir,
        )
    except subprocess.CalledProcessError as exc:
        fnirt_img = None
        print(f"Warning: FNIRT alignment failed ({exc}); using FLIRT.", flush=True)

    if fnirt_img is not None:
        fnirt_data = np.rint(fnirt_img.get_fdata(dtype=np.float32)).astype(int)
        atlas_candidates.append(("fnirt", fnirt_img, _atlas_overlap_ratio(fnirt_data, brain_mask)))

    best_name, atlas_in_ref, best_ratio = max(atlas_candidates, key=lambda item: item[2])
    print(f"Using {best_name} atlas alignment (brain overlap {best_ratio:.3f}).", flush=True)
    atlas_data = np.rint(atlas_in_ref.get_fdata(dtype=np.float32)).astype(int)

    roi_map = np.zeros(atlas_data.shape, dtype=np.int16)
    for row in roi_rows:
        roi_map[atlas_data == row["label_index"]] = row["rank"]
    roi_map[~brain_mask] = 0

    max_rank = max(row["rank"] for row in roi_rows)
    cmap = _make_colormap(max_rank)
    roi_img = nib.Nifti1Image(roi_map, anat_img.affine, anat_img.header)

    overlay_html = out_dir / f"roi_overlay_sub{sub}_ses{ses}_run{run}.html"
    view = plotting.view_img(
        roi_img,
        bg_img=anat_img,
        cmap=cmap,
        threshold=0.5,
        colorbar=False,
        symmetric_cmap=False,
        resampling_interpolation="nearest",
        vmax=max_rank,
        vmin=0,
        opacity=0.7,
    )
    view.save_as_html(str(overlay_html))

    overlay_png = out_dir / f"roi_overlay_sub{sub}_ses{ses}_run{run}.png"
    display = plotting.plot_roi(
        roi_img,
        bg_img=anat_img,
        cmap=cmap,
        alpha=0.7,
        threshold=0.5,
        colorbar=False,
        draw_cross=False,
        resampling_interpolation="nearest",
        vmax=max_rank,
        vmin=0,
    )
    display.savefig(str(overlay_png))
    display.close()

    legend_path = out_dir / f"roi_overlay_legend_sub{sub}_ses{ses}_run{run}.csv"
    with open(legend_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "label_index", "label"])
        for row in roi_rows:
            writer.writerow([row["rank"], row["label_index"], row["label"]])

    print(f"Saved overlay: {overlay_html}", flush=True)
    print(f"Saved snapshot: {overlay_png}", flush=True)
    print(f"Saved legend: {legend_path}", flush=True)
    print(f"Saved ROI ranking: {rank_csv_path}", flush=True)


if __name__ == "__main__":
    main()
