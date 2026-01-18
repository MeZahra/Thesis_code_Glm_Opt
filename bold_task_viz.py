#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
GLMSINGLE_ROOT = REPO_ROOT / "GLMsingle"
if str(GLMSINGLE_ROOT) not in sys.path:
    sys.path.insert(0, str(GLMSINGLE_ROOT))

from glmsingle.design.convolve_design import convolve_design


def _parse_runs(runs_csv: str) -> list[int]:
    runs = []
    for item in runs_csv.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            runs.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid run value: {item!r}") from exc
    if not runs:
        raise ValueError("No runs specified.")
    return runs


def _load_design(design_path: Path):
    designinfo = np.load(design_path, allow_pickle=True).item()
    if "designSINGLE" not in designinfo:
        raise KeyError("DESIGNINFO missing designSINGLE.")
    if "params" not in designinfo:
        raise KeyError("DESIGNINFO missing params.")
    params = designinfo["params"]
    if "hrflibrary" not in params:
        raise KeyError("DESIGNINFO params missing hrflibrary.")
    return designinfo["designSINGLE"], params["hrflibrary"]


def _load_model(model_path: Path):
    model = np.load(model_path, allow_pickle=True).item()
    if "betasmd" not in model:
        raise KeyError("Model missing betasmd.")
    if "HRFindexrun" not in model:
        raise KeyError("Model missing HRFindexrun.")
    return model["betasmd"], model["HRFindexrun"]


def _flatten_betas(betasmd: np.ndarray) -> tuple[np.ndarray, int]:
    if betasmd.ndim < 2:
        raise ValueError(f"Unexpected betasmd shape: {betasmd.shape}")
    numtrials = betasmd.shape[-1]
    voxels = int(np.prod(betasmd.shape[:-1]))
    betas = betasmd.reshape((voxels, numtrials)).astype(np.float32, copy=False)
    return betas, numtrials


def _flatten_hrfindex(hrfindexrun: np.ndarray, numruns: int) -> np.ndarray:
    if hrfindexrun.ndim < 2:
        raise ValueError(f"Unexpected HRFindexrun shape: {hrfindexrun.shape}")
    if hrfindexrun.shape[-1] != numruns:
        raise ValueError(
            f"HRFindexrun last dim ({hrfindexrun.shape[-1]}) "
            f"does not match numruns ({numruns})."
        )
    voxels = int(np.prod(hrfindexrun.shape[:-1]))
    return hrfindexrun.reshape((voxels, numruns)).astype(np.int64, copy=False)


def _convolve_by_hrf(design_single: np.ndarray, hrflibrary: np.ndarray) -> list[np.ndarray]:
    num_hrf = hrflibrary.shape[1]
    conv = []
    for h in range(num_hrf):
        conv_h = convolve_design(design_single, hrflibrary[:, h]).astype(np.float32)
        conv.append(conv_h)
    return conv


def _write_task_prediction(
    betas: np.ndarray,
    hrf_idx_run: np.ndarray,
    conv_by_hrf: list[np.ndarray],
    out_path: Path,
    chunk_size: int,
):
    voxels, numtrials = betas.shape
    ntime = conv_by_hrf[0].shape[0]
    pred = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(voxels, ntime)
    )

    for h, conv in enumerate(conv_by_hrf):
        voxel_idx = np.flatnonzero(hrf_idx_run == h)
        if voxel_idx.size == 0:
            continue
        conv_t = conv.T
        for start in range(0, voxel_idx.size, chunk_size):
            chunk = voxel_idx[start : start + chunk_size]
            pred[chunk, :] = betas[chunk, :] @ conv_t

    pred.flush()


def _parse_int_list(values_csv: str, allow_empty: bool, label: str) -> list[int]:
    values = []
    for item in values_csv.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(int(item))
        except ValueError as exc:
            raise ValueError(f"Invalid {label} value: {item!r}") from exc
    if not values and not allow_empty:
        raise ValueError(f"No {label} specified.")
    return values


def _is_nifti(path: Path) -> bool:
    if path.suffix == ".nii":
        return True
    suffixes = path.suffixes
    return len(suffixes) >= 2 and suffixes[-2:] == [".nii", ".gz"]


def _load_numpy_or_nifti(path: Path) -> np.ndarray:
    if _is_nifti(path):
        img = nib.load(str(path))
        return img.get_fdata(dtype=np.float32)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    raise ValueError(f"Unsupported file type: {path}")


def _load_mask(
    mask_path: Path,
    expected_size: Optional[int],
    expected_shape: Optional[tuple[int, int, int]],
) -> np.ndarray:
    mask_data = _load_numpy_or_nifti(mask_path)
    mask = np.asarray(mask_data)
    if mask.ndim == 3:
        if expected_shape is not None and mask.shape != expected_shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match expected {expected_shape}."
            )
        mask_flat = mask.reshape(-1).astype(bool)
    elif mask.ndim == 1:
        mask_flat = mask.astype(bool)
    else:
        raise ValueError(f"Mask must be 1D or 3D, got shape {mask.shape}.")
    if expected_size is not None and mask_flat.size != expected_size:
        raise ValueError(
            f"Mask has {mask_flat.size} voxels, expected {expected_size}."
        )
    if not np.any(mask_flat):
        raise ValueError("Mask has no nonzero voxels.")
    return mask_flat


def _load_bold_data(
    bold_path: Path,
    mask_path: Optional[Path],
    expected_timepoints: Optional[int],
) -> np.ndarray:
    bold_data = _load_numpy_or_nifti(bold_path)
    bold = np.asarray(bold_data)
    if bold.ndim == 2:
        if expected_timepoints is not None:
            if bold.shape[1] != expected_timepoints and bold.shape[0] == expected_timepoints:
                bold = bold.T
                print("Transposed BOLD array to (voxels, timepoints) layout.", flush=True)
        if mask_path is not None:
            mask_flat = _load_mask(mask_path, expected_size=bold.shape[0], expected_shape=None)
            bold = bold[mask_flat]
        return bold.astype(np.float32, copy=False)
    if bold.ndim == 4:
        volume_shape = bold.shape[:3]
        timepoints = bold.shape[3]
        bold_2d = bold.reshape(-1, timepoints)
        if mask_path is not None:
            mask_flat = _load_mask(
                mask_path, expected_size=bold_2d.shape[0], expected_shape=volume_shape
            )
            bold_2d = bold_2d[mask_flat]
        return bold_2d.astype(np.float32, copy=False)
    raise ValueError(f"BOLD data must be 2D or 4D, got shape {bold.shape}.")


def _extract_trial_segments(
    bold_data: np.ndarray,
    trial_len: int,
    num_trials: int,
    rest_after: list[int],
    rest_len: int,
) -> np.ndarray:
    num_voxels, num_timepoints = bold_data.shape
    segments = np.full((num_voxels, num_trials, trial_len), np.nan, dtype=np.float32)
    start = 0
    rest_after_set = set(rest_after)
    for trial_idx in range(num_trials):
        end = start + trial_len
        if end > num_timepoints:
            raise ValueError(
                f"Not enough timepoints for trial {trial_idx + 1}: need {end}, have {num_timepoints}."
            )
        segments[:, trial_idx, :] = bold_data[:, start:end]
        start = end
        if (trial_idx + 1) in rest_after_set:
            start += rest_len
    if start > num_timepoints:
        raise ValueError(
            f"Rest blocks exceed data length: ended at {start}, but data has {num_timepoints}."
        )
    return segments


def _offset_step(trials: np.ndarray) -> float:
    finite = trials[np.isfinite(trials)]
    if finite.size == 0:
        return 1.0
    p05, p95 = np.percentile(finite, [5, 95])
    span = float(p95 - p05)
    if span <= 0:
        span = float(np.max(finite) - np.min(finite))
    if span <= 0:
        span = 1.0
    return span * 1.2


def _plot_single_voxel_trials(
    trials: np.ndarray,
    voxel_idx: int,
    out_path: Path,
    tr: float,
):
    num_trials, trial_len = trials.shape
    timepoints = np.arange(1, trial_len + 1, dtype=np.int64)
    step = _offset_step(trials)
    fig_height = max(6.0, num_trials * 0.12)
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    for idx in range(num_trials):
        ax.plot(
            timepoints,
            trials[idx] + step * idx,
            color="black",
            linewidth=0.6,
            alpha=0.7,
        )
    ax.set_title(f"Voxel {voxel_idx} trials (n={num_trials})")
    ax.set_xlabel(f"Timepoints (TR={tr:g}s)")
    ax.set_ylabel("Trial (offset)")
    ax.set_xticks(timepoints)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_variance_heatmap(
    variance: np.ndarray,
    voxel_indices: np.ndarray,
    out_path: Path,
):
    data = variance[voxel_indices, :]
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Timepoints")
    ax.set_ylabel("Voxel")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(np.arange(1, data.shape[1] + 1))
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(voxel_indices)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Variance across trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _run_trial_viz(args):
    bold_path = Path(args.bold_path)
    if not bold_path.is_absolute():
        bold_path = (REPO_ROOT / bold_path).resolve()
    if not bold_path.exists():
        raise FileNotFoundError(f"BOLD data not found: {bold_path}")

    mask_path = None
    if args.mask_path:
        mask_path = Path(args.mask_path)
        if not mask_path.is_absolute():
            mask_path = (REPO_ROOT / mask_path).resolve()
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.trial_len <= 0 or args.num_trials <= 0:
        raise ValueError("trial_len and num_trials must be positive.")
    if args.rest_len < 0:
        raise ValueError("rest_len must be >= 0.")

    rest_after = _parse_int_list(args.rest_after, allow_empty=True, label="rest-after")
    rest_after = sorted(set(rest_after))
    for trial_idx in rest_after:
        if trial_idx <= 0 or trial_idx >= args.num_trials:
            raise ValueError(
                f"rest-after entries must be between 1 and {args.num_trials - 1}, got {trial_idx}."
            )

    expected_timepoints = args.num_trials * args.trial_len + args.rest_len * len(rest_after)
    bold_data = _load_bold_data(bold_path, mask_path, expected_timepoints)
    num_voxels, num_timepoints = bold_data.shape
    if num_timepoints != expected_timepoints:
        raise ValueError(
            f"BOLD data has {num_timepoints} timepoints, expected {expected_timepoints} "
            f"(trials={args.num_trials}, trial_len={args.trial_len}, "
            f"rest_after={rest_after}, rest_len={args.rest_len})."
        )
    print(f"Loaded BOLD data: {bold_data.shape}", flush=True)

    trials = _extract_trial_segments(
        bold_data,
        trial_len=args.trial_len,
        num_trials=args.num_trials,
        rest_after=rest_after,
        rest_len=args.rest_len,
    )
    print(f"Reshaped trials: {trials.shape}", flush=True)

    trial_variance = np.nanvar(trials, axis=1).astype(np.float32, copy=False)
    variance_path = Path(args.variance_out)
    if not variance_path.is_absolute():
        variance_path = out_dir / variance_path
    np.save(variance_path, trial_variance)
    print(f"Saved trial variance: {variance_path}", flush=True)

    if num_voxels == 0:
        raise ValueError("No voxels available after masking.")

    rng = np.random.default_rng(args.seed)

    if args.single_voxel_count > 0:
        voxel_count = min(args.single_voxel_count, num_voxels)
        voxel_indices = rng.choice(num_voxels, size=voxel_count, replace=False)
        single_voxel_dir = Path(args.single_voxel_dir)
        if not single_voxel_dir.is_absolute():
            single_voxel_dir = out_dir / single_voxel_dir
        single_voxel_dir.mkdir(parents=True, exist_ok=True)
        for voxel_idx in voxel_indices:
            out_path = single_voxel_dir / f"voxel_{voxel_idx}_trials.png"
            _plot_single_voxel_trials(trials[voxel_idx], int(voxel_idx), out_path, args.tr)
        print(f"Saved {voxel_count} single-voxel trial plots in {single_voxel_dir}.", flush=True)

    if args.variance_voxel_count > 0:
        voxel_count = min(args.variance_voxel_count, num_voxels)
        voxel_indices = rng.choice(num_voxels, size=voxel_count, replace=False)
        variance_fig = Path(args.variance_fig)
        if not variance_fig.is_absolute():
            variance_fig = out_dir / variance_fig
        _plot_variance_heatmap(trial_variance, voxel_indices, variance_fig)
        print(f"Saved variance heatmap: {variance_fig}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute X_task * beta_task from GLMsingle outputs, "
            "or visualize trial-wise BOLD when --bold-path is set."
        )
    )
    parser.add_argument("--runs", default="1,2", help="Comma-separated run numbers (1-based).")
    parser.add_argument(
        "--model-path",
        default="data/sub09_ses01/TYPED_FITHRF_GLMDENOISE_RR.npy",
    )
    parser.add_argument(
        "--design-path",
        default="data/sub09_ses01/DESIGNINFO.npy",
    )
    parser.add_argument(
        "--out-dir",
        default="data/sub09_ses01",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Voxel chunk size for matrix multiplies.",
    )
    viz_group = parser.add_argument_group("Trial visualization")
    viz_group.add_argument(
        "--bold-path",
        type=Path,
        default= '/scratch/st-mmckeown-1/zkavian/fmri_opt/Thesis_code_Glm_Opt/data/sub09_ses01/sub-pd009_ses-1_run-1_task-mv_bold_corrected_smoothed_reg.nii.gz',
        help="Path to BOLD data (.npy or NIfTI). When set, run trial visualization pipeline.",
    )
    viz_group.add_argument(
        "--mask-path",
        type=Path, 
        help="Optional mask (.npy or NIfTI) to select voxels.",
    )
    viz_group.add_argument("--num-trials", type=int, default=90, help="Number of trials (default: 90).")
    viz_group.add_argument("--trial-len", type=int, default=9, help="Timepoints per trial (default: 9).")
    viz_group.add_argument(
        "--rest-after",
        default="30,60",
        help="Comma-separated trial indices after which rest blocks occur (default: 30,60).",
    )
    viz_group.add_argument("--rest-len", type=int, default=20, help="Rest block length in timepoints.")
    viz_group.add_argument("--tr", type=float, default=1.0, help="TR in seconds (default: 1.0).")
    viz_group.add_argument(
        "--single-voxel-count",
        type=int,
        default=10,
        help="Number of random voxels for per-trial plots (default: 10).",
    )
    viz_group.add_argument(
        "--variance-voxel-count",
        type=int,
        default=20,
        help="Number of random voxels for variance heatmap (default: 20).",
    )
    viz_group.add_argument(
        "--variance-out",
        default="trial_variance.npy",
        help="Output path for variance array (default: trial_variance.npy).",
    )
    viz_group.add_argument(
        "--variance-fig",
        default="trial_variance_heatmap.png",
        help="Output path for variance heatmap (default: trial_variance_heatmap.png).",
    )
    viz_group.add_argument(
        "--single-voxel-dir",
        default="single_voxel_trials",
        help="Directory for single-voxel trial plots (default: single_voxel_trials).",
    )
    viz_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for voxel selection (default: random).",
    )
    args = parser.parse_args()

    if args.bold_path is not None:
        _run_trial_viz(args)
        return

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()
    design_path = Path(args.design_path)
    if not design_path.is_absolute():
        design_path = (REPO_ROOT / design_path).resolve()

    design_single_list, hrflibrary = _load_design(design_path)
    betasmd, hrfindexrun = _load_model(model_path)

    betas, numtrials = _flatten_betas(betasmd)
    numruns = len(design_single_list)
    hrfindex_flat = _flatten_hrfindex(hrfindexrun, numruns)

    if hrflibrary.ndim != 2:
        raise ValueError(f"Unexpected hrflibrary shape: {hrflibrary.shape}")

    if design_single_list[0].shape[1] != numtrials:
        raise ValueError(
            f"Design has {design_single_list[0].shape[1]} trials, "
            f"but betasmd has {numtrials} trials."
        )

    runs = _parse_runs(args.runs)
    for run in runs:
        run_idx = run - 1
        if run_idx < 0 or run_idx >= numruns:
            raise ValueError(f"Run {run} is out of range for {numruns} runs.")

        design_single = design_single_list[run_idx]
        conv_by_hrf = _convolve_by_hrf(design_single, hrflibrary)

        out_path = out_dir / f"Xtask_beta_task_run{run}.npy"
        _write_task_prediction(
            betas,
            hrfindex_flat[:, run_idx],
            conv_by_hrf,
            out_path,
            args.chunk_size,
        )
        print(f"Saved task prediction: {out_path}")


if __name__ == "__main__":
    main()
