#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(
        description="Compute X_task * beta_task from GLMsingle outputs and save it."
    )
    parser.add_argument("--runs", default="1,2", help="Comma-separated run numbers (1-based).")
    parser.add_argument(
        "--model-path",
        default="GLMsingle/GLMOutputs-sub09-ses1-std/TYPED_FITHRF_GLMDENOISE_RR.npy",
    )
    parser.add_argument(
        "--design-path",
        default="GLMsingle/GLMOutputs-sub09-ses1-std/DESIGNINFO.npy",
    )
    parser.add_argument(
        "--out-dir",
        default="GLMsingle/GLMOutputs-sub09-ses1-std",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Voxel chunk size for matrix multiplies.",
    )
    args = parser.parse_args()

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
