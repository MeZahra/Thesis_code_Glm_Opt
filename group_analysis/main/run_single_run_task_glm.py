#!/usr/bin/env python3
"""Run a first-level task GLM for one MNI-space BOLD run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from nilearn import plotting
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.maskers import NiftiMasker
from nilearn.reporting import get_clusters_table


REPO_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_BOLD = Path(
    "/Data/zahra/bold_data/"
    "sub-pd004_ses-1_run-1_task-mv_bold_corrected_smoothed_mnireg-2mm.nii.gz"
)
DEFAULT_GO_TIMES = Path("/Data/zahra/go_times/PSPD004-ses-1-go-times.txt")
DEFAULT_BRAIN_MASK = Path("/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain_mask.nii.gz")
DEFAULT_CSF_MASK = Path("/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain_seg_csf.nii.gz")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "single_run_task_glm" / "sub-pd004_ses-1_run-1"


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _validate_compatibility(
    bold_img: nib.Nifti1Image,
    mask_img: nib.Nifti1Image,
    csf_img: nib.Nifti1Image,
) -> None:
    bold_shape = tuple(bold_img.shape[:3])
    for label, img in (("brain mask", mask_img), ("CSF mask", csf_img)):
        if tuple(img.shape[:3]) != bold_shape:
            raise ValueError(
                f"Shape mismatch: BOLD spatial shape {bold_shape} vs "
                f"{label} shape {img.shape[:3]}"
            )
        if not np.allclose(bold_img.affine, img.affine, atol=1e-4):
            raise ValueError(f"Affine mismatch between BOLD image and {label}.")


def _select_run_onsets(go_times_path: Path, run_row: int) -> np.ndarray:
    go_times = np.loadtxt(go_times_path, dtype=float)
    if go_times.ndim == 1:
        go_times = go_times[None, :]
    if run_row < 1 or run_row > go_times.shape[0]:
        raise ValueError(
            f"Requested run row {run_row}, but {go_times_path} has "
            f"{go_times.shape[0]} row(s)."
        )
    return go_times[run_row - 1]


def _make_events(
    raw_onsets: np.ndarray,
    *,
    tr: float,
    duration: float,
    onset_mode: str,
    n_scans: int,
) -> tuple[pd.DataFrame, list[str]]:
    if onset_mode == "one_based_tr":
        onsets = (raw_onsets - 1.0) * tr
    elif onset_mode == "seconds":
        onsets = raw_onsets
    else:
        raise ValueError(f"Unknown onset mode: {onset_mode}")

    events = pd.DataFrame(
        {
            "onset": onsets.astype(float),
            "duration": np.full(raw_onsets.shape, float(duration)),
            "trial_type": "task",
        }
    )
    events = events.sort_values("onset").reset_index(drop=True)

    warnings = []
    if (events["onset"] < 0).any():
        warnings.append("At least one event starts before scan time 0.")

    run_duration = n_scans * tr
    beyond = events["onset"] + events["duration"] > run_duration
    if beyond.any():
        warnings.append(
            f"{int(beyond.sum())} event(s) extend past the acquired run "
            f"duration ({run_duration:.3f}s); Nilearn will effectively truncate "
            "their modeled response at the available scans."
        )

    return events, warnings


def _mean_csf_confounds(
    bold_img: nib.Nifti1Image,
    csf_img: nib.Nifti1Image,
    *,
    output_dir: Path,
) -> pd.DataFrame:
    masker = NiftiMasker(mask_img=csf_img, standardize=False)
    csf_data = masker.fit_transform(bold_img)
    if csf_data.size == 0 or csf_data.shape[1] == 0:
        raise ValueError("CSF mask selected no voxels.")

    csf_mean = np.nanmean(csf_data, axis=1)
    csf_mean = csf_mean.astype(np.float64, copy=False)
    csf_std = float(np.nanstd(csf_mean))
    if csf_std > 0:
        csf_mean = (csf_mean - float(np.nanmean(csf_mean))) / csf_std
    else:
        csf_mean = csf_mean - float(np.nanmean(csf_mean))

    confounds = pd.DataFrame({"csf_mean": csf_mean})
    confounds.to_csv(output_dir / "confounds.tsv", sep="\t", index=False)
    return confounds


def _save_design_matrix(model: FirstLevelModel, output_dir: Path) -> pd.DataFrame:
    design = model.design_matrices_[0]
    design.to_csv(output_dir / "design_matrix.tsv", sep="\t", index=True)

    ax = plotting.plot_design_matrix(design)
    ax.figure.set_size_inches(12, 6)
    ax.figure.tight_layout()
    ax.figure.savefig(output_dir / "design_matrix.png", dpi=160)
    plt.close(ax.figure)
    return design


def _save_contrast_maps(
    model: FirstLevelModel,
    contrast: str,
    output_dir: Path,
) -> dict[str, Path]:
    maps = model.compute_contrast(contrast, output_type="all")
    saved = {}
    for name, img in maps.items():
        path = output_dir / f"{contrast}_{name}.nii.gz"
        img.to_filename(path)
        saved[name] = path
    return saved


def _save_threshold_outputs(
    z_map: nib.Nifti1Image,
    output_dir: Path,
    contrast: str,
) -> dict[str, float | int | str | None]:
    fdr_map, fdr_threshold = threshold_stats_img(
        z_map,
        alpha=0.05,
        height_control="fdr",
        cluster_threshold=0,
        two_sided=False,
    )
    fdr_path = output_dir / f"{contrast}_z_score_fdr05_pos.nii.gz"
    fdr_map.to_filename(fdr_path)

    fpr_map, fpr_threshold = threshold_stats_img(
        z_map,
        alpha=0.001,
        height_control="fpr",
        cluster_threshold=0,
        two_sided=False,
    )
    fpr_path = output_dir / f"{contrast}_z_score_uncorrected_p001_pos.nii.gz"
    fpr_map.to_filename(fpr_path)

    z_data = z_map.get_fdata(dtype=np.float32)
    active_mask = np.isfinite(z_data) & (z_data >= float(fdr_threshold))
    active_img = nib.Nifti1Image(
        active_mask.astype(np.uint8),
        affine=z_map.affine,
        header=z_map.header,
    )
    active_img.header.set_data_dtype(np.uint8)
    active_path = output_dir / f"{contrast}_active_voxels_fdr05_pos_mask.nii.gz"
    active_img.to_filename(active_path)

    cluster_table_path = output_dir / f"{contrast}_clusters_fdr05_pos.csv"
    if np.any(active_mask):
        cluster_table = get_clusters_table(
            z_map,
            stat_threshold=float(fdr_threshold),
            cluster_threshold=0,
            two_sided=False,
        )
    else:
        cluster_table = pd.DataFrame()
    cluster_table.to_csv(cluster_table_path, index=False)

    return {
        "fdr05_positive_z_threshold": float(fdr_threshold),
        "fpr001_positive_z_threshold": float(fpr_threshold),
        "active_voxels_fdr05_positive": int(np.count_nonzero(active_mask)),
        "max_z": float(np.nanmax(z_data)),
        "fdr05_map": str(fdr_path),
        "fpr001_map": str(fpr_path),
        "active_mask": str(active_path),
        "cluster_table": str(cluster_table_path),
    }


def _save_figures(
    z_map: nib.Nifti1Image,
    effect_map: nib.Nifti1Image,
    fdr_map_path: Path,
    output_dir: Path,
    contrast: str,
) -> None:
    z_png = output_dir / f"{contrast}_z_score.png"
    display = plotting.plot_stat_map(
        z_map,
        title="Task > baseline z-score",
        threshold=3.09,
        display_mode="ortho",
        cut_coords=(0, -20, 50),
        colorbar=True,
    )
    display.savefig(z_png, dpi=160)
    display.close()

    fdr_png = output_dir / f"{contrast}_z_score_fdr05_pos.png"
    display = plotting.plot_stat_map(
        str(fdr_map_path),
        title="Task > baseline, FDR q<0.05",
        display_mode="ortho",
        cut_coords=(0, -20, 50),
        colorbar=True,
    )
    display.savefig(fdr_png, dpi=160)
    display.close()

    effect_png = output_dir / f"{contrast}_effect_size.png"
    display = plotting.plot_stat_map(
        effect_map,
        title="Task > baseline effect size",
        display_mode="ortho",
        cut_coords=(0, -20, 50),
        colorbar=True,
    )
    display.savefig(effect_png, dpi=160)
    display.close()

    pdf_path = output_dir / f"{contrast}_glm_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        for png_path in (z_png, fdr_png, effect_png):
            img = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig)
            plt.close(fig)


def _save_html_report(
    model: FirstLevelModel,
    contrast: str,
    output_dir: Path,
) -> None:
    report = model.generate_report(
        contrasts=contrast,
        title="sub-pd004 ses-1 run-1 task GLM",
        threshold=3.09,
        height_control=None,
        alpha=0.001,
        two_sided=False,
    )
    report.save_as_html(output_dir / f"{contrast}_nilearn_report.html")


def run_glm(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bold_path = _require_file(args.bold, "BOLD image")
    go_times_path = _require_file(args.go_times, "go-times file")
    brain_mask_path = _require_file(args.brain_mask, "brain mask")
    csf_mask_path = _require_file(args.csf_mask, "CSF mask")

    bold_img = nib.load(str(bold_path))
    brain_mask_img = nib.load(str(brain_mask_path))
    csf_mask_img = nib.load(str(csf_mask_path))
    _validate_compatibility(bold_img, brain_mask_img, csf_mask_img)

    n_scans = int(bold_img.shape[-1])
    raw_onsets = _select_run_onsets(go_times_path, args.run_row)
    events, warnings = _make_events(
        raw_onsets,
        tr=args.tr,
        duration=args.duration,
        onset_mode=args.onset_mode,
        n_scans=n_scans,
    )
    events.to_csv(output_dir / "events.tsv", sep="\t", index=False)

    confounds = _mean_csf_confounds(bold_img, csf_mask_img, output_dir=output_dir)

    model = FirstLevelModel(
        t_r=args.tr,
        hrf_model=args.hrf_model,
        drift_model="cosine",
        high_pass=args.high_pass,
        noise_model=args.noise_model,
        mask_img=brain_mask_img,
        signal_scaling=False,
        standardize=False,
        minimize_memory=False,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    model = model.fit(bold_img, events=events, confounds=confounds)

    design = _save_design_matrix(model, output_dir)
    saved_maps = _save_contrast_maps(model, "task", output_dir)
    threshold_summary = _save_threshold_outputs(
        nib.load(str(saved_maps["z_score"])),
        output_dir,
        "task",
    )
    _save_figures(
        nib.load(str(saved_maps["z_score"])),
        nib.load(str(saved_maps["effect_size"])),
        Path(threshold_summary["fdr05_map"]),
        output_dir,
        "task",
    )
    _save_html_report(model, "task", output_dir)

    summary = {
        "bold": str(bold_path),
        "go_times": str(go_times_path),
        "brain_mask": str(brain_mask_path),
        "csf_mask": str(csf_mask_path),
        "output_dir": str(output_dir),
        "n_scans": n_scans,
        "tr": float(args.tr),
        "duration": float(args.duration),
        "run_row": int(args.run_row),
        "onset_mode": args.onset_mode,
        "n_events": int(len(events)),
        "event_onset_min": float(events["onset"].min()),
        "event_onset_max": float(events["onset"].max()),
        "hrf_model": args.hrf_model,
        "high_pass": float(args.high_pass),
        "noise_model": args.noise_model,
        "design_columns": list(design.columns),
        "contrast_maps": {k: str(v) for k, v in saved_maps.items()},
        "thresholds": threshold_summary,
        "warnings": warnings,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a first-level task GLM for one MNI-space BOLD run."
    )
    parser.add_argument("--bold", type=Path, default=DEFAULT_BOLD)
    parser.add_argument("--go-times", type=Path, default=DEFAULT_GO_TIMES)
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--csf-mask", type=Path, default=DEFAULT_CSF_MASK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--run-row",
        type=int,
        default=1,
        help="1-based row number in the go-times file. Row 1 corresponds to run 1.",
    )
    parser.add_argument("--tr", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=9.0)
    parser.add_argument(
        "--onset-mode",
        choices=("one_based_tr", "seconds"),
        default="one_based_tr",
        help=(
            "Interpret go-time values as 1-based TR indices or as seconds from "
            "scan start."
        ),
    )
    parser.add_argument("--hrf-model", default="spm")
    parser.add_argument("--high-pass", type=float, default=0.01)
    parser.add_argument("--noise-model", choices=("ar1", "ols"), default="ar1")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    summary = run_glm(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
