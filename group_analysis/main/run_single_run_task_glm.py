#!/usr/bin/env python3
"""Run a first-level task GLM for one MNI-space BOLD run."""

from __future__ import annotations

import argparse
import json
import re
import sys
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
DEFAULT_BG_IMG = Path("/Data/zahra/anatomy_masks/MNI152_T1_2mm_brain.nii.gz")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "single_run_task_glm"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_OUTPUT_ROOT / "sub-pd004_ses-1_run-1"
)
BOLD_DIR = Path("/Data/zahra/bold_data")
GO_TIMES_DIR = Path("/Data/zahra/go_times")
BOLD_GLOB = "*_task-mv_bold_corrected_smoothed_mnireg-2mm.nii.gz"
BOLD_FILENAME_RE = re.compile(
    r"^(?P<subject>sub-pd(?P<subject_id>\d+))_"
    r"(?P<session>ses-(?P<session_id>\d+))_"
    r"(?P<run>run-(?P<run_id>\d+))_"
    r"task-mv_bold_corrected_smoothed_mnireg-2mm\.nii\.gz$"
)
ORTHO_CUT_COORDS = (0, -20, 50)
MULTI_SLICE_CUTS = {
    "x": (-48, -36, -24, -12, 0, 12, 24, 36, 48),
    "y": (-72, -56, -40, -24, -8, 8, 24, 40, 56),
    "z": (-24, -12, 0, 12, 24, 36, 48, 60, 72),
}


def _parse_bold_filename(path: Path) -> dict[str, str | int]:
    match = BOLD_FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot parse subject/session/run from BOLD filename: {path}")
    label = "_".join(
        (match.group("subject"), match.group("session"), match.group("run"))
    )
    return {
        "subject": match.group("subject"),
        "subject_id": match.group("subject_id"),
        "session": match.group("session"),
        "session_id": match.group("session_id"),
        "run": match.group("run"),
        "run_id": int(match.group("run_id")),
        "label": label,
    }


def _go_times_for_bold(path: Path, go_times_dir: Path) -> Path:
    info = _parse_bold_filename(path)
    return (
        go_times_dir
        / f"PSPD{info['subject_id']}-ses-{info['session_id']}-go-times.txt"
    )


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
) -> dict[str, str]:
    figure_paths = {}

    z_png = output_dir / f"{contrast}_z_score.png"
    display = plotting.plot_stat_map(
        z_map,
        title="Task > baseline z-score",
        threshold=3.09,
        display_mode="ortho",
        cut_coords=ORTHO_CUT_COORDS,
        colorbar=True,
    )
    display.savefig(z_png, dpi=160)
    display.close()
    figure_paths["z_score_ortho_png"] = str(z_png)

    for axis, cut_coords in MULTI_SLICE_CUTS.items():
        z_slices_png = output_dir / f"{contrast}_z_score_slices_{axis}.png"
        display = plotting.plot_stat_map(
            z_map,
            title=f"Task > baseline z-score ({axis}-axis slices)",
            threshold=3.09,
            display_mode=axis,
            cut_coords=cut_coords,
            colorbar=True,
            draw_cross=False,
        )
        display.savefig(z_slices_png, dpi=160)
        display.close()
        figure_paths[f"z_score_slices_{axis}_png"] = str(z_slices_png)

    fdr_png = output_dir / f"{contrast}_z_score_fdr05_pos.png"
    display = plotting.plot_stat_map(
        str(fdr_map_path),
        title="Task > baseline, FDR q<0.05",
        display_mode="ortho",
        cut_coords=ORTHO_CUT_COORDS,
        colorbar=True,
        cmap="hot",
        symmetric_cbar=False,
    )
    display.savefig(fdr_png, dpi=160)
    display.close()
    figure_paths["z_score_fdr05_pos_ortho_png"] = str(fdr_png)

    for axis, cut_coords in MULTI_SLICE_CUTS.items():
        fdr_slices_png = (
            output_dir / f"{contrast}_z_score_fdr05_pos_slices_{axis}.png"
        )
        display = plotting.plot_stat_map(
            str(fdr_map_path),
            title=f"Task > baseline, FDR q<0.05 ({axis}-axis slices)",
            display_mode=axis,
            cut_coords=cut_coords,
            colorbar=True,
            cmap="hot",
            symmetric_cbar=False,
            draw_cross=False,
        )
        display.savefig(fdr_slices_png, dpi=160)
        display.close()
        figure_paths[f"z_score_fdr05_pos_slices_{axis}_png"] = str(fdr_slices_png)

    effect_png = output_dir / f"{contrast}_effect_size.png"
    display = plotting.plot_stat_map(
        effect_map,
        title="Task > baseline effect size",
        display_mode="ortho",
        cut_coords=ORTHO_CUT_COORDS,
        colorbar=True,
    )
    display.savefig(effect_png, dpi=160)
    display.close()
    figure_paths["effect_size_ortho_png"] = str(effect_png)

    for axis, cut_coords in MULTI_SLICE_CUTS.items():
        effect_slices_png = output_dir / f"{contrast}_effect_size_slices_{axis}.png"
        display = plotting.plot_stat_map(
            effect_map,
            title=f"Task > baseline effect size ({axis}-axis slices)",
            display_mode=axis,
            cut_coords=cut_coords,
            colorbar=True,
            draw_cross=False,
        )
        display.savefig(effect_slices_png, dpi=160)
        display.close()
        figure_paths[f"effect_size_slices_{axis}_png"] = str(effect_slices_png)

    pdf_path = output_dir / f"{contrast}_glm_summary.pdf"
    with PdfPages(pdf_path) as pdf:
        for png_path in (
            z_png,
            *(
                output_dir / f"{contrast}_z_score_slices_{axis}.png"
                for axis in ("z", "y", "x")
            ),
            fdr_png,
            *(
                output_dir / f"{contrast}_z_score_fdr05_pos_slices_{axis}.png"
                for axis in ("z", "y", "x")
            ),
            effect_png,
            *(
                output_dir / f"{contrast}_effect_size_slices_{axis}.png"
                for axis in ("z", "y", "x")
            ),
        ):
            img = plt.imread(png_path)
            fig, ax = plt.subplots(figsize=(11, 4.5))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout(pad=0)
            pdf.savefig(fig)
            plt.close(fig)
    figure_paths["glm_summary_pdf"] = str(pdf_path)
    return figure_paths


def _save_interactive_views(
    z_map: nib.Nifti1Image,
    effect_map: nib.Nifti1Image,
    fdr_map_path: Path,
    bg_img_path: Path,
    output_dir: Path,
    contrast: str,
) -> dict[str, str]:
    html_paths = {}

    z_html = output_dir / f"{contrast}_z_score_interactive.html"
    view = plotting.view_img(
        z_map,
        title="Task > baseline z-score",
        threshold=3.09,
        bg_img=str(bg_img_path),
        colorbar=True,
        symmetric_cmap=True,
        cmap="RdBu_r",
        resampling_interpolation="continuous",
    )
    view.save_as_html(z_html)
    html_paths["z_score_interactive_html"] = str(z_html)

    fdr_html = output_dir / f"{contrast}_z_score_fdr05_pos_interactive.html"
    view = plotting.view_img(
        str(fdr_map_path),
        title="Task > baseline, FDR q<0.05",
        threshold=1e-6,
        bg_img=str(bg_img_path),
        colorbar=True,
        symmetric_cmap=False,
        cmap="hot",
        resampling_interpolation="continuous",
    )
    view.save_as_html(fdr_html)
    html_paths["z_score_fdr05_pos_interactive_html"] = str(fdr_html)

    effect_html = output_dir / f"{contrast}_effect_size_interactive.html"
    view = plotting.view_img(
        effect_map,
        title="Task > baseline effect size",
        threshold=1e-6,
        bg_img=str(bg_img_path),
        colorbar=True,
        symmetric_cmap=True,
        cmap="RdBu_r",
        resampling_interpolation="continuous",
    )
    view.save_as_html(effect_html)
    html_paths["effect_size_interactive_html"] = str(effect_html)

    return html_paths


def _save_html_report(
    model: FirstLevelModel,
    contrast: str,
    output_dir: Path,
    title: str,
) -> str:
    report = model.generate_report(
        contrasts=contrast,
        title=title,
        threshold=3.09,
        height_control=None,
        alpha=0.001,
        two_sided=False,
    )
    report_path = output_dir / f"{contrast}_nilearn_report.html"
    report.save_as_html(report_path)
    return str(report_path)


def _regenerate_report_artifacts(
    output_dir: Path,
    contrast: str,
    bg_img_path: Path,
) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    bg_img_path = _require_file(bg_img_path, "background image")
    z_map_path = _require_file(output_dir / f"{contrast}_z_score.nii.gz", "z-score map")
    effect_map_path = _require_file(
        output_dir / f"{contrast}_effect_size.nii.gz",
        "effect-size map",
    )
    fdr_map_path = _require_file(
        output_dir / f"{contrast}_z_score_fdr05_pos.nii.gz",
        "FDR-positive z-score map",
    )

    artifacts = {
        "figures": _save_figures(
            nib.load(str(z_map_path)),
            nib.load(str(effect_map_path)),
            fdr_map_path,
            output_dir,
            contrast,
        ),
        "interactive_html": _save_interactive_views(
            nib.load(str(z_map_path)),
            nib.load(str(effect_map_path)),
            fdr_map_path,
            bg_img_path,
            output_dir,
            contrast,
        ),
    }

    summary_path = output_dir / "summary.json"
    summary: dict[str, object] = {}
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    summary["bg_img"] = str(bg_img_path)
    summary["report_artifacts"] = artifacts
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run_glm(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bold_path = _require_file(args.bold, "BOLD image")
    go_times_path = _require_file(args.go_times, "go-times file")
    brain_mask_path = _require_file(args.brain_mask, "brain mask")
    csf_mask_path = _require_file(args.csf_mask, "CSF mask")
    bg_img_path = _require_file(args.bg_img, "background image")

    bold_img = nib.load(str(bold_path))
    try:
        run_label = str(_parse_bold_filename(bold_path)["label"])
    except ValueError:
        run_label = bold_path.name.removesuffix(".nii.gz")

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
    report_artifacts = {
        "figures": _save_figures(
            nib.load(str(saved_maps["z_score"])),
            nib.load(str(saved_maps["effect_size"])),
            Path(threshold_summary["fdr05_map"]),
            output_dir,
            "task",
        ),
        "interactive_html": _save_interactive_views(
            nib.load(str(saved_maps["z_score"])),
            nib.load(str(saved_maps["effect_size"])),
            Path(threshold_summary["fdr05_map"]),
            bg_img_path,
            output_dir,
            "task",
        ),
    }
    nilearn_report = _save_html_report(
        model,
        "task",
        output_dir,
        title=f"{run_label} task GLM",
    )

    summary = {
        "run_label": run_label,
        "bold": str(bold_path),
        "go_times": str(go_times_path),
        "brain_mask": str(brain_mask_path),
        "csf_mask": str(csf_mask_path),
        "bg_img": str(bg_img_path),
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
        "report_artifacts": report_artifacts,
        "nilearn_report": nilearn_report,
        "warnings": warnings,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _discover_batch_runs(args: argparse.Namespace) -> list[dict[str, object]]:
    bold_dir = args.bold_dir.expanduser().resolve()
    go_times_dir = args.go_times_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not bold_dir.is_dir():
        raise FileNotFoundError(f"Missing BOLD directory: {bold_dir}")
    if not go_times_dir.is_dir():
        raise FileNotFoundError(f"Missing go-times directory: {go_times_dir}")

    runs = []
    for bold_path in sorted(bold_dir.glob(args.bold_glob)):
        info = _parse_bold_filename(bold_path)
        go_times_path = _go_times_for_bold(bold_path, go_times_dir)
        runs.append(
            {
                **info,
                "bold": bold_path,
                "go_times": go_times_path,
                "run_row": info["run_id"],
                "output_dir": output_root / str(info["label"]),
            }
        )
    if not runs:
        raise FileNotFoundError(
            f"No BOLD files matched {args.bold_glob!r} in {bold_dir}"
        )
    return runs


def run_batch(args: argparse.Namespace) -> dict[str, object]:
    batch_runs = _discover_batch_runs(args)

    if args.list_runs:
        return {
            "mode": "list_runs",
            "n_runs": len(batch_runs),
            "runs": [
                {
                    "label": run["label"],
                    "bold": str(run["bold"]),
                    "go_times": str(run["go_times"]),
                    "run_row": int(run["run_row"]),
                    "output_dir": str(run["output_dir"]),
                }
                for run in batch_runs
            ],
        }

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []
    skipped = []
    errors = []

    total_runs = len(batch_runs)
    for index, run in enumerate(batch_runs, start=1):
        label = str(run["label"])
        output_dir = Path(run["output_dir"])
        summary_path = output_dir / "summary.json"
        if args.skip_existing and summary_path.is_file():
            print(
                f"[{index}/{total_runs}] Skipping existing {label}",
                file=sys.stderr,
                flush=True,
            )
            skipped.append(
                {
                    "label": label,
                    "summary": str(summary_path),
                    "output_dir": str(output_dir),
                }
            )
            continue

        print(
            f"[{index}/{total_runs}] Running {label}",
            file=sys.stderr,
            flush=True,
        )
        run_args = argparse.Namespace(**vars(args))
        run_args.bold = Path(run["bold"])
        run_args.go_times = Path(run["go_times"])
        run_args.output_dir = output_dir
        run_args.run_row = int(run["run_row"])

        try:
            summary = run_glm(run_args)
        except Exception as exc:
            error = {
                "label": label,
                "bold": str(run["bold"]),
                "go_times": str(run["go_times"]),
                "output_dir": str(output_dir),
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(error)
            print(
                f"[{index}/{total_runs}] Failed {label}: {error['error']}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                raise
        else:
            print(
                f"[{index}/{total_runs}] Completed {label}",
                file=sys.stderr,
                flush=True,
            )
            completed.append(
                {
                    "label": label,
                    "summary": str(output_dir / "summary.json"),
                    "output_dir": str(output_dir),
                    "active_voxels_fdr05_positive": summary["thresholds"][
                        "active_voxels_fdr05_positive"
                    ],
                    "max_z": summary["thresholds"]["max_z"],
                }
            )

    batch_summary = {
        "mode": "batch",
        "bold_dir": str(args.bold_dir.expanduser().resolve()),
        "go_times_dir": str(args.go_times_dir.expanduser().resolve()),
        "output_root": str(output_root),
        "n_discovered": len(batch_runs),
        "n_completed": len(completed),
        "n_skipped": len(skipped),
        "n_errors": len(errors),
        "completed": completed,
        "skipped": skipped,
        "errors": errors,
    }
    with (output_root / "batch_summary.json").open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2)
    return batch_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a first-level task GLM for one MNI-space BOLD run."
    )
    parser.add_argument("--bold", type=Path, default=DEFAULT_BOLD)
    parser.add_argument("--go-times", type=Path, default=DEFAULT_GO_TIMES)
    parser.add_argument("--brain-mask", type=Path, default=DEFAULT_BRAIN_MASK)
    parser.add_argument("--csf-mask", type=Path, default=DEFAULT_CSF_MASK)
    parser.add_argument("--bg-img", type=Path, default=DEFAULT_BG_IMG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
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
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Run the GLM for every matching BOLD file in --bold-dir, saving each "
            "run under --output-root/<sub>_<ses>_<run>/."
        ),
    )
    parser.add_argument("--bold-dir", type=Path, default=BOLD_DIR)
    parser.add_argument("--go-times-dir", type=Path, default=GO_TIMES_DIR)
    parser.add_argument("--bold-glob", default=BOLD_GLOB)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="In --batch mode, skip runs that already have summary.json.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="In --batch mode, keep processing later runs if one run fails.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="In --batch mode, list discovered runs and exit without fitting GLMs.",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Regenerate static PDF/PNG figures and interactive HTML from existing "
            "contrast maps in --output-dir without refitting the GLM."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch:
        summary = run_batch(args)
    elif args.figures_only:
        summary = _regenerate_report_artifacts(args.output_dir, "task", args.bg_img)
    else:
        summary = run_glm(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
