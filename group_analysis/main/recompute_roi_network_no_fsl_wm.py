#!/usr/bin/env python3
"""Recompute the ROI-edge network after upstream FSL white-matter removal.

This script is intended as a QC/reanalysis companion for the 32-node
``roi_edge_network`` result.  It does two things:

1. Classifies the original selected voxels, especially the "Unassigned Active
   Voxels" ROI, against FSL MNI tissue priors and writes overlay masks/figures.
2. Removes FSL white-matter voxels from the selected beta rows before ROI
   aggregation, then reruns the ROI-edge KSG connectivity and pairwise
   Laplacian spectral-distance analysis from the filtered voxel construction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, plotting
from numpy.lib.format import open_memmap


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATA_DIR = REPO_ROOT / "results" / "connectivity" / "tmp" / "data"
SOURCE_BETA = SOURCE_DATA_DIR / "selected_beta_trials.npy"
SOURCE_VOXELS = SOURCE_DATA_DIR / "selected_voxel_indices.npz"
SOURCE_ANAT = SOURCE_DATA_DIR / "MNI152_T1_2mm_brain.nii.gz"
SOURCE_COLUMN_INDICES = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "data"
    / "selected_beta_trials_subject_session_column_indices.npz"
)
SOURCE_TRIAL_MANIFEST = SOURCE_DATA_DIR / "concat_manifest_group.tsv"
ROI_IMG = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_rois_fitted.nii.gz"
ROI_SUMMARY = REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_roi_summary.json"
FSL_PRIOR_DIR = Path("/usr/local/fsl/data/standard/tissuepriors")
FSL_WHITE_PRIOR = FSL_PRIOR_DIR / "avg152T1_white.img"
FSL_GRAY_PRIOR = FSL_PRIOR_DIR / "avg152T1_gray.img"
FSL_CSF_PRIOR = FSL_PRIOR_DIR / "avg152T1_csf.img"
DEFAULT_OUT_ROOT = REPO_ROOT / "results" / "connectivity" / "roi_edge_network_no_fsl_wm_upstream"

NETWORK_METRIC = "mutual_information_ksg"
COMPARISON_METRIC = "laplacian_spectral_distance_signed"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify original unassigned ROI voxels by FSL tissue priors, remove "
            "FSL white-matter voxels upstream, and rerun KSG ROI-edge connectivity."
        )
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--white-threshold", type=float, default=0.5)
    parser.add_argument("--row-chunk", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-data-build", action="store_true")
    parser.add_argument("--skip-connectivity", action="store_true")
    parser.add_argument("--skip-pairwise", action="store_true")
    return parser.parse_args()


def _load_selected_ijk(path: Path, volume_shape: tuple[int, int, int]) -> np.ndarray:
    pack = np.load(path, allow_pickle=True)
    if "selected_ijk" in pack.files:
        ijk = np.asarray(pack["selected_ijk"], dtype=np.int32)
    elif "selected_flat_indices" in pack.files:
        flat = np.asarray(pack["selected_flat_indices"], dtype=np.int64)
        ijk = np.column_stack(np.unravel_index(flat, volume_shape)).astype(np.int32, copy=False)
    else:
        raise KeyError(f"{path} must contain selected_ijk or selected_flat_indices.")
    if ijk.ndim != 2 or ijk.shape[1] != 3:
        raise ValueError(f"Expected selected_ijk shape (N, 3), got {ijk.shape}.")
    return ijk


def _load_roi_names(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names: dict[int, str] = {}
    for row in payload.get("roi_rows", []):
        roi_id = row.get("roi_id")
        roi_name = row.get("roi_name")
        if isinstance(roi_id, int) and isinstance(roi_name, str):
            names[int(roi_id)] = roi_name
    for idx, name in enumerate(payload.get("all_roi_names_in_order", []), start=1):
        if idx not in names and isinstance(name, str):
            names[idx] = name
    return names


def _resampled_prior(path: Path, ref_img: nib.Nifti1Image) -> np.ndarray:
    img = nib.load(str(path))
    if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine):
        img = image.resample_to_img(
            img,
            ref_img,
            interpolation="continuous",
            force_resample=True,
            copy_header=True,
        )
    return np.squeeze(np.asarray(img.get_fdata(), dtype=np.float32))


def _dominant_tissue(white: np.ndarray, gray: np.ndarray, csf: np.ndarray) -> np.ndarray:
    labels = np.asarray(["gray", "white", "csf"], dtype=object)
    stack = np.vstack([gray, white, csf])
    return labels[np.argmax(stack, axis=0)]


def _write_mask(path: Path, mask_at_selected: np.ndarray, selected_ijk: np.ndarray, ref_img: nib.Nifti1Image) -> None:
    data = np.zeros(ref_img.shape[:3], dtype=np.uint8)
    coords = selected_ijk[np.asarray(mask_at_selected, dtype=bool)]
    if coords.size:
        x, y, z = coords.T
        data[x, y, z] = 1
    nib.save(nib.Nifti1Image(data, ref_img.affine, ref_img.header), str(path))


def _plot_mask(mask_path: Path, anat_path: Path, out_png: Path, title: str) -> None:
    display = plotting.plot_roi(
        str(mask_path),
        bg_img=str(anat_path),
        display_mode="ortho",
        draw_cross=False,
        title=title,
        alpha=0.75,
        colorbar=False,
    )
    display.savefig(str(out_png), dpi=180)
    display.close()


def _write_bar_plot(summary_df: pd.DataFrame, out_png: Path) -> None:
    plot_df = summary_df.loc[summary_df["scope"].isin(["all_selected", "unassigned_roi"])].copy()
    labels = plot_df["scope"].map(
        {
            "all_selected": "All selected",
            "unassigned_roi": "Unassigned ROI",
        }
    )
    values = np.vstack(
        [
            plot_df["pct_fsl_white_ge_threshold"].to_numpy(dtype=float),
            plot_df["pct_nonwhite"].to_numpy(dtype=float),
        ]
    ).T
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(plot_df.shape[0])
    ax.bar(x, values[:, 0], color="#9a9a9a", label="FSL WM")
    ax.bar(x, values[:, 1], bottom=values[:, 0], color="#4c78a8", label="non-WM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Voxels (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _classify_and_write_qc(
    out_root: Path,
    white_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    qc_dir = out_root / "tissue_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    roi_img = nib.load(str(ROI_IMG))
    roi_data = np.asarray(roi_img.get_fdata(), dtype=np.int32)
    selected_ijk = _load_selected_ijk(SOURCE_VOXELS, roi_data.shape)
    x, y, z = selected_ijk.T
    roi_labels_at_selected = roi_data[x, y, z]
    roi_names = _load_roi_names(ROI_SUMMARY)
    roi_name_at_selected = np.asarray(
        [roi_names.get(int(roi_id), f"ROI_{int(roi_id)}") for roi_id in roi_labels_at_selected],
        dtype=object,
    )

    white = _resampled_prior(FSL_WHITE_PRIOR, roi_img)[x, y, z]
    gray = _resampled_prior(FSL_GRAY_PRIOR, roi_img)[x, y, z]
    csf = _resampled_prior(FSL_CSF_PRIOR, roi_img)[x, y, z]

    fsl_white = white >= float(white_threshold)
    dominant = _dominant_tissue(white=white, gray=gray, csf=csf)
    unassigned = np.char.find(np.char.lower(roi_name_at_selected.astype(str)), "unassigned active voxels") >= 0
    unassigned_nonwhite = unassigned & ~fsl_white

    def scope_row(name: str, mask: np.ndarray) -> dict[str, object]:
        n = int(np.count_nonzero(mask))
        if n == 0:
            return {
                "scope": name,
                "n_voxels": 0,
                "n_fsl_white_ge_threshold": 0,
                "pct_fsl_white_ge_threshold": 0.0,
                "n_nonwhite": 0,
                "pct_nonwhite": 0.0,
                "n_dominant_gray": 0,
                "n_dominant_white": 0,
                "n_dominant_csf": 0,
                "mean_white_prior": np.nan,
                "mean_gray_prior": np.nan,
                "mean_csf_prior": np.nan,
            }
        return {
            "scope": name,
            "n_voxels": n,
            "n_fsl_white_ge_threshold": int(np.count_nonzero(mask & fsl_white)),
            "pct_fsl_white_ge_threshold": float(np.mean(fsl_white[mask]) * 100.0),
            "n_nonwhite": int(np.count_nonzero(mask & ~fsl_white)),
            "pct_nonwhite": float(np.mean((~fsl_white)[mask]) * 100.0),
            "n_dominant_gray": int(np.count_nonzero(dominant[mask] == "gray")),
            "n_dominant_white": int(np.count_nonzero(dominant[mask] == "white")),
            "n_dominant_csf": int(np.count_nonzero(dominant[mask] == "csf")),
            "mean_white_prior": float(np.mean(white[mask])),
            "mean_gray_prior": float(np.mean(gray[mask])),
            "mean_csf_prior": float(np.mean(csf[mask])),
        }

    summary_df = pd.DataFrame(
        [
            scope_row("all_selected", np.ones(selected_ijk.shape[0], dtype=bool)),
            scope_row("unassigned_roi", unassigned),
            scope_row("unassigned_fsl_white", unassigned & fsl_white),
            scope_row("unassigned_nonwhite", unassigned_nonwhite),
        ]
    )
    summary_df.to_csv(qc_dir / "tissue_prior_summary.csv", index=False)
    _write_bar_plot(summary_df, qc_dir / "tissue_prior_wm_nonwm_bar.png")

    roi_rows: list[dict[str, object]] = []
    for roi_id in sorted(int(v) for v in np.unique(roi_labels_at_selected) if int(v) > 0):
        mask = roi_labels_at_selected == roi_id
        row = scope_row(roi_names.get(roi_id, f"ROI_{roi_id}"), mask)
        row["roi_id"] = roi_id
        roi_rows.append(row)
    pd.DataFrame(roi_rows).to_csv(qc_dir / "tissue_prior_by_roi.csv", index=False)

    unassigned_df = pd.DataFrame(
        {
            "source_selected_row": np.flatnonzero(unassigned).astype(int),
            "x": selected_ijk[unassigned, 0].astype(int),
            "y": selected_ijk[unassigned, 1].astype(int),
            "z": selected_ijk[unassigned, 2].astype(int),
            "white_prior": white[unassigned].astype(float),
            "gray_prior": gray[unassigned].astype(float),
            "csf_prior": csf[unassigned].astype(float),
            "dominant_tissue": dominant[unassigned].astype(str),
            "fsl_white_ge_threshold": fsl_white[unassigned].astype(bool),
            "separation_class": np.where(
                fsl_white[unassigned],
                "true_white_matter_by_fsl_prior",
                np.where(
                    dominant[unassigned] == "gray",
                    "unmapped_probable_gray_matter",
                    "unmapped_nonwhite_other",
                ),
            ),
        }
    )
    unassigned_df.to_csv(qc_dir / "unassigned_voxel_tissue_classification.csv", index=False)

    mask_specs = [
        ("unassigned_all_mask.nii.gz", unassigned, "Unassigned active voxels"),
        ("unassigned_fsl_white_mask.nii.gz", unassigned & fsl_white, "Unassigned FSL white matter"),
        ("unassigned_nonwhite_mask.nii.gz", unassigned_nonwhite, "Unassigned non-WM voxels"),
        ("selected_fsl_white_mask.nii.gz", fsl_white, "All selected FSL white matter"),
    ]
    for filename, mask, title in mask_specs:
        mask_path = qc_dir / filename
        _write_mask(mask_path, mask, selected_ijk, roi_img)
        _plot_mask(mask_path, SOURCE_ANAT, qc_dir / filename.replace(".nii.gz", ".png"), title)

    keep_mask = ~fsl_white
    manifest = {
        "source_beta": str(SOURCE_BETA),
        "source_voxels": str(SOURCE_VOXELS),
        "roi_img": str(ROI_IMG),
        "roi_summary": str(ROI_SUMMARY),
        "white_prior": str(FSL_WHITE_PRIOR),
        "gray_prior": str(FSL_GRAY_PRIOR),
        "csf_prior": str(FSL_CSF_PRIOR),
        "white_prior_threshold": float(white_threshold),
        "n_selected_before": int(selected_ijk.shape[0]),
        "n_selected_after_fsl_white_removal": int(np.count_nonzero(keep_mask)),
        "n_selected_removed_fsl_white": int(np.count_nonzero(fsl_white)),
        "n_unassigned_total": int(np.count_nonzero(unassigned)),
        "n_unassigned_fsl_white": int(np.count_nonzero(unassigned & fsl_white)),
        "n_unassigned_nonwhite": int(np.count_nonzero(unassigned_nonwhite)),
    }
    (qc_dir / "tissue_qc_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return keep_mask, selected_ijk, manifest


def _write_filtered_voxel_indices(
    out_path: Path,
    selected_ijk: np.ndarray,
    keep_mask: np.ndarray,
) -> np.ndarray:
    source_pack = np.load(SOURCE_VOXELS, allow_pickle=True)
    row_indices = np.flatnonzero(keep_mask).astype(np.int64)
    payload: dict[str, np.ndarray] = {
        "selected_ijk": selected_ijk[row_indices].astype(np.int32, copy=False),
        "source_selected_row_indices": row_indices,
    }
    if "selected_active_indices" in source_pack.files:
        payload["selected_active_indices"] = np.asarray(source_pack["selected_active_indices"], dtype=np.int64)[row_indices]
    if "selected_flat_indices" in source_pack.files:
        payload["selected_flat_indices"] = np.asarray(source_pack["selected_flat_indices"], dtype=np.int64)[row_indices]
    if "selected_weights" in source_pack.files:
        payload["selected_weights"] = np.asarray(source_pack["selected_weights"], dtype=np.float32)[row_indices]
    if "weight_threshold" in source_pack.files:
        payload["weight_threshold"] = np.asarray(source_pack["weight_threshold"])
    np.savez(out_path, **payload)
    return row_indices


def _write_filtered_beta_splits(
    out_data_dir: Path,
    row_indices: np.ndarray,
    row_chunk: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    beta = np.load(SOURCE_BETA, mmap_mode="r")
    columns = np.load(SOURCE_COLUMN_INDICES, allow_pickle=True)
    written: list[dict[str, object]] = []
    step = int(max(1, row_chunk))

    for label in columns.files:
        cols = np.asarray(columns[label], dtype=np.int64)
        out_path = out_data_dir / f"selected_beta_trials_{label}.npy"
        if out_path.exists() and not overwrite:
            arr = np.load(out_path, mmap_mode="r")
            written.append({"label": label, "path": str(out_path), "shape": [int(arr.shape[0]), int(arr.shape[1])]})
            continue

        out = open_memmap(
            out_path,
            mode="w+",
            dtype=beta.dtype,
            shape=(int(row_indices.size), int(cols.size)),
        )
        for start in range(0, row_indices.size, step):
            stop = min(start + step, row_indices.size)
            src_rows = row_indices[start:stop]
            out[start:stop, :] = np.asarray(beta[np.ix_(src_rows, cols)], dtype=beta.dtype)
        out.flush()
        del out
        written.append({"label": label, "path": str(out_path), "shape": [int(row_indices.size), int(cols.size)]})
        print(f"Saved filtered beta split {label}: rows={row_indices.size}, trials={cols.size}", flush=True)

    return written


def _build_filtered_data(
    out_root: Path,
    keep_mask: np.ndarray,
    selected_ijk: np.ndarray,
    row_chunk: int,
    overwrite: bool,
) -> Path:
    out_data_dir = out_root / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    voxel_out = out_data_dir / "selected_voxel_indices.npz"
    row_indices = _write_filtered_voxel_indices(voxel_out, selected_ijk, keep_mask)
    pd.DataFrame(
        {
            "source_selected_row": row_indices.astype(int),
            "x": selected_ijk[row_indices, 0].astype(int),
            "y": selected_ijk[row_indices, 1].astype(int),
            "z": selected_ijk[row_indices, 2].astype(int),
        }
    ).to_csv(out_data_dir / "selected_voxel_indices.csv", index=False)

    written = _write_filtered_beta_splits(
        out_data_dir=out_data_dir,
        row_indices=row_indices,
        row_chunk=row_chunk,
        overwrite=overwrite,
    )
    if SOURCE_TRIAL_MANIFEST.exists():
        shutil.copy2(SOURCE_TRIAL_MANIFEST, out_data_dir / SOURCE_TRIAL_MANIFEST.name)
    shutil.copy2(SOURCE_COLUMN_INDICES, out_data_dir / SOURCE_COLUMN_INDICES.name)
    manifest = {
        "source_beta": str(SOURCE_BETA),
        "source_voxels": str(SOURCE_VOXELS),
        "source_column_indices": str(SOURCE_COLUMN_INDICES),
        "n_source_rows": int(keep_mask.size),
        "n_rows_after_filter": int(row_indices.size),
        "n_rows_removed": int(keep_mask.size - row_indices.size),
        "beta_splits": written,
    }
    (out_data_dir / "filtered_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_data_dir


def _run(cmd: list[str]) -> None:
    print("\n[cmd] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _run_connectivity(out_root: Path, out_data_dir: Path) -> Path:
    network_dir = out_root / "roi_edge_network"
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "main" / "roi_edge_connectivity.py"),
            "--data-dir",
            str(out_data_dir),
            "--beta-pattern",
            "selected_beta_trials_sub-*_ses-*.npy",
            "--roi-img",
            str(ROI_IMG),
            "--roi-summary",
            str(ROI_SUMMARY),
            "--voxel-indices-path",
            str(out_data_dir / "selected_voxel_indices.npz"),
            "--out-dir",
            str(network_dir),
            "--advanced-metrics-out-subdir",
            "advanced_metrics",
            "--advanced-metrics",
            NETWORK_METRIC,
            "--no-html",
        ]
    )
    return network_dir


def _run_pairwise(network_dir: Path) -> None:
    _run(
        [
            sys.executable,
            str(REPO_ROOT / "group_analysis" / "main" / "analyze_pairwise_metric_separation.py"),
            "--advanced-root",
            str(network_dir / "advanced_metrics"),
            "--out-dir",
            str(network_dir),
            "--metrics",
            NETWORK_METRIC,
            "--comparison-metrics",
            COMPARISON_METRIC,
        ]
    )


def _summary_row(name: str, summary_csv: Path) -> dict[str, object] | None:
    if not summary_csv.exists():
        return None
    df = pd.read_csv(summary_csv)
    subset = df.loc[
        (df["connectivity_metric"] == NETWORK_METRIC)
        & (df["comparison_metric"] == COMPARISON_METRIC)
        & (df["cohort"] == "cross_subject_only")
    ].copy()
    if subset.empty:
        return None
    row = subset.iloc[0].to_dict()
    return {
        "analysis": name,
        "n_within_condition": int(row["n_within_condition"]),
        "n_between_condition": int(row["n_between_condition"]),
        "off_off_mean_raw": float(row["off_off_mean_raw"]),
        "on_on_mean_raw": float(row["on_on_mean_raw"]),
        "off_on_mean_raw": float(row["off_on_mean_raw"]),
        "on_on_minus_off_off": float(row["on_on_mean_raw"] - row["off_off_mean_raw"]),
        "within_minus_between_lme_coef": float(row["lme_coef_within_minus_between"]),
        "within_minus_between_lme_p": float(row["lme_p_two_sided"]),
        "auc_within_gt_between": float(row["auc_within_gt_between"]),
        "cohen_d_oriented": float(row["cohen_d_oriented"]),
        "source_summary_csv": str(summary_csv),
    }


def _off_off_on_on_stats(stats_csv: Path) -> dict[str, object]:
    empty = {
        "off_off_vs_on_on_estimate_on_minus_off": np.nan,
        "off_off_vs_on_on_p": np.nan,
        "off_off_vs_on_on_stars": "",
        "source_distribution_stats_csv": str(stats_csv),
    }
    if not stats_csv.exists():
        return empty
    df = pd.read_csv(stats_csv)
    subset = df.loc[
        (df["connectivity_metric"] == NETWORK_METRIC)
        & (df["comparison_metric"] == COMPARISON_METRIC)
        & (df["cohort"] == "cross_subject_only")
        & (df["group_a"] == "OFF-OFF")
        & (df["group_b"] == "ON-ON")
    ].copy()
    if subset.empty:
        return empty
    row = subset.iloc[0]
    return {
        "off_off_vs_on_on_estimate_on_minus_off": float(row["estimate_group_b_minus_group_a"]),
        "off_off_vs_on_on_p": float(row["p_value_two_sided"]),
        "off_off_vs_on_on_stars": str(row["significance_stars"]),
        "source_distribution_stats_csv": str(stats_csv),
    }


def _write_comparison(out_root: Path, network_dir: Path) -> None:
    specs = [
        (
            "original_32_node",
            REPO_ROOT / "results" / "connectivity" / "roi_edge_network" / "1tmp" / "cross_subject_only_summary.csv",
            REPO_ROOT
            / "results"
            / "connectivity"
            / "roi_edge_network"
            / "1tmp"
            / "laplacian_spectral_distance_signed_distribution_stats.csv",
        ),
        (
            "posthoc_drop_unassigned_30_node",
            REPO_ROOT
            / "results"
            / "connectivity"
            / "roi_edge_network"
            / NETWORK_METRIC
            / "without_unassigned"
            / "cross_subject_only_summary.csv",
            REPO_ROOT
            / "results"
            / "connectivity"
            / "roi_edge_network"
            / NETWORK_METRIC
            / "without_unassigned"
            / "laplacian_spectral_distance_signed_distribution_stats.csv",
        ),
        (
            "upstream_fsl_wm_removed",
            network_dir / "cross_subject_only_summary.csv",
            network_dir / "laplacian_spectral_distance_signed_distribution_stats.csv",
        ),
    ]
    rows = []
    for name, summary_csv, stats_csv in specs:
        row = _summary_row(name, summary_csv)
        if row is None:
            continue
        row.update(_off_off_on_on_stats(stats_csv))
        rows.append(row)
    df = pd.DataFrame([row for row in rows if row is not None])
    if df.empty:
        return
    df.to_csv(out_root / "method_comparison_cross_subject_summary.csv", index=False)


def main() -> None:
    args = _parse_args()
    out_root = args.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    keep_mask, selected_ijk, qc_manifest = _classify_and_write_qc(
        out_root=out_root,
        white_threshold=float(args.white_threshold),
    )

    out_data_dir = out_root / "data"
    if not args.skip_data_build:
        out_data_dir = _build_filtered_data(
            out_root=out_root,
            keep_mask=keep_mask,
            selected_ijk=selected_ijk,
            row_chunk=int(args.row_chunk),
            overwrite=bool(args.overwrite),
        )

    network_dir = out_root / "roi_edge_network"
    if not args.skip_connectivity:
        network_dir = _run_connectivity(out_root=out_root, out_data_dir=out_data_dir)
    if not args.skip_pairwise:
        _run_pairwise(network_dir=network_dir)
    _write_comparison(out_root=out_root, network_dir=network_dir)

    run_manifest = {
        "qc": qc_manifest,
        "out_root": str(out_root),
        "data_dir": str(out_data_dir),
        "network_dir": str(network_dir),
        "connectivity_metric": NETWORK_METRIC,
        "comparison_metric": COMPARISON_METRIC,
        "skip_data_build": bool(args.skip_data_build),
        "skip_connectivity": bool(args.skip_connectivity),
        "skip_pairwise": bool(args.skip_pairwise),
    }
    (out_root / "analysis_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"\nSaved upstream FSL-WM reanalysis to: {out_root}", flush=True)


if __name__ == "__main__":
    main()
