#!/usr/bin/env python3
"""Compare voxel-wise GLMsingle and standard GLM maps with a scatter plot."""

from __future__ import annotations

import argparse
import base64
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_GLMSINGLE_INPUT = (
    REPO_ROOT
    / "results"
    / "glmsingle_mni_typea"
    / "sub-pd004"
    / "ses-1"
    / "GLMOutputs-mni-std"
    / "TYPEA_ONOFF_betasmd_jet_saturated_interactive.html"
)
DEFAULT_STANDARD_INPUT = (
    REPO_ROOT
    / "results"
    / "single_run_task_glm"
    / "sub-pd004_ses-1_run-1"
    / "task_z_score_interactive.html"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "glm_comparison"
    / "sub-pd004_ses-1_run-1"
)


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _interactive_html_stem(path: Path) -> str:
    name = path.name
    suffix = "_interactive.html"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _resolve_map_from_html(path: Path) -> Path:
    """Resolve known Nilearn HTML report names to their companion NIfTI maps."""
    path = _require_file(path, "input map or HTML")
    if path.suffix != ".html":
        return path

    stem = _interactive_html_stem(path)
    candidates = [
        path.with_name(f"{stem}.nii.gz"),
        path.with_name(f"{stem}_mni.nii.gz"),
    ]

    if stem.endswith("_jet_saturated"):
        base_stem = stem[: -len("_jet_saturated")]
        candidates.extend(
            [
                path.with_name(f"{base_stem}_positive_jet_saturated_mni.nii.gz"),
                path.with_name(f"{base_stem}_signed_jet_saturated_mni.nii.gz"),
                path.with_name(f"{base_stem}_mni.nii.gz"),
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve a companion NIfTI map for {path}. Checked:\n  {checked}"
    )


def _load_map(path: Path, label: str) -> tuple[nib.Nifti1Image, np.ndarray]:
    path = _require_file(path, label)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"{label} must be a 3D image, got shape {data.shape}")
    return img, data


def _validate_same_grid(x_img: nib.Nifti1Image, y_img: nib.Nifti1Image) -> None:
    if x_img.shape[:3] != y_img.shape[:3]:
        raise ValueError(
            f"Shape mismatch: GLMsingle shape {x_img.shape[:3]} vs "
            f"standard GLM shape {y_img.shape[:3]}"
        )
    if not np.allclose(x_img.affine, y_img.affine, atol=1e-4):
        raise ValueError("Affine mismatch between GLMsingle and standard GLM maps.")


def _voxel_mask(x_data: np.ndarray, y_data: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(x_data) & np.isfinite(y_data)
    x_nonzero = x_data != 0
    y_nonzero = y_data != 0

    if mode == "all-finite":
        return finite
    if mode == "either-nonzero":
        return finite & (x_nonzero | y_nonzero)
    if mode == "both-nonzero":
        return finite & x_nonzero & y_nonzero
    if mode == "x-nonzero":
        return finite & x_nonzero

    raise ValueError(f"Unknown voxel mask mode: {mode}")


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    if x.size < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }

    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if x.size < 2 or np.nanstd(x) == 0:
        return None
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def _summary_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_median": float(np.nanmedian(values)),
    }


def _zscore_values(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Cannot z-score GLMsingle values with zero or invalid std.")
    return (values - mean) / std, mean, std


def _write_scatter_csv(
    output_csv: Path,
    mask: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_column: str,
) -> None:
    voxel_ijk = np.column_stack(np.where(mask))
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "i",
                "j",
                "k",
                x_column,
                "standard_glm_value",
            ]
        )
        writer.writerows(
            (
                int(ijk[0]),
                int(ijk[1]),
                int(ijk[2]),
                float(x_value),
                float(y_value),
            )
            for ijk, x_value, y_value in zip(voxel_ijk, x_values, y_values)
        )


def _write_html_report(output_html: Path, output_png: Path, summary: dict[str, object]) -> None:
    encoded_png = base64.b64encode(output_png.read_bytes()).decode("ascii")
    rows = "\n".join(
        f"<tr><th>{key}</th><td>{value}</td></tr>"
        for key, value in summary.items()
        if isinstance(value, (int, float, str)) or value is None
    )
    output_html.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>GLMsingle vs standard GLM voxel scatter</title>
  <style>
    body {{
      color: #202124;
      font-family: Arial, sans-serif;
      margin: 24px;
      max-width: 1100px;
    }}
    img {{
      display: block;
      height: auto;
      max-width: 100%;
    }}
    table {{
      border-collapse: collapse;
      margin-top: 20px;
    }}
    th, td {{
      border-bottom: 1px solid #ddd;
      padding: 6px 10px;
      text-align: left;
    }}
    th {{
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <h1>GLMsingle vs standard GLM voxel scatter</h1>
  <img alt="Voxel-wise GLMsingle vs standard GLM scatter plot" src="data:image/png;base64,{encoded_png}">
  <table>
    <tbody>
{rows}
    </tbody>
  </table>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a voxel-wise scatter plot comparing a GLMsingle beta map "
            "against a standard GLM map."
        )
    )
    parser.add_argument(
        "--glmsingle-input",
        "--glmsingle-html",
        dest="glmsingle_input",
        type=Path,
        default=DEFAULT_GLMSINGLE_INPUT,
        help="GLMsingle HTML view or NIfTI map for the x-axis.",
    )
    parser.add_argument(
        "--standard-input",
        "--standard-html",
        dest="standard_input",
        type=Path,
        default=DEFAULT_STANDARD_INPUT,
        help="Standard GLM HTML view or NIfTI map for the y-axis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for scatter outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--voxel-mask",
        choices=("x-nonzero", "both-nonzero", "either-nonzero", "all-finite"),
        default="x-nonzero",
        help=(
            "Which voxels to plot. Default x-nonzero plots voxels with a "
            "nonzero GLMsingle value."
        ),
    )
    parser.add_argument(
        "--no-zscore-glmsingle",
        action="store_true",
        help="Plot raw GLMsingle values instead of z-scoring them across plotted voxels.",
    )
    parser.add_argument(
        "--prefix",
        default="glmsingle_typea_saturated_zscore_vs_standard_glm",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.18,
        help="Scatter marker opacity.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip saving the plotted voxel values to CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    glmsingle_input = _require_file(args.glmsingle_input, "GLMsingle input")
    standard_input = _require_file(args.standard_input, "standard GLM input")
    glmsingle_map = _resolve_map_from_html(glmsingle_input)
    standard_map = _resolve_map_from_html(standard_input)

    x_img, x_data = _load_map(glmsingle_map, "GLMsingle map")
    y_img, y_data = _load_map(standard_map, "standard GLM map")
    _validate_same_grid(x_img, y_img)

    mask = _voxel_mask(x_data, y_data, args.voxel_mask)
    if not np.any(mask):
        raise ValueError(f"No voxels selected with --voxel-mask {args.voxel_mask}")

    x_raw_values = x_data[mask].astype(np.float64, copy=False)
    y_values = y_data[mask].astype(np.float64, copy=False)
    zscore_glmsingle = not args.no_zscore_glmsingle

    if zscore_glmsingle:
        x_values, x_z_mean, x_z_std = _zscore_values(x_raw_values)
        x_label = "GLMsingle_TypeA"
        x_csv_column = "glmsingle_zscore"
    else:
        x_values = x_raw_values
        x_z_mean = None
        x_z_std = None
        x_label = "GLMsingle_TypeA"
        x_csv_column = "glmsingle_value"

    corr = _safe_corr(x_values, y_values)
    fit = _fit_line(x_values, y_values)
    summary: dict[str, object] = {
        "glmsingle_input": str(glmsingle_input),
        "standard_glm_input": str(standard_input),
        "glmsingle_map": str(glmsingle_map),
        "standard_glm_map": str(standard_map),
        "voxel_mask": args.voxel_mask,
        "glmsingle_zscored": zscore_glmsingle,
        "glmsingle_zscore_mean": x_z_mean,
        "glmsingle_zscore_std": x_z_std,
        "n_voxels": int(x_values.size),
        "image_shape": list(x_img.shape[:3]),
        **_summary_stats(x_raw_values, "glmsingle_raw"),
        **_summary_stats(x_values, "glmsingle"),
        **_summary_stats(y_values, "standard_glm"),
        **corr,
    }
    if fit is not None:
        slope, intercept = fit
        summary["least_squares_slope"] = slope
        summary["least_squares_intercept"] = intercept

    output_png = output_dir / f"{args.prefix}_{args.voxel_mask}.png"
    output_pdf = output_dir / f"{args.prefix}_{args.voxel_mask}.pdf"
    output_html = output_dir / f"{args.prefix}_{args.voxel_mask}.html"
    output_json = output_dir / f"{args.prefix}_{args.voxel_mask}_summary.json"
    output_csv = output_dir / f"{args.prefix}_{args.voxel_mask}_voxels.csv"

    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    ax.scatter(
        x_values,
        y_values,
        s=args.point_size,
        alpha=args.alpha,
        color="#2f6f91",
        edgecolors="none",
        rasterized=True,
    )
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)

    if fit is not None:
        slope, intercept = fit
        line_x = np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), 200)
        ax.plot(
            line_x,
            slope * line_x + intercept,
            color="#c2410c",
            linewidth=1.8,
            label="least-squares fit",
        )
        ax.legend(loc="lower right", frameon=False)

    pearson_label = (
        "Pearson r = n/a"
        if summary["pearson_r"] is None
        else f"Pearson r = {summary['pearson_r']:.3f}"
    )
    spearman_label = (
        "Spearman rho = n/a"
        if summary["spearman_rho"] is None
        else f"Spearman rho = {summary['spearman_rho']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        f"n = {x_values.size:,}\n{pearson_label}\n{spearman_label}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "#dddddd"},
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Standard GLM")
    fig.tight_layout()
    fig.savefig(output_png, dpi=220)
    fig.savefig(output_pdf)
    plt.close(fig)

    if not args.no_csv:
        _write_scatter_csv(output_csv, mask, x_values, y_values, x_csv_column)
        summary["voxel_csv"] = str(output_csv)

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html_report(output_html, output_png, summary)

    print(f"GLMsingle map: {glmsingle_map}")
    print(f"Standard GLM map: {standard_map}")
    print(f"Voxel mask: {args.voxel_mask}")
    print(f"GLMsingle z-scored: {zscore_glmsingle}")
    if zscore_glmsingle:
        print(f"GLMsingle z-score mean/std: {x_z_mean:.6g} / {x_z_std:.6g}")
    print(f"Plotted voxels: {x_values.size:,}")
    if summary["pearson_r"] is not None:
        print(f"Pearson r: {summary['pearson_r']:.6f}")
    if summary["spearman_rho"] is not None:
        print(f"Spearman rho: {summary['spearman_rho']:.6f}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved PDF: {output_pdf}")
    print(f"Saved HTML: {output_html}")
    print(f"Saved summary: {output_json}")
    if not args.no_csv:
        print(f"Saved voxel CSV: {output_csv}")


if __name__ == "__main__":
    main()
