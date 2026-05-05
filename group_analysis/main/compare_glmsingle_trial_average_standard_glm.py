#!/usr/bin/env python3
"""Compare trial-averaged GLMsingle beta weights with a standard GLM map."""

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
DEFAULT_GLMSINGLE_DIR = (
    REPO_ROOT
    / "results"
    / "glmsingle_mni_typea"
    / "sub-pd004"
    / "ses-1"
    / "GLMOutputs-mni-std"
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
DEFAULT_MODELS = (
    ("TYPEB", DEFAULT_GLMSINGLE_DIR / "TYPEB_FITHRF.npy"),
    ("TYPEC", DEFAULT_GLMSINGLE_DIR / "TYPEC_FITHRF_GLMDENOISE.npy"),
    ("TYPED", DEFAULT_GLMSINGLE_DIR / "TYPED_FITHRF_GLMDENOISE_RR.npy"),
)
MODEL_LABELS = {
    "TYPEB": "GLMsingle TYPEB FITHRF trial-mean beta",
    "TYPEC": "GLMsingle TYPEC FITHRF GLMDENOISE trial-mean beta",
    "TYPED": "GLMsingle TYPED FITHRF GLMDENOISE RR trial-mean beta",
}
PLOT_X_LABELS = {
    "TYPEB": "GLMsingle_TypeB",
    "TYPEC": "GLMsingle_TypeC",
    "TYPED": "GLMsingle_TypeD",
}
DEFAULT_SATURATION_PERCENTILE = 99.5


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _interactive_html_stem(path: Path) -> str:
    suffix = "_interactive.html"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def _resolve_map_from_html(path: Path) -> Path:
    path = _require_file(path, "standard GLM input")
    if path.suffix != ".html":
        return path

    stem = _interactive_html_stem(path)
    candidates = [
        path.with_name(f"{stem}.nii.gz"),
        path.with_name(f"{stem}_mni.nii.gz"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve a companion NIfTI map for {path}. Checked:\n  {checked}"
    )


def _parse_model_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path

    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Model label is empty in --model {spec!r}")
    return label, Path(path.strip())


def _trial_mean(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim < 2:
        raise ValueError(
            f"Expected a trial dimension on the last axis, got shape {values.shape}"
        )

    finite = np.isfinite(values)
    trial_count = finite.sum(axis=-1)
    summed = np.where(finite, values, 0).sum(axis=-1, dtype=np.float64)
    averaged = np.full(trial_count.shape, np.nan, dtype=np.float64)
    np.divide(summed, trial_count, out=averaged, where=trial_count > 0)
    return averaged.reshape(-1), trial_count.reshape(-1)


def _load_trial_average(path: Path, field: str) -> tuple[np.ndarray, dict[str, object]]:
    path = _require_file(path, "GLMsingle model")
    glm = np.load(path, allow_pickle=True).item()
    if field not in glm:
        keys = ", ".join(sorted(str(key) for key in glm.keys()))
        raise KeyError(f"{path} does not contain field {field!r}. Available keys: {keys}")

    values = np.asarray(glm[field])
    averaged, trial_count = _trial_mean(values)
    finite = np.isfinite(averaged)
    summary = {
        "path": str(path),
        "field": field,
        "source_shape": list(values.shape),
        "n_trials_max": int(values.shape[-1]),
        "n_voxels_total": int(averaged.size),
        "n_voxels_finite": int(finite.sum()),
        "trial_count_min": int(np.nanmin(trial_count)),
        "trial_count_max": int(np.nanmax(trial_count)),
        "raw_mean_min": float(np.nanmin(averaged)),
        "raw_mean_max": float(np.nanmax(averaged)),
        "raw_mean_mean": float(np.nanmean(averaged)),
        "raw_mean_std": float(np.nanstd(averaged)),
        "raw_mean_median": float(np.nanmedian(averaged)),
    }
    return averaged, summary


def _load_mask_indices(path: Path, n_voxels: int, image_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = _require_file(path, "mask indices")
    indices = np.load(path, allow_pickle=True)
    if indices.shape[0] != 3:
        raise ValueError(f"Expected mask indices with shape (3, n_voxels), got {indices.shape}")

    coords = tuple(np.asarray(indices[axis], dtype=np.int64) for axis in range(3))
    if coords[0].size != n_voxels:
        raise ValueError(
            f"Mask index count {coords[0].size} does not match GLMsingle voxel count {n_voxels}"
        )
    for axis, coord in enumerate(coords):
        if coord.min() < 0 or coord.max() >= image_shape[axis]:
            raise ValueError(
                f"Mask indices exceed standard GLM image shape on axis {axis}: "
                f"min={coord.min()}, max={coord.max()}, shape={image_shape}"
            )
    return coords


def _voxel_mask(x: np.ndarray, y: np.ndarray, mode: str) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    x_nonzero = x != 0
    y_nonzero = y != 0

    if mode == "all-finite":
        return finite
    if mode == "x-nonzero":
        return finite & x_nonzero
    if mode == "y-nonzero":
        return finite & y_nonzero
    if mode == "either-nonzero":
        return finite & (x_nonzero | y_nonzero)
    if mode == "both-nonzero":
        return finite & x_nonzero & y_nonzero

    raise ValueError(f"Unknown voxel mask mode: {mode}")


def _zscore_values(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Cannot z-score GLMsingle values with zero or invalid std.")
    return (values - mean) / std, mean, std


def _saturate_signed_tails(
    values: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    if percentile <= 50 or percentile >= 100:
        raise ValueError("--saturation-percentile must be greater than 50 and less than 100.")

    saturated = values.astype(np.float64, copy=True)
    finite = np.isfinite(saturated)
    positive = saturated[finite & (saturated > 0)]
    negative = saturated[finite & (saturated < 0)]

    positive_threshold = (
        float(np.nanpercentile(positive, percentile)) if positive.size else None
    )
    negative_threshold = (
        float(np.nanpercentile(negative, 100.0 - percentile)) if negative.size else None
    )

    high_clip_count = 0
    low_clip_count = 0
    if positive_threshold is not None:
        high_mask = finite & (saturated > positive_threshold)
        high_clip_count = int(high_mask.sum())
        saturated[high_mask] = positive_threshold
    if negative_threshold is not None:
        low_mask = finite & (saturated < negative_threshold)
        low_clip_count = int(low_mask.sum())
        saturated[low_mask] = negative_threshold

    summary = {
        "percentile": float(percentile),
        "positive_threshold": positive_threshold,
        "negative_threshold": negative_threshold,
        "high_clip_count": high_clip_count,
        "low_clip_count": low_clip_count,
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
    }
    return saturated, summary


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


def _write_voxel_csv(
    output_csv: Path,
    masks: dict[str, np.ndarray],
    raw_values: dict[str, np.ndarray],
    plotted_values: dict[str, np.ndarray],
    standard_values: np.ndarray,
    zscored: bool,
    saturated_values: dict[str, np.ndarray] | None = None,
) -> None:
    labels = list(raw_values.keys())
    combined_mask = np.logical_or.reduce(list(masks.values()))
    plotted_suffix = "zscore" if zscored else "plotted_mean_betasmd"
    header = ["voxel_index", "standard_glm_value"]
    header.extend(f"{label}_mean_betasmd" for label in labels)
    if saturated_values is not None:
        header.extend(f"{label}_saturated_mean_betasmd" for label in labels)
    header.extend(f"{label}_{plotted_suffix}" for label in labels)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for voxel_index in np.flatnonzero(combined_mask):
            row: list[int | float | str] = [
                int(voxel_index),
                float(standard_values[voxel_index]),
            ]
            row.extend(float(raw_values[label][voxel_index]) for label in labels)
            if saturated_values is not None:
                row.extend(
                    float(saturated_values[label][voxel_index])
                    if masks[label][voxel_index]
                    and np.isfinite(saturated_values[label][voxel_index])
                    else ""
                    for label in labels
                )
            row.extend(
                float(plotted_values[label][voxel_index])
                if masks[label][voxel_index] and np.isfinite(plotted_values[label][voxel_index])
                else ""
                for label in labels
            )
            writer.writerow(row)


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
  <title>Trial-averaged GLMsingle vs standard GLM scatter</title>
  <style>
    body {{
      color: #202124;
      font-family: Arial, sans-serif;
      margin: 24px;
      max-width: 1400px;
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
  <h1>Trial-averaged GLMsingle vs standard GLM scatter</h1>
  <img alt="Voxel-wise trial-averaged GLMsingle versus standard GLM scatter plot" src="data:image/png;base64,{encoded_png}">
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
            "Create voxel-wise scatter plots comparing trial-averaged GLMsingle "
            "beta weights against a standard GLM z-score map."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "GLMsingle model file to plot. Repeat for multiple models. If omitted, "
            "the TYPEB/TYPEC/TYPED sub-pd004 ses-1 files are used."
        ),
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
        "--mask-indices",
        type=Path,
        default=DEFAULT_GLMSINGLE_DIR / "mask_indices.npy",
        help="Mask indices that map GLMsingle voxel vectors into the standard GLM image.",
    )
    parser.add_argument(
        "--field",
        default="betasmd",
        help="GLMsingle dictionary field to average over trials. Default: betasmd.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for scatter outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--voxel-mask",
        choices=("x-nonzero", "y-nonzero", "both-nonzero", "either-nonzero", "all-finite"),
        default="x-nonzero",
        help=(
            "Which voxels to plot. Default x-nonzero plots voxels with nonzero "
            "trial-averaged GLMsingle values."
        ),
    )
    parser.add_argument(
        "--no-zscore-glmsingle",
        action="store_true",
        help="Plot raw trial-averaged GLMsingle values instead of spatial z-scores.",
    )
    parser.add_argument(
        "--saturation-percentile",
        type=float,
        default=DEFAULT_SATURATION_PERCENTILE,
        help=(
            "Before z-scoring, clip positive GLMsingle values at this percentile "
            "of the positive tail and negative values at 100 - percentile of the "
            f"negative tail. Default: {DEFAULT_SATURATION_PERCENTILE}."
        ),
    )
    parser.add_argument(
        "--no-saturate-glmsingle",
        action="store_true",
        help="Skip signed-tail saturation before z-scoring GLMsingle values.",
    )
    parser.add_argument(
        "--prefix",
        default="glmsingle_typeb_typec_typed_trialmean_saturated_zscore_vs_standard_glm",
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
    model_specs = (
        [_parse_model_spec(spec) for spec in args.model]
        if args.model
        else list(DEFAULT_MODELS)
    )
    if len(model_specs) < 1:
        raise ValueError("At least one GLMsingle model file is required.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    standard_input = _require_file(args.standard_input, "standard GLM input")
    standard_map = _resolve_map_from_html(standard_input)
    standard_img = nib.load(str(standard_map))
    standard_data = standard_img.get_fdata(dtype=np.float32)
    if standard_data.ndim != 3:
        raise ValueError(f"Standard GLM map must be 3D, got shape {standard_data.shape}")

    raw_values: dict[str, np.ndarray] = {}
    model_summaries: dict[str, dict[str, object]] = {}
    for label, path in model_specs:
        if label in raw_values:
            raise ValueError(f"Duplicate model label: {label}")
        raw_values[label], model_summaries[label] = _load_trial_average(path, args.field)

    n_voxels = next(iter(raw_values.values())).size
    for label, values in raw_values.items():
        if values.size != n_voxels:
            raise ValueError(f"Voxel count mismatch for {label}: {values.size} vs {n_voxels}")

    coords = _load_mask_indices(args.mask_indices, n_voxels, standard_data.shape)
    standard_values = standard_data[coords].astype(np.float64, copy=False)

    zscore_glmsingle = not args.no_zscore_glmsingle
    saturate_glmsingle = zscore_glmsingle and not args.no_saturate_glmsingle
    masks: dict[str, np.ndarray] = {}
    saturated_values: dict[str, np.ndarray] = {}
    plotted_values: dict[str, np.ndarray] = {}
    zscore_params: dict[str, dict[str, float | None]] = {}
    saturation_params: dict[str, dict[str, float | int | None]] = {}
    pair_summaries: dict[str, dict[str, object]] = {}

    for label, values in raw_values.items():
        mask = _voxel_mask(values, standard_values, args.voxel_mask)
        if not np.any(mask):
            raise ValueError(f"No voxels selected for {label} with --voxel-mask {args.voxel_mask}")
        masks[label] = mask
        values_for_score = values
        if saturate_glmsingle:
            saturated, saturation_summary = _saturate_signed_tails(
                values[mask],
                args.saturation_percentile,
            )
            full_saturated = values.astype(np.float64, copy=True)
            full_saturated[mask] = saturated
            saturated_values[label] = full_saturated
            values_for_score = full_saturated
            saturation_params[label] = saturation_summary
        else:
            saturated_values[label] = values
            saturation_params[label] = {
                "percentile": None,
                "positive_threshold": None,
                "negative_threshold": None,
                "high_clip_count": 0,
                "low_clip_count": 0,
                "n_positive": int((values[mask] > 0).sum()),
                "n_negative": int((values[mask] < 0).sum()),
            }

        if zscore_glmsingle:
            plotted, mean, std = _zscore_values(values_for_score[mask])
            full_plotted = np.full(values.shape, np.nan, dtype=np.float64)
            full_plotted[mask] = plotted
            plotted_values[label] = full_plotted
            zscore_params[label] = {"mean": mean, "std": std}
        else:
            plotted_values[label] = values
            zscore_params[label] = {"mean": None, "std": None}

        x = plotted_values[label][mask]
        y = standard_values[mask]
        corr = _safe_corr(x, y)
        fit = _fit_line(x, y)
        pair_summary: dict[str, object] = {
            "n_voxels": int(mask.sum()),
            "glmsingle_zscored": zscore_glmsingle,
            "glmsingle_saturated_before_zscore": saturate_glmsingle,
            "glmsingle_zscore_mean": zscore_params[label]["mean"],
            "glmsingle_zscore_std": zscore_params[label]["std"],
            **{
                f"glmsingle_saturation_{key}": value
                for key, value in saturation_params[label].items()
            },
            **_summary_stats(values[mask], "glmsingle_raw"),
            **_summary_stats(saturated_values[label][mask], "glmsingle_saturated"),
            **_summary_stats(x, "glmsingle_plotted"),
            **_summary_stats(y, "standard_glm"),
            **corr,
        }
        if fit is not None:
            pair_summary["least_squares_slope"] = fit[0]
            pair_summary["least_squares_intercept"] = fit[1]
        pair_summaries[f"{label}_vs_standard_glm"] = pair_summary

    fig, axes = plt.subplots(1, len(raw_values), figsize=(6.1 * len(raw_values), 5.8))
    if len(raw_values) == 1:
        axes = [axes]

    for ax, label in zip(axes, raw_values.keys()):
        mask = masks[label]
        x = plotted_values[label][mask]
        y = standard_values[mask]
        corr = pair_summaries[f"{label}_vs_standard_glm"]
        fit = _fit_line(x, y)

        ax.scatter(
            x,
            y,
            s=args.point_size,
            alpha=args.alpha,
            color="#2f6f91",
            edgecolors="none",
            rasterized=True,
        )
        ax.axhline(0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.axvline(0, color="#777777", linewidth=0.8, linestyle="--", alpha=0.7)
        if fit is not None:
            line_x = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            ax.plot(
                line_x,
                fit[0] * line_x + fit[1],
                color="#c2410c",
                linewidth=1.8,
                label="least-squares fit",
            )
            ax.legend(loc="lower right", frameon=False, fontsize=8)

        pearson_label = (
            "Pearson r = n/a"
            if corr["pearson_r"] is None
            else f"Pearson r = {corr['pearson_r']:.3f}"
        )
        spearman_label = (
            "Spearman rho = n/a"
            if corr["spearman_rho"] is None
            else f"Spearman rho = {corr['spearman_rho']:.3f}"
        )
        ax.text(
            0.03,
            0.97,
            f"n = {int(mask.sum()):,}\n{pearson_label}\n{spearman_label}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "alpha": 0.85,
                "edgecolor": "#dddddd",
            },
        )

        x_label = PLOT_X_LABELS.get(label, label)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Standard GLM")
    fig.tight_layout()

    suffix = args.voxel_mask
    output_png = output_dir / f"{args.prefix}_{suffix}.png"
    output_pdf = output_dir / f"{args.prefix}_{suffix}.pdf"
    output_html = output_dir / f"{args.prefix}_{suffix}.html"
    output_json = output_dir / f"{args.prefix}_{suffix}_summary.json"
    output_csv = output_dir / f"{args.prefix}_{suffix}_voxels.csv"

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, object] = {
        "standard_glm_input": str(standard_input),
        "standard_glm_map": str(standard_map),
        "mask_indices": str(args.mask_indices.expanduser().resolve()),
        "field": args.field,
        "trial_axis": -1,
        "voxel_mask": args.voxel_mask,
        "glmsingle_zscored": zscore_glmsingle,
        "glmsingle_saturated_before_zscore": saturate_glmsingle,
        "saturation_percentile": args.saturation_percentile if saturate_glmsingle else None,
        "models": model_summaries,
        "saturation": saturation_params,
        "pairwise": pair_summaries,
    }

    if not args.no_csv:
        _write_voxel_csv(
            output_csv,
            masks,
            raw_values,
            plotted_values,
            standard_values,
            zscore_glmsingle,
            saturated_values if saturate_glmsingle else None,
        )
        summary["voxel_csv"] = str(output_csv)

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html_report(
        output_html,
        output_png,
        {
            "field": args.field,
            "trial_axis": -1,
            "voxel_mask": args.voxel_mask,
            "glmsingle_zscored": zscore_glmsingle,
            "glmsingle_saturated_before_zscore": saturate_glmsingle,
            "saturation_percentile": args.saturation_percentile if saturate_glmsingle else None,
            "standard_glm_map": str(standard_map),
            "png": str(output_png),
            "pdf": str(output_pdf),
            "summary_json": str(output_json),
            "voxel_csv": str(output_csv) if not args.no_csv else None,
        },
    )

    print(f"Standard GLM map: {standard_map}")
    print(f"Averaged field: {args.field} over last dimension")
    print(f"Voxel mask: {args.voxel_mask}")
    print(f"GLMsingle z-scored: {zscore_glmsingle}")
    print(f"GLMsingle saturated before z-score: {saturate_glmsingle}")
    for key, pair_summary in pair_summaries.items():
        pearson = pair_summary["pearson_r"]
        spearman = pair_summary["spearman_rho"]
        print(f"{key}: n={pair_summary['n_voxels']:,}, Pearson r={pearson:.6f}, Spearman rho={spearman:.6f}")
        if pair_summary["glmsingle_saturated_before_zscore"]:
            print(
                "  saturation thresholds: "
                f"negative={pair_summary['glmsingle_saturation_negative_threshold']:.6g}, "
                f"positive={pair_summary['glmsingle_saturation_positive_threshold']:.6g}; "
                f"clipped low={pair_summary['glmsingle_saturation_low_clip_count']}, "
                f"high={pair_summary['glmsingle_saturation_high_clip_count']}"
            )
    print(f"Saved PNG: {output_png}")
    print(f"Saved PDF: {output_pdf}")
    print(f"Saved HTML: {output_html}")
    print(f"Saved summary: {output_json}")
    if not args.no_csv:
        print(f"Saved voxel CSV: {output_csv}")


if __name__ == "__main__":
    main()
