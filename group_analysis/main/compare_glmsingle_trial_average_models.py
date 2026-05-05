#!/usr/bin/env python3
"""Compare trial-averaged GLMsingle beta weights across model outputs."""

from __future__ import annotations

import argparse
import base64
import csv
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = (
    REPO_ROOT
    / "results"
    / "glmsingle_mni_typea"
    / "sub-pd004"
    / "ses-1"
    / "GLMOutputs-mni-std"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "glm_comparison" / "sub-pd004_ses-1"
DEFAULT_MODELS = (
    ("TYPEB", DEFAULT_INPUT_DIR / "TYPEB_FITHRF.npy"),
    ("TYPEC", DEFAULT_INPUT_DIR / "TYPEC_FITHRF_GLMDENOISE.npy"),
    ("TYPED", DEFAULT_INPUT_DIR / "TYPED_FITHRF_GLMDENOISE_RR.npy"),
)
MODEL_LABELS = {
    "TYPEB": "TYPEB FITHRF",
    "TYPEC": "TYPEC FITHRF GLMDENOISE",
    "TYPED": "TYPED FITHRF GLMDENOISE RR",
}


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


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
    return averaged, trial_count


def _load_trial_average(path: Path, field: str) -> tuple[np.ndarray, dict[str, object]]:
    path = _require_file(path, "GLMsingle model")
    glm = np.load(path, allow_pickle=True).item()
    if field not in glm:
        keys = ", ".join(sorted(str(key) for key in glm.keys()))
        raise KeyError(f"{path} does not contain field {field!r}. Available keys: {keys}")

    values = np.asarray(glm[field])
    averaged, trial_count = _trial_mean(values)
    flat_average = averaged.reshape(-1)
    flat_trial_count = trial_count.reshape(-1)
    finite = np.isfinite(flat_average)

    summary = {
        "path": str(path),
        "field": field,
        "source_shape": list(values.shape),
        "averaged_shape": list(averaged.shape),
        "n_voxels_total": int(flat_average.size),
        "n_voxels_finite": int(finite.sum()),
        "n_trials_max": int(values.shape[-1]),
        "trial_count_min": int(np.nanmin(flat_trial_count)),
        "trial_count_max": int(np.nanmax(flat_trial_count)),
        "mean_min": float(np.nanmin(flat_average)),
        "mean_max": float(np.nanmax(flat_average)),
        "mean_mean": float(np.nanmean(flat_average)),
        "mean_std": float(np.nanstd(flat_average)),
        "mean_median": float(np.nanmedian(flat_average)),
    }
    return flat_average, summary


def _voxel_mask(averages: dict[str, np.ndarray], mode: str) -> np.ndarray:
    arrays = list(averages.values())
    finite = np.logical_and.reduce([np.isfinite(values) for values in arrays])
    any_nonzero = np.logical_or.reduce([values != 0 for values in arrays])
    all_nonzero = np.logical_and.reduce([values != 0 for values in arrays])

    if mode == "all-finite":
        return finite
    if mode == "any-nonzero":
        return finite & any_nonzero
    if mode == "all-nonzero":
        return finite & all_nonzero

    raise ValueError(f"Unknown voxel mask mode: {mode}")


def _zscore(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    selected = values[mask]
    mean = float(np.nanmean(selected))
    std = float(np.nanstd(selected))
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Cannot z-score values with zero or invalid standard deviation.")
    return (values - mean) / std, mean, std


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


def _axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    xy_min = float(np.nanmin(np.concatenate([x, y])))
    xy_max = float(np.nanmax(np.concatenate([x, y])))
    if xy_min == xy_max:
        pad = 1.0 if xy_min == 0 else abs(xy_min) * 0.05
    else:
        pad = (xy_max - xy_min) * 0.04
    return xy_min - pad, xy_max + pad


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
    mask: np.ndarray,
    averages: dict[str, np.ndarray],
    plotted_values: dict[str, np.ndarray],
    zscored: bool,
) -> None:
    labels = list(averages.keys())
    value_suffix = "zscore" if zscored else "mean_betasmd"
    header = ["voxel_index"]
    header.extend(f"{label}_mean_betasmd" for label in labels)
    if zscored:
        header.extend(f"{label}_{value_suffix}" for label in labels)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for voxel_index in np.flatnonzero(mask):
            row: list[int | float] = [int(voxel_index)]
            row.extend(float(averages[label][voxel_index]) for label in labels)
            if zscored:
                row.extend(float(plotted_values[label][voxel_index]) for label in labels)
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
  <title>Trial-averaged GLMsingle model scatter</title>
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
  <h1>Trial-averaged GLMsingle model scatter</h1>
  <img alt="Pairwise voxel scatter plot for trial-averaged GLMsingle beta weights" src="data:image/png;base64,{encoded_png}">
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
            "Create pairwise voxel scatter plots for GLMsingle model outputs after "
            "averaging beta weights over the last, trial dimension."
        )
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Model file to plot. Repeat for multiple models. If omitted, the "
            "TYPEB/TYPEC/TYPED sub-pd004 ses-1 files are used."
        ),
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
        "--prefix",
        default="glmsingle_typeb_typec_typed_trialmean_betasmd_pairwise",
        help="Filename prefix for outputs.",
    )
    parser.add_argument(
        "--voxel-mask",
        choices=("any-nonzero", "all-nonzero", "all-finite"),
        default="any-nonzero",
        help="Which voxels to plot after trial averaging. Default: any-nonzero.",
    )
    parser.add_argument(
        "--zscore-averages",
        action="store_true",
        help="Z-score each trial-averaged model map across plotted voxels before plotting.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.5,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.16,
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
    if len(model_specs) < 2:
        raise ValueError("At least two model files are required for a scatter plot.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    averages: dict[str, np.ndarray] = {}
    model_summaries: dict[str, dict[str, object]] = {}
    shape = None
    for label, path in model_specs:
        if label in averages:
            raise ValueError(f"Duplicate model label: {label}")
        averages[label], model_summaries[label] = _load_trial_average(path, args.field)
        if shape is None:
            shape = averages[label].shape
        elif averages[label].shape != shape:
            raise ValueError(
                f"Shape mismatch for {label}: {averages[label].shape} vs {shape}"
            )

    mask = _voxel_mask(averages, args.voxel_mask)
    if not np.any(mask):
        raise ValueError(f"No voxels selected with --voxel-mask {args.voxel_mask}")

    zscore_params: dict[str, dict[str, float]] = {}
    if args.zscore_averages:
        plotted_values = {}
        for label, values in averages.items():
            z_values, mean, std = _zscore(values, mask)
            plotted_values[label] = z_values
            zscore_params[label] = {"mean": mean, "std": std}
    else:
        plotted_values = averages

    pair_summaries: dict[str, dict[str, object]] = {}
    pairs = list(itertools.combinations(averages.keys(), 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.6 * len(pairs), 5.2))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (x_label_key, y_label_key) in zip(axes, pairs):
        x = plotted_values[x_label_key][mask]
        y = plotted_values[y_label_key][mask]
        raw_x = averages[x_label_key][mask]
        raw_y = averages[y_label_key][mask]
        corr = _safe_corr(x, y)
        fit = _fit_line(x, y)
        pair_key = f"{x_label_key}_vs_{y_label_key}"
        pair_summary: dict[str, object] = {
            "n_voxels": int(x.size),
            **_summary_stats(raw_x, f"{x_label_key}_raw"),
            **_summary_stats(raw_y, f"{y_label_key}_raw"),
            **corr,
        }
        if fit is not None:
            slope, intercept = fit
            pair_summary["least_squares_slope"] = slope
            pair_summary["least_squares_intercept"] = intercept
        pair_summaries[pair_key] = pair_summary

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

        axis_min, axis_max = _axis_limits(x, y)
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            color="#777777",
            linewidth=1.0,
            linestyle=":",
            label="identity",
        )
        if fit is not None:
            line_x = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
            ax.plot(
                line_x,
                fit[0] * line_x + fit[1],
                color="#c2410c",
                linewidth=1.8,
                label="least-squares fit",
            )
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.set_aspect("equal", adjustable="box")

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
            f"n = {x.size:,}\n{pearson_label}\n{spearman_label}",
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

        x_label = MODEL_LABELS.get(x_label_key, x_label_key)
        y_label = MODEL_LABELS.get(y_label_key, y_label_key)
        value_label = "z-scored trial-mean beta" if args.zscore_averages else "trial-mean beta"
        ax.set_title(f"{x_label_key} vs {y_label_key}")
        ax.set_xlabel(f"{x_label} {value_label}")
        ax.set_ylabel(f"{y_label} {value_label}")
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle(
        f"Voxel-wise GLMsingle {args.field} scatter after averaging over trials",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()

    suffix = f"{args.voxel_mask}"
    if args.zscore_averages:
        suffix += "_zscore"
    output_png = output_dir / f"{args.prefix}_{suffix}.png"
    output_pdf = output_dir / f"{args.prefix}_{suffix}.pdf"
    output_html = output_dir / f"{args.prefix}_{suffix}.html"
    output_json = output_dir / f"{args.prefix}_{suffix}_summary.json"
    output_csv = output_dir / f"{args.prefix}_{suffix}_voxels.csv"

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    summary: dict[str, object] = {
        "field": args.field,
        "trial_axis": -1,
        "voxel_mask": args.voxel_mask,
        "zscore_averages": bool(args.zscore_averages),
        "n_voxels_plotted": int(mask.sum()),
        "models": model_summaries,
        "pairwise": pair_summaries,
    }
    if zscore_params:
        summary["zscore_params"] = zscore_params

    if not args.no_csv:
        _write_voxel_csv(output_csv, mask, averages, plotted_values, args.zscore_averages)
        summary["voxel_csv"] = str(output_csv)

    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html_report(
        output_html,
        output_png,
        {
            "field": args.field,
            "trial_axis": -1,
            "voxel_mask": args.voxel_mask,
            "zscore_averages": bool(args.zscore_averages),
            "n_voxels_plotted": int(mask.sum()),
            "png": str(output_png),
            "pdf": str(output_pdf),
            "summary_json": str(output_json),
            "voxel_csv": str(output_csv) if not args.no_csv else None,
        },
    )

    print(f"Averaged field: {args.field} over last dimension")
    print(f"Voxel mask: {args.voxel_mask}")
    print(f"Z-scored averages: {args.zscore_averages}")
    print(f"Plotted voxels: {int(mask.sum()):,}")
    for pair_key, pair_summary in pair_summaries.items():
        pearson = pair_summary["pearson_r"]
        spearman = pair_summary["spearman_rho"]
        print(f"{pair_key}: Pearson r={pearson:.6f}, Spearman rho={spearman:.6f}")
    print(f"Saved PNG: {output_png}")
    print(f"Saved PDF: {output_pdf}")
    print(f"Saved HTML: {output_html}")
    print(f"Saved summary: {output_json}")
    if not args.no_csv:
        print(f"Saved voxel CSV: {output_csv}")


if __name__ == "__main__":
    main()
