#!/usr/bin/env python3
"""Plot overlapping probability histograms for saved GLM comparison voxel values."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPARISON_DIR = (
    REPO_ROOT
    / "results"
    / "glm_comparison"
    / "sub-pd004_ses-1_run-1"
)
TYPEA_STEM = "glmsingle_typea_saturated_zscore_vs_standard_glm_x-nonzero"
TYPEBCD_STEM = (
    "glmsingle_typeb_typec_typed_trialmean_saturated_zscore_vs_standard_glm_x-nonzero"
)
TYPEA_CSV = DEFAULT_COMPARISON_DIR / f"{TYPEA_STEM}_voxels.csv"
TYPEBCD_CSV = DEFAULT_COMPARISON_DIR / f"{TYPEBCD_STEM}_voxels.csv"


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _load_columns(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    path = _require_file(path, "voxel CSV")
    data = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=np.float64,
        encoding="utf-8",
    )
    missing = [column for column in columns if column not in data.dtype.names]
    if missing:
        available = ", ".join(data.dtype.names or ())
        raise KeyError(f"{path} is missing columns {missing}. Available columns: {available}")
    return {column: np.asarray(data[column], dtype=np.float64) for column in columns}


def _finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _summary(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    return {
        f"{prefix}_n": int(values.size),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_std": float(np.nanstd(values)),
        f"{prefix}_median": float(np.nanmedian(values)),
    }


def _bin_edges(
    values_a: np.ndarray,
    values_b: np.ndarray,
    bins: int,
    x_min: float | None,
    x_max: float | None,
) -> np.ndarray:
    if x_min is None:
        x_min = float(np.nanmin(np.concatenate([values_a, values_b])))
    if x_max is None:
        x_max = float(np.nanmax(np.concatenate([values_a, values_b])))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min >= x_max:
        raise ValueError(f"Invalid histogram limits: {x_min}, {x_max}")
    return np.linspace(x_min, x_max, bins + 1)


def _plot_probability_histograms(
    output_prefix: Path,
    panels: list[dict[str, object]],
    bins: int,
    x_min: float | None,
    x_max: float | None,
    title: str,
) -> dict[str, object]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.1), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    summary: dict[str, object] = {
        "bins": bins,
        "x_min": x_min,
        "x_max": x_max,
        "panels": {},
    }
    colors = {
        "glmsingle": "#2f6f91",
        "standard": "#c2410c",
    }

    for ax, panel in zip(axes, panels):
        label = str(panel["label"])
        glmsingle = np.asarray(panel["glmsingle"], dtype=np.float64)
        standard = np.asarray(panel["standard"], dtype=np.float64)
        glmsingle, standard = _finite_pair(glmsingle, standard)
        edges = _bin_edges(glmsingle, standard, bins, x_min, x_max)

        glmsingle_weights = np.full(glmsingle.shape, 1.0 / glmsingle.size)
        standard_weights = np.full(standard.shape, 1.0 / standard.size)
        ax.hist(
            standard,
            bins=edges,
            weights=standard_weights,
            histtype="stepfilled",
            alpha=0.38,
            color=colors["standard"],
            edgecolor=colors["standard"],
            linewidth=1.2,
            label="Standard GLM z-score",
        )
        ax.hist(
            glmsingle,
            bins=edges,
            weights=glmsingle_weights,
            histtype="stepfilled",
            alpha=0.38,
            color=colors["glmsingle"],
            edgecolor=colors["glmsingle"],
            linewidth=1.2,
            label=str(panel["glmsingle_label"]),
        )
        ax.axvline(0, color="#777777", linewidth=0.9, linestyle="--", alpha=0.75)
        ax.axvline(
            float(np.nanmean(standard)),
            color=colors["standard"],
            linewidth=1.4,
            linestyle="-",
            alpha=0.9,
        )
        ax.axvline(
            float(np.nanmean(glmsingle)),
            color=colors["glmsingle"],
            linewidth=1.4,
            linestyle="-",
            alpha=0.9,
        )
        ax.set_title(label)
        ax.set_xlabel("Z-score")
        ax.grid(axis="y", color="#dddddd", linewidth=0.7, alpha=0.55)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

        summary["panels"][label] = {
            **_summary(glmsingle, "glmsingle"),
            **_summary(standard, "standard_glm"),
        }

    axes[0].set_ylabel("Probability")
    fig.suptitle(title, y=1.02, fontsize=13)
    fig.tight_layout()

    output_png = output_prefix.with_suffix(".png")
    output_pdf = output_prefix.with_suffix(".pdf")
    output_html = output_prefix.with_suffix(".html")
    output_json = output_prefix.with_name(f"{output_prefix.name}_summary.json")

    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    summary.update(
        {
            "png": str(output_png),
            "pdf": str(output_pdf),
            "html": str(output_html),
            "summary_json": str(output_json),
        }
    )
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_html(output_html, output_png, title, summary)
    return summary


def _write_html(
    output_html: Path,
    output_png: Path,
    title: str,
    summary: dict[str, object],
) -> None:
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
  <title>{title}</title>
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
  <h1>{title}</h1>
  <img alt="{title}" src="data:image/png;base64,{encoded_png}">
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
            "Create overlapping probability histograms from saved GLMsingle vs "
            "standard GLM voxel comparison CSV files."
        )
    )
    parser.add_argument("--typea-csv", type=Path, default=TYPEA_CSV)
    parser.add_argument("--typebcd-csv", type=Path, default=TYPEBCD_CSV)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_COMPARISON_DIR,
        help=f"Directory for histogram outputs. Default: {DEFAULT_COMPARISON_DIR}",
    )
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument(
        "--x-min",
        type=float,
        default=None,
        help="Optional fixed lower x-axis limit for all panels.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Optional fixed upper x-axis limit for all panels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    typea = _load_columns(args.typea_csv, ["glmsingle_zscore", "standard_glm_value"])
    typebcd_columns = _load_columns(
        args.typebcd_csv,
        ["TYPEB_zscore", "TYPEC_zscore", "TYPED_zscore", "standard_glm_value"],
    )

    typea_summary = _plot_probability_histograms(
        output_prefix=output_dir / f"{TYPEA_STEM}_probability_histogram",
        panels=[
            {
                "label": "TYPEA vs standard GLM",
                "glmsingle": typea["glmsingle_zscore"],
                "standard": typea["standard_glm_value"],
                "glmsingle_label": "TYPEA saturated z-score",
            }
        ],
        bins=args.bins,
        x_min=args.x_min,
        x_max=args.x_max,
        title="Probability histogram: TYPEA saturated z-score vs standard GLM",
    )

    typebcd_summary = _plot_probability_histograms(
        output_prefix=output_dir / f"{TYPEBCD_STEM}_probability_histogram",
        panels=[
            {
                "label": "TYPEB vs standard GLM",
                "glmsingle": typebcd_columns["TYPEB_zscore"],
                "standard": typebcd_columns["standard_glm_value"],
                "glmsingle_label": "TYPEB saturated z-score",
            },
            {
                "label": "TYPEC vs standard GLM",
                "glmsingle": typebcd_columns["TYPEC_zscore"],
                "standard": typebcd_columns["standard_glm_value"],
                "glmsingle_label": "TYPEC saturated z-score",
            },
            {
                "label": "TYPED vs standard GLM",
                "glmsingle": typebcd_columns["TYPED_zscore"],
                "standard": typebcd_columns["standard_glm_value"],
                "glmsingle_label": "TYPED saturated z-score",
            },
        ],
        bins=args.bins,
        x_min=args.x_min,
        x_max=args.x_max,
        title="Probability histogram: TYPEB/TYPEC/TYPED saturated z-scores vs standard GLM",
    )

    print(f"Saved TYPEA histogram PNG: {typea_summary['png']}")
    print(f"Saved TYPEA histogram PDF: {typea_summary['pdf']}")
    print(f"Saved TYPEA histogram HTML: {typea_summary['html']}")
    print(f"Saved TYPEB/C/D histogram PNG: {typebcd_summary['png']}")
    print(f"Saved TYPEB/C/D histogram PDF: {typebcd_summary['pdf']}")
    print(f"Saved TYPEB/C/D histogram HTML: {typebcd_summary['html']}")


if __name__ == "__main__":
    main()
