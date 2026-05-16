#!/usr/bin/env python3
"""Create paper-friendly ROI burden heatmaps from significant-ROI tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[2]
DELTA_ROOT = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "GVS_effects"
    / "gvs_similarity_hemi"
    / "roi_condition_reference_deltas"
)
TABLES_DIR = DELTA_ROOT / "tables" / "without_unassigned"
PLOTS_DIR = DELTA_ROOT / "plots" / "without_unassigned"

SESSION_RE = re.compile(r"^GVS(\d+)$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"^sub-pd(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class HeatmapSpec:
    prefix: str
    cmap_name: str


SPECS = (
    HeatmapSpec(
        prefix="off_condition_minus_sham_off_roi_mean_delta",
        cmap_name="Blues",
    ),
    HeatmapSpec(
        prefix="on_condition_minus_sham_on_roi_mean_delta",
        cmap_name="Oranges",
    ),
)


def _subject_sort_key(subject: str) -> tuple[int, str]:
    match = SUBJECT_RE.match(str(subject))
    if match:
        return (int(match.group(1)), str(subject))
    return (10**9, str(subject))


def _session_sort_key(label: str) -> tuple[int, str]:
    match = SESSION_RE.match(str(label))
    if match:
        return (int(match.group(1)), str(label))
    return (10**9, str(label))


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.casefold().isin({"true", "1", "yes"})


def _load_stats(prefix: str) -> pd.DataFrame:
    path = TABLES_DIR / f"{prefix}_ttest_stats_by_subject_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing source table: {path}")
    stats = pd.read_csv(path)
    required = {"subject", "target_condition_label", "roi_label", "significant_fdr"}
    missing = required.difference(stats.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return stats


def _active_session_labels(stats_frames: list[pd.DataFrame]) -> list[str]:
    labels: set[str] = set()
    for stats in stats_frames:
        labels.update(
            str(label)
            for label in stats["target_condition_label"].dropna().unique().tolist()
            if str(label).casefold() != "sham"
        )
    return sorted(labels, key=_session_sort_key)


def _significant_roi_sets(
    stats: pd.DataFrame,
    *,
    subjects: list[str],
    sessions: list[str],
) -> dict[tuple[str, str], set[str]]:
    significant = stats.loc[_as_bool(stats["significant_fdr"])].copy()
    significant["subject"] = significant["subject"].astype(str)
    significant["target_condition_label"] = significant["target_condition_label"].astype(str)
    significant["roi_label"] = significant["roi_label"].astype(str)

    roi_sets: dict[tuple[str, str], set[str]] = {
        (subject, session): set() for subject in subjects for session in sessions
    }
    for (subject, session), cell_df in significant.groupby(
        ["subject", "target_condition_label"],
        dropna=False,
        observed=False,
        sort=False,
    ):
        key = (str(subject), str(session))
        if key in roi_sets:
            roi_sets[key] = set(cell_df["roi_label"].tolist())
    return roi_sets


def _count_matrix(
    roi_sets: dict[tuple[str, str], set[str]],
    *,
    subjects: list[str],
    sessions: list[str],
) -> np.ndarray:
    return np.array(
        [[len(roi_sets[(subject, session)]) for session in sessions] for subject in subjects],
        dtype=float,
    )


def _write_count_csv(
    counts: np.ndarray,
    *,
    subjects: list[str],
    sessions: list[str],
    out_path: Path,
) -> None:
    out = pd.DataFrame(counts.astype(int), index=subjects, columns=sessions)
    out.index.name = "subject"
    out.to_csv(out_path)


def _plot_roi_burden_heatmap(
    counts: np.ndarray,
    *,
    subjects: list[str],
    sessions: list[str],
    cmap_name: str,
    vmax: int,
    out_stem: Path,
) -> None:
    fig_h = max(7.2, 0.48 * len(subjects) + 2.1)
    fig_w = max(10.0, 0.95 * len(sessions) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    image = ax.imshow(
        counts,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_name,
        vmin=0,
        vmax=max(1, vmax),
    )

    ax.set_xlabel("GVS session", fontsize=13)
    ax.set_ylabel("Subject", fontsize=13)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=11)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=11)
    ax.tick_params(axis="both", which="major", length=4, width=0.9)

    ax.set_xticks(np.arange(-0.5, len(sessions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = max(1.0, 0.62 * float(max(1, vmax)))
    for row_idx, subject in enumerate(subjects):
        for col_idx, session in enumerate(sessions):
            value = int(counts[row_idx, col_idx])
            if value == 0:
                text = "-"
                color = "#7a7a7a"
            else:
                text = str(value)
                color = "white" if value >= threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=11,
                color=color,
            )

    cbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("No. of significant ROIs", fontsize=12)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def _draw_heatmap_panel(
    ax: plt.Axes,
    counts: np.ndarray,
    *,
    subjects: list[str],
    sessions: list[str],
    cmap_name: str,
    vmax: int,
    show_ylabel: bool = True,
) -> plt.AxesImage:
    image = ax.imshow(
        counts,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_name,
        vmin=0,
        vmax=max(1, vmax),
    )

    if show_ylabel:
        ax.set_ylabel("Subject", fontsize=11)
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_xticklabels(sessions, fontsize=10)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=10)
    ax.tick_params(axis="both", which="major", length=4, width=0.9)

    ax.set_xticks(np.arange(-0.5, len(sessions), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(subjects), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = max(1.0, 0.62 * float(max(1, vmax)))
    for row_idx, subject in enumerate(subjects):
        for col_idx, session in enumerate(sessions):
            value = int(counts[row_idx, col_idx])
            if value == 0:
                text = "-"
                color = "#7a7a7a"
            else:
                text = str(value)
                color = "white" if value >= threshold else "black"
            ax.text(
                col_idx,
                row_idx,
                text,
                ha="center",
                va="center",
                fontsize=10,
                color=color,
            )
    return image


def _plot_combined_roi_burden_heatmaps(
    panels: list[tuple[list[str], np.ndarray, str]],
    *,
    sessions: list[str],
    vmax: int,
    out_stem: Path,
) -> None:
    fig_w = max(15.0, 1.7 * len(sessions) + 4.0)
    fig_h = max(7.4, max(0.48 * len(subjects) for subjects, _, _ in panels) + 2.1)
    fig, axes = plt.subplots(1, len(panels), figsize=(fig_w, fig_h))
    axes_array = np.atleast_1d(axes)

    for panel_idx, (ax, (subjects, counts, cmap_name)) in enumerate(
        zip(axes_array, panels, strict=True)
    ):
        image = _draw_heatmap_panel(
            ax,
            counts,
            subjects=subjects,
            sessions=sessions,
            cmap_name=cmap_name,
            vmax=vmax,
            show_ylabel=panel_idx == 0,
        )
        ax.set_xlabel("GVS session", fontsize=11)
        cbar = fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02)
        if panel_idx != 0:
            cbar.set_label("No. of significant ROIs", fontsize=10)
        cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    fig.savefig(out_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_by_prefix = {spec.prefix: _load_stats(spec.prefix) for spec in SPECS}
    sessions = _active_session_labels(list(stats_by_prefix.values()))

    prepared: dict[str, tuple[list[str], dict[tuple[str, str], set[str]], np.ndarray]] = {}
    max_count = 0
    for spec in SPECS:
        stats = stats_by_prefix[spec.prefix]
        subjects = sorted(stats["subject"].astype(str).unique().tolist(), key=_subject_sort_key)
        roi_sets = _significant_roi_sets(stats, subjects=subjects, sessions=sessions)
        counts = _count_matrix(roi_sets, subjects=subjects, sessions=sessions)
        max_count = max(max_count, int(np.nanmax(counts)) if counts.size else 0)
        prepared[spec.prefix] = (subjects, roi_sets, counts)

    for spec in SPECS:
        subjects, roi_sets, counts = prepared[spec.prefix]

        stem = f"{spec.prefix}_subject_session_roi_burden_heatmap"
        _write_count_csv(
            counts,
            subjects=subjects,
            sessions=sessions,
            out_path=PLOTS_DIR / f"{stem}.csv",
        )
        _plot_roi_burden_heatmap(
            counts,
            subjects=subjects,
            sessions=sessions,
            cmap_name=spec.cmap_name,
            vmax=max_count,
            out_stem=PLOTS_DIR / stem,
        )
        print(PLOTS_DIR / f"{stem}.png")
        print(PLOTS_DIR / f"{stem}.pdf")

    combined_stem = "off_on_condition_minus_sham_subject_session_roi_burden_heatmaps"
    combined_panels = [
        (prepared[spec.prefix][0], prepared[spec.prefix][2], spec.cmap_name)
        for spec in SPECS
    ]
    _plot_combined_roi_burden_heatmaps(
        combined_panels,
        sessions=sessions,
        vmax=max_count,
        out_stem=PLOTS_DIR / combined_stem,
    )
    print(PLOTS_DIR / f"{combined_stem}.png")
    print(PLOTS_DIR / f"{combined_stem}.pdf")


if __name__ == "__main__":
    main()
