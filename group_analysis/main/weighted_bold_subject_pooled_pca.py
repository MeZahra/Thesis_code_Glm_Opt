#!/usr/bin/env python3
"""Run one subject-level PCA on run-mean-centered weighted BOLD trials."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

try:
    from weighted_bold_pca import (
        DEFAULT_BOLD_DIR,
        DEFAULT_WEIGHT_MAP,
        DROP_SLICES,
        MEDICATION_LABELS,
        compute_weighted_ts,
        load_weights,
        parse_bold_name,
    )
except ImportError:
    from group_analysis.main.weighted_bold_pca import (
        DEFAULT_BOLD_DIR,
        DEFAULT_WEIGHT_MAP,
        DROP_SLICES,
        MEDICATION_LABELS,
        compute_weighted_ts,
        load_weights,
        parse_bold_name,
    )


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "weighted_bold_subject_pooled_pca"
SESSION_COLORS = {
    1: "#1f77b4",
    2: "#d62728",
}


class RunTrials(NamedTuple):
    session: int
    run: int
    trials: np.ndarray


class ScoredRun(NamedTuple):
    session: int
    run: int
    scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each subject, compute weighted BOLD time series per run, remove "
            "the cleaned-run mean, reshape to trial x timepoint, pool all runs, "
            "fit one PCA(n_components=2), and plot PC1 vs PC2 by medication session."
        )
    )
    parser.add_argument("--bold-dir", type=Path, default=DEFAULT_BOLD_DIR)
    parser.add_argument("--weight-map", type=Path, default=DEFAULT_WEIGHT_MAP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject IDs to process, for example sub-pd004 sub-pd009.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def center_run_and_reshape(weighted_ts: np.ndarray) -> np.ndarray:
    keep_mask = np.ones(weighted_ts.shape[0], dtype=bool)
    for drop_slice in DROP_SLICES:
        keep_mask[drop_slice] = False

    weighted_ts_clean = weighted_ts[keep_mask]
    if weighted_ts_clean.shape != (810,):
        raise ValueError(f"Expected cleaned time series shape (810,), got {weighted_ts_clean.shape}")

    weighted_ts_centered = weighted_ts_clean - weighted_ts_clean.mean()
    return weighted_ts_centered.reshape(90, 9)


def fit_subject_pca(run_trials: list[RunTrials]) -> tuple[list[ScoredRun], np.ndarray]:
    if not run_trials:
        raise ValueError("Cannot fit PCA without run trials.")

    pooled_trials = np.vstack([item.trials for item in run_trials])
    if pooled_trials.shape[0] < 2:
        raise ValueError(f"Need at least two pooled trials for PCA, got {pooled_trials.shape[0]}")

    pca = PCA(n_components=2)
    pooled_scores = pca.fit_transform(pooled_trials)

    scored_runs: list[ScoredRun] = []
    start = 0
    for item in run_trials:
        stop = start + item.trials.shape[0]
        scored_runs.append(ScoredRun(item.session, item.run, pooled_scores[start:stop]))
        start = stop

    return scored_runs, pca.explained_variance_ratio_


def session_label(session: int) -> str:
    medication = MEDICATION_LABELS.get(session, f"ses-{session}")
    return f"{medication} (ses-{session})"


def save_subject_figure(
    subject: str,
    scored_runs: list[ScoredRun],
    explained_variance_ratio: np.ndarray,
    out_dir: Path,
    dpi: int,
) -> Path | None:
    if not scored_runs:
        return None

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    plotted = False

    for session in sorted({item.session for item in scored_runs}):
        session_scores = [
            item.scores
            for item in scored_runs
            if item.session == session and item.scores.size > 0
        ]
        if not session_scores:
            continue

        scores = np.vstack(session_scores)
        ax.scatter(
            scores[:, 0],
            scores[:, 1],
            s=22,
            alpha=0.75,
            color=SESSION_COLORS.get(session, "0.35"),
            label=f"{session_label(session)} n={scores.shape[0]}",
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.axhline(0, color="0.85", linewidth=0.8)
    ax.axvline(0, color="0.85", linewidth=0.8)
    ax.set_title(f"{subject}: run-mean-centered pooled PCA")
    ax.set_xlabel(f"PC1 ({explained_variance_ratio[0] * 100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({explained_variance_ratio[1] * 100:.1f}% var)")
    ax.legend(frameon=False)

    out_path = out_dir / f"{subject}_weighted_bold_subject_pooled_pca.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    weights = load_weights(args.weight_map)
    bold_paths = sorted(args.bold_dir.glob("*_bold_corrected_smoothed_mnireg-2mm.nii.gz"))
    if not bold_paths:
        raise FileNotFoundError(f"No BOLD NIfTI files found in {args.bold_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested_subjects = set(args.subjects) if args.subjects is not None else None
    trials_by_subject: dict[str, list[RunTrials]] = defaultdict(list)

    for bold_path in bold_paths:
        subject, session, run = parse_bold_name(bold_path)
        if requested_subjects is not None and subject not in requested_subjects:
            continue
        weighted_ts = compute_weighted_ts(bold_path, weights)
        trials = center_run_and_reshape(weighted_ts)
        trials_by_subject[subject].append(RunTrials(session, run, trials))
        print(
            f"Processed {bold_path.name}: weighted_ts={weighted_ts.shape}, "
            f"centered_trials={trials.shape}"
        )

    if not trials_by_subject:
        requested = ", ".join(sorted(requested_subjects)) if requested_subjects else "all subjects"
        raise FileNotFoundError(f"No matching BOLD NIfTI files found for {requested}")

    for subject, run_trials in sorted(trials_by_subject.items()):
        sorted_trials = sorted(run_trials, key=lambda item: (item.session, item.run))
        scored_runs, explained_variance_ratio = fit_subject_pca(sorted_trials)
        out_path = save_subject_figure(
            subject,
            scored_runs,
            explained_variance_ratio,
            args.out_dir,
            args.dpi,
        )
        if out_path is not None:
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
