#!/usr/bin/env python3
"""Plot one metric heatmap grid for subjects and sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show metric heatmaps in comparison-oriented figures: "
            "(1) all subjects for one session, "
            "(2) session-by-subject global scale, "
            "(3) session-by-subject row-scaled, and "
            "(4) delta vs reference session. "
            "X ticks are shown only on the last row and Y ticks only on the first column."
        )
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(
            "results/connectivity/between_sessions/subject_session_metrics/"
            "roi_edge_network/advanced_metrics"
        ),
        help="Directory that contains sub-pd*_ses-* metric folders.",
    )
    parser.add_argument("--session", type=int, default=1, help="Session number to plot.")
    parser.add_argument(
        "--metric",
        default="mutual_information_ksg",
        help="Metric folder/file name inside each subject/session directory.",
    )
    parser.add_argument(
        "--exclude-subject-ids",
        default="17",
        help=(
            "Comma-separated numeric subject IDs to exclude (e.g., '17,23'). "
            "Default excludes subject 17."
        ),
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=4,
        help="Number of subplot columns in the output grid.",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=12,
        help="Maximum number of ROI tick labels to show per axis.",
    )
    parser.add_argument("--tick-fontsize", type=float, default=5.0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--delta-cmap", default="coolwarm")
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.0,
        help="Shared lower color limit for all subject heatmaps.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Shared upper color limit; default uses the global subject max.",
    )
    parser.add_argument(
        "--value-percentile",
        type=float,
        default=99.5,
        help="Robust percentile for automatic heatmap vmax when --vmax is not set.",
    )
    parser.add_argument(
        "--delta-percentile",
        type=float,
        default=99.0,
        help="Robust percentile for absolute delta scaling in difference heatmaps.",
    )
    parser.add_argument(
        "--reference-session",
        type=int,
        default=None,
        help="Reference session for delta figure. Default uses the smallest session id.",
    )
    parser.add_argument(
        "--delta-subject-rows",
        type=int,
        default=2,
        help="Number of subject rows per session block in the delta figure.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Default: <base-dir>/session<id>_<metric>_all_subjects_grid.png",
    )
    parser.add_argument(
        "--session-subject-output",
        type=Path,
        default=None,
        help=(
            "Output path for session-by-subject figure. "
            "Default: <base-dir>/all_sessions_<metric>_session_subject_grid.png"
        ),
    )
    parser.add_argument(
        "--row-scaled-output",
        type=Path,
        default=None,
        help=(
            "Output path for row-scaled session-by-subject figure. "
            "Default: <base-dir>/all_sessions_<metric>_session_subject_grid_row_scaled.png"
        ),
    )
    parser.add_argument(
        "--delta-output",
        type=Path,
        default=None,
        help=(
            "Output path for delta-vs-reference figure. "
            "Default: <base-dir>/all_sessions_<metric>_delta_vs_ses-<ref>_subject_grid.png"
        ),
    )
    parser.add_argument("--show", action="store_true", help="Display figure interactively.")
    return parser.parse_args()


def _subject_tag_sort_key(subject_tag: str) -> tuple[int, str]:
    numeric_part = subject_tag.replace("sub-pd", "")
    try:
        return int(numeric_part), subject_tag
    except ValueError:
        return 10**9, subject_tag


def _subject_sort_key(subject_dir: Path) -> tuple[int, str]:
    subject_tag = subject_dir.name.split("_", 1)[0]
    return _subject_tag_sort_key(subject_tag)


def _parse_subject_session(folder_name: str) -> tuple[str, int] | None:
    if "_ses-" not in folder_name:
        return None
    subject_tag, session_part = folder_name.split("_ses-", 1)
    if not subject_tag:
        return None
    try:
        session = int(session_part)
    except ValueError:
        return None
    return subject_tag, session


def _load_labels(label_path: Path) -> list[str]:
    return [line.strip() for line in label_path.read_text().splitlines() if line.strip()]


def _subject_numeric_id(subject_tag: str) -> int | None:
    numeric_part = subject_tag.replace("sub-pd", "")
    try:
        return int(numeric_part)
    except ValueError:
        return None


def _parse_subject_id_set(spec: str) -> set[int]:
    if not str(spec).strip():
        return set()
    out: set[int] = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not token:
            continue
        out.add(int(token))
    return out


def _build_tick_subset(labels: list[str], max_ticks: int) -> tuple[np.ndarray, list[str]]:
    n = len(labels)
    if n == 0:
        return np.array([], dtype=int), []
    if max_ticks <= 0 or n <= max_ticks:
        tick_idx = np.arange(n, dtype=int)
    else:
        tick_idx = np.linspace(0, n - 1, num=max_ticks, dtype=int)
        tick_idx = np.unique(tick_idx)
    return tick_idx, [labels[int(idx)] for idx in tick_idx]


def _finite_values(arrays: list[np.ndarray]) -> np.ndarray:
    chunks = []
    for arr in arrays:
        values = np.asarray(arr, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            chunks.append(finite.ravel())
    if not chunks:
        return np.array([], dtype=np.float64)
    return np.concatenate(chunks)


def _robust_vmax(arrays: list[np.ndarray], percentile: float, vmin: float) -> float:
    finite = _finite_values(arrays)
    if finite.size == 0:
        return float(vmin + 1e-6)

    percentile = float(np.clip(percentile, 0.0, 100.0))
    vmax = float(np.percentile(finite, percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.max(finite))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(vmin + 1e-6)
    return vmax


def main() -> None:
    args = _parse_args()
    excluded_subject_ids = _parse_subject_id_set(args.exclude_subject_ids)

    by_session: dict[int, dict[str, tuple[Path, Path]]] = {}
    for subject_dir in sorted(args.base_dir.glob("sub-pd*_ses-*"), key=_subject_sort_key):
        parsed = _parse_subject_session(subject_dir.name)
        if parsed is None:
            continue
        subject_tag, session = parsed
        metric_dir = subject_dir / args.metric
        matrix_path = metric_dir / f"{args.metric}.npy"
        label_path = metric_dir / f"{args.metric}_connectome.labels.txt"
        if matrix_path.exists() and label_path.exists():
            by_session.setdefault(session, {})[subject_tag] = (matrix_path, label_path)

    if excluded_subject_ids:
        filtered_by_session: dict[int, dict[str, tuple[Path, Path]]] = {}
        for session, subject_map in by_session.items():
            kept = {
                subject_tag: paths
                for subject_tag, paths in subject_map.items()
                if _subject_numeric_id(subject_tag) not in excluded_subject_ids
            }
            if kept:
                filtered_by_session[session] = kept
        by_session = filtered_by_session

    if not by_session or args.session not in by_session:
        raise FileNotFoundError(
            f"No matrices found under {args.base_dir} for session {args.session} and metric {args.metric}."
        )

    matrix_cache: dict[tuple[int, str], np.ndarray] = {}
    for session, subject_map in by_session.items():
        for subject_tag, (matrix_path, _) in subject_map.items():
            matrix = np.load(matrix_path)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Expected square 2D matrix in {matrix_path}, got {matrix.shape}.")
            matrix_cache[(session, subject_tag)] = np.asarray(matrix, dtype=np.float64)

    session_subjects = sorted(by_session[args.session].keys(), key=_subject_tag_sort_key)
    matrices: list[tuple[str, np.ndarray]] = []
    for subject_tag in session_subjects:
        matrix = matrix_cache[(args.session, subject_tag)]
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"Expected square matrix for {subject_tag} in session {args.session}.")
        matrices.append((subject_tag, np.asarray(matrix, dtype=np.float64)))

    matrix_shape = matrices[0][1].shape
    if any(matrix.shape != matrix_shape for matrix in matrix_cache.values()):
        raise ValueError("All matrices must share the same shape to be plotted together.")

    first_session = sorted(by_session.keys())[0]
    first_subject = sorted(by_session[first_session].keys(), key=_subject_tag_sort_key)[0]
    labels = _load_labels(by_session[first_session][first_subject][1])
    if len(labels) != matrix_shape[0]:
        labels = [str(idx) for idx in range(matrix_shape[0])]
    tick_idx, tick_labels = _build_tick_subset(labels, int(args.max_ticks))

    n_subjects = len(matrices)
    n_cols = max(1, int(args.n_cols))
    n_rows = int(np.ceil(n_subjects / float(n_cols)))

    vmin = float(args.vmin)
    vmax = (
        _robust_vmax(
            arrays=list(matrix_cache.values()),
            percentile=float(args.value_percentile),
            vmin=vmin,
        )
        if args.vmax is None
        else float(args.vmax)
    )
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 3.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    image_handle = None
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n_subjects:
            ax.axis("off")
            continue

        row_idx, col_idx = divmod(idx, n_cols)
        subject_tag, matrix = matrices[idx]
        image_handle = ax.imshow(
            np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
            cmap=args.cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
        )
        ax.set_title(subject_tag, fontsize=9)

        if row_idx == n_rows - 1:
            ax.set_xticks(tick_idx)
            ax.set_xticklabels(tick_labels, rotation=90, fontsize=args.tick_fontsize)
        else:
            ax.set_xticks([])

        if col_idx == 0:
            ax.set_yticks(tick_idx)
            ax.set_yticklabels(tick_labels, fontsize=args.tick_fontsize)
        else:
            ax.set_yticks([])

    if image_handle is not None:
        fig.colorbar(
            image_handle,
            ax=axes.ravel().tolist(),
            fraction=0.02,
            pad=0.01,
            label=f"{args.metric} value",
        )

    fig.suptitle(
        f"All subjects, session {args.session}: {args.metric}",
        fontsize=14,
    )

    output_path = args.output
    if output_path is None:
        output_path = args.base_dir / f"session{args.session}_{args.metric}_all_subjects_grid.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Saved figure: {output_path}", flush=True)

    sessions_sorted = sorted(by_session.keys())
    subjects_sorted = sorted(
        {subject for subject_map in by_session.values() for subject in subject_map.keys()},
        key=_subject_tag_sort_key,
    )
    n_session_rows = len(sessions_sorted)
    n_subject_cols = len(subjects_sorted)

    fig2, axes2 = plt.subplots(
        n_session_rows,
        n_subject_cols,
        figsize=(3.0 * n_subject_cols, 3.2 * n_session_rows),
        squeeze=False,
        constrained_layout=True,
    )

    image_handle2 = None
    for row_idx, session in enumerate(sessions_sorted):
        for col_idx, subject_tag in enumerate(subjects_sorted):
            ax = axes2[row_idx, col_idx]
            matrix = matrix_cache.get((session, subject_tag))
            if matrix is None:
                ax.axis("off")
                continue

            image_handle2 = ax.imshow(
                np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
                cmap=args.cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )

            if row_idx == 0:
                ax.set_title(subject_tag, fontsize=9)

            if row_idx == n_session_rows - 1:
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=args.tick_fontsize)
            else:
                ax.set_xticks([])

            if col_idx == 0:
                ax.set_yticks(tick_idx)
                ax.set_yticklabels(tick_labels, fontsize=args.tick_fontsize)
                ax.set_ylabel(f"ses-{session}", fontsize=9)
            else:
                ax.set_yticks([])

    if image_handle2 is not None:
        fig2.colorbar(
            image_handle2,
            ax=axes2.ravel().tolist(),
            fraction=0.02,
            pad=0.01,
            label=f"{args.metric} value",
        )

    fig2.suptitle(
        f"Session (rows) x Subject (columns): {args.metric}",
        fontsize=14,
    )

    session_subject_output = args.session_subject_output
    if session_subject_output is None:
        session_subject_output = (
            args.base_dir / f"all_sessions_{args.metric}_session_subject_grid.png"
        )
    session_subject_output.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(session_subject_output, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Saved figure: {session_subject_output}", flush=True)

    # Row-scaled figure: one robust scale per session row.
    row_limits: dict[int, tuple[float, float]] = {}
    for session in sessions_sorted:
        row_matrices = [
            matrix_cache[(session, subject_tag)]
            for subject_tag in subjects_sorted
            if (session, subject_tag) in matrix_cache
        ]
        row_limits[session] = (
            vmin,
            _robust_vmax(
                arrays=row_matrices,
                percentile=float(args.value_percentile),
                vmin=vmin,
            ),
        )

    fig3, axes3 = plt.subplots(
        n_session_rows,
        n_subject_cols,
        figsize=(3.0 * n_subject_cols, 3.2 * n_session_rows),
        squeeze=False,
        constrained_layout=True,
    )

    row_first_image: dict[int, object] = {}
    for row_idx, session in enumerate(sessions_sorted):
        row_vmin, row_vmax = row_limits[session]
        for col_idx, subject_tag in enumerate(subjects_sorted):
            ax = axes3[row_idx, col_idx]
            matrix = matrix_cache.get((session, subject_tag))
            if matrix is None:
                ax.axis("off")
                continue

            image = ax.imshow(
                np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0),
                cmap=args.cmap,
                vmin=row_vmin,
                vmax=row_vmax,
                aspect="auto",
            )
            if row_idx not in row_first_image:
                row_first_image[row_idx] = image

            if row_idx == 0:
                ax.set_title(subject_tag, fontsize=9)

            if row_idx == n_session_rows - 1:
                ax.set_xticks(tick_idx)
                ax.set_xticklabels(tick_labels, rotation=90, fontsize=args.tick_fontsize)
            else:
                ax.set_xticks([])

            if col_idx == 0:
                ax.set_yticks(tick_idx)
                ax.set_yticklabels(tick_labels, fontsize=args.tick_fontsize)
                ax.set_ylabel(f"ses-{session}\nscale:[{row_vmin:.3g}, {row_vmax:.3g}]", fontsize=8)
            else:
                ax.set_yticks([])

    for row_idx, session in enumerate(sessions_sorted):
        row_image = row_first_image.get(row_idx)
        if row_image is None:
            continue
        fig3.colorbar(
            row_image,
            ax=axes3[row_idx, :].tolist(),
            fraction=0.02,
            pad=0.01,
            label=f"ses-{session} {args.metric}",
        )

    fig3.suptitle(
        f"Session (rows) x Subject (columns) [row-scaled]: {args.metric}",
        fontsize=14,
    )

    row_scaled_output = args.row_scaled_output
    if row_scaled_output is None:
        row_scaled_output = (
            args.base_dir / f"all_sessions_{args.metric}_session_subject_grid_row_scaled.png"
        )
    row_scaled_output.parent.mkdir(parents=True, exist_ok=True)
    fig3.savefig(row_scaled_output, dpi=int(args.dpi), bbox_inches="tight")
    print(f"Saved figure: {row_scaled_output}", flush=True)

    # Delta figure: direct session change vs reference session per subject.
    reference_session = (
        int(args.reference_session)
        if args.reference_session is not None
        else int(sessions_sorted[0])
    )
    if reference_session not in sessions_sorted:
        raise ValueError(
            f"Reference session {reference_session} not found. Available: {sessions_sorted}"
        )

    delta_sessions = [session for session in sessions_sorted if session != reference_session]
    if delta_sessions:
        delta_cache: dict[tuple[int, str], np.ndarray] = {}
        delta_arrays: list[np.ndarray] = []
        for session in delta_sessions:
            for subject_tag in subjects_sorted:
                ref_key = (reference_session, subject_tag)
                cur_key = (session, subject_tag)
                if ref_key not in matrix_cache or cur_key not in matrix_cache:
                    continue
                delta = matrix_cache[cur_key] - matrix_cache[ref_key]
                delta_cache[(session, subject_tag)] = delta
                delta_arrays.append(delta)

        if delta_arrays:
            delta_abs_values = np.abs(_finite_values(delta_arrays))
            if delta_abs_values.size == 0:
                delta_max = 1e-6
            else:
                delta_percentile = float(np.clip(args.delta_percentile, 0.0, 100.0))
                delta_max = float(np.percentile(delta_abs_values, delta_percentile))
                if not np.isfinite(delta_max) or delta_max <= 0.0:
                    delta_max = float(np.max(delta_abs_values))
                if not np.isfinite(delta_max) or delta_max <= 0.0:
                    delta_max = 1e-6

            delta_subject_rows = max(1, int(args.delta_subject_rows))
            delta_subject_cols = int(np.ceil(len(subjects_sorted) / float(delta_subject_rows)))
            total_delta_rows = len(delta_sessions) * delta_subject_rows

            fig4, axes4 = plt.subplots(
                total_delta_rows,
                delta_subject_cols,
                figsize=(3.0 * delta_subject_cols, 3.2 * total_delta_rows),
                squeeze=False,
                constrained_layout=True,
            )

            image_handle4 = None
            for session_idx, session in enumerate(delta_sessions):
                row_offset = session_idx * delta_subject_rows

                for subject_idx, subject_tag in enumerate(subjects_sorted):
                    local_row = subject_idx // delta_subject_cols
                    col_idx = subject_idx % delta_subject_cols
                    row_idx = row_offset + local_row
                    ax = axes4[row_idx, col_idx]
                    delta = delta_cache.get((session, subject_tag))
                    if delta is None:
                        ax.axis("off")
                        continue

                    image_handle4 = ax.imshow(
                        np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0),
                        cmap=args.delta_cmap,
                        vmin=-delta_max,
                        vmax=delta_max,
                        aspect="auto",
                    )

                    if session_idx == 0:
                        ax.set_title(subject_tag, fontsize=9)

                    if row_idx == total_delta_rows - 1:
                        ax.set_xticks(tick_idx)
                        ax.set_xticklabels(tick_labels, rotation=90, fontsize=args.tick_fontsize)
                    else:
                        ax.set_xticks([])

                    if col_idx == 0:
                        ax.set_yticks(tick_idx)
                        ax.set_yticklabels(tick_labels, fontsize=args.tick_fontsize)
                        if local_row == 0:
                            ax.set_ylabel(f"ses-{session} - ses-{reference_session}", fontsize=9)
                    else:
                        ax.set_yticks([])

                # Hide trailing unused cells for this session block.
                n_used = len(subjects_sorted)
                n_cells = delta_subject_rows * delta_subject_cols
                for empty_idx in range(n_used, n_cells):
                    local_row = empty_idx // delta_subject_cols
                    col_idx = empty_idx % delta_subject_cols
                    row_idx = row_offset + local_row
                    axes4[row_idx, col_idx].axis("off")

            if image_handle4 is not None:
                fig4.colorbar(
                    image_handle4,
                    ax=axes4.ravel().tolist(),
                    fraction=0.02,
                    pad=0.01,
                    label=f"Delta {args.metric}",
                )

            fig4.suptitle(
                f"Session change vs ses-{reference_session} (rows) x Subject (columns): {args.metric}",
                fontsize=14,
            )

            delta_output = args.delta_output
            if delta_output is None:
                delta_output = (
                    args.base_dir
                    / f"all_sessions_{args.metric}_delta_vs_ses-{reference_session}_subject_grid.png"
                )
            delta_output.parent.mkdir(parents=True, exist_ok=True)
            fig4.savefig(delta_output, dpi=int(args.dpi), bbox_inches="tight")
            print(f"Saved figure: {delta_output}", flush=True)
            plt.close(fig4)
        else:
            print(
                "Skipped delta figure: no subjects had both reference and comparison sessions.",
                flush=True,
            )
    else:
        print("Skipped delta figure: only one session available.", flush=True)

    if args.show:
        plt.show()
    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


if __name__ == "__main__":
    main()
