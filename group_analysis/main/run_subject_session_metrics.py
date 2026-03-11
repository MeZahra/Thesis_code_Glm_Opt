#!/usr/bin/env python3
"""Per-subject/session connectivity metrics with run1+run2 concatenation.

Workflow:
1) Split selected_beta_trials.npy into one file per subject-session label
   (`sub-*_ses-*`) by concatenating run1 and run2 trials.
2) Run ROI-edge connectivity + advanced metrics for each subject-session file.
3) Average advanced-metric matrices across subjects for each session.

All outputs are written under:
  results/connectivity/between_sessions/subject_session_metrics/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

DEFAULT_INPUT_BETA = REPO_ROOT / "results" / "connectivity" / "data" / "selected_beta_trials.npy"
DEFAULT_INPUT_VOXEL_INDICES = (
    REPO_ROOT / "results" / "connectivity" / "data" / "selected_voxel_indices.npz"
)
DEFAULT_MANIFEST = REPO_ROOT / "results" / "connectivity" / "tmp" / "concat_manifest_group.tsv"
DEFAULT_OUT_ROOT = HERE / "subject_session_metrics"

DEFAULT_METRICS = ",".join(
    [
        "wavelet_transform_coherence",
        "partial_correlation",
        "mutual_information",
        "mutual_information_ksg",
        "linear_correlation_network",
        "instantaneous_phase_sync",
    ]
)

LABEL_RE = re.compile(r"^(sub-[^_]+)_ses-(\d+)$")


@dataclass
class SubjectSessionSplit:
    label: str
    subject: str
    session: int
    runs: list[int]
    columns: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute non-Granger connectivity metrics for each subject/session "
            "(run1+run2 concatenated) and average across subjects per session."
        )
    )
    parser.add_argument("--input-beta", type=Path, default=DEFAULT_INPUT_BETA)
    parser.add_argument("--input-voxel-indices", type=Path, default=DEFAULT_INPUT_VOXEL_INDICES)
    parser.add_argument(
        "--voxel-weight-img",
        type=Path,
        default=None,
        help="Optional NIfTI voxel-weight map applied rowwise before connectivity.",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument(
        "--advanced-metrics",
        default=DEFAULT_METRICS,
        help="Comma-separated non-Granger metrics for roi_edge_connectivity.py",
    )
    parser.add_argument(
        "--allow-partial-runs",
        action="store_true",
        help="Include subject-session labels with only one run. Default requires run1 and run2.",
    )
    parser.add_argument(
        "--skip-connectivity",
        action="store_true",
        help="Only create per-subject/session split files and metadata.",
    )
    parser.add_argument(
        "--skip-session-average",
        action="store_true",
        help="Skip averaging advanced metrics across subjects per session.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files and copied metadata.",
    )
    return parser.parse_args()


def build_subject_session_splits(
    manifest_path: Path,
    n_trials_total: int,
    allow_partial_runs: bool,
) -> list[SubjectSessionSplit]:
    manifest_df = pd.read_csv(manifest_path, sep="\t")
    needed = {"offset_start", "offset_end", "sub_tag", "ses", "run"}
    missing = needed - set(manifest_df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns {sorted(missing)}: {manifest_path}")

    coverage = np.zeros(n_trials_total, dtype=np.int16)
    store: dict[tuple[str, int], dict[str, list[int] | set[int]]] = {}

    for row in manifest_df.itertuples(index=False):
        start = int(row.offset_start)
        end = int(row.offset_end)
        sub = str(row.sub_tag)
        ses = int(row.ses)
        run = int(row.run)

        if start < 0 or end < start or end > n_trials_total:
            raise ValueError(
                f"Invalid manifest offsets start={start} end={end} for total trials {n_trials_total}"
            )

        key = (sub, ses)
        if key not in store:
            store[key] = {"cols": [], "runs": set()}
        run_cols = np.arange(start, end, dtype=np.int64)
        store[key]["cols"].extend(run_cols.tolist())
        store[key]["runs"].add(run)
        coverage[run_cols] += 1

    if np.any(coverage != 1):
        bad = np.flatnonzero(coverage != 1)
        raise RuntimeError(
            f"Manifest coverage invalid: {bad.size} trial columns are not covered exactly once."
        )

    splits: list[SubjectSessionSplit] = []
    for (sub, ses), payload in sorted(store.items(), key=lambda x: (x[0][0], x[0][1])):
        runs = sorted(list(payload["runs"]))  # type: ignore[arg-type]
        if (not allow_partial_runs) and (set(runs) != {1, 2}):
            continue
        cols = np.asarray(payload["cols"], dtype=np.int64)  # type: ignore[arg-type]
        if cols.size > 1 and np.any(cols[1:] < cols[:-1]):
            cols = np.sort(cols)
        label = f"{sub}_ses-{ses}"
        splits.append(
            SubjectSessionSplit(
                label=label,
                subject=sub,
                session=ses,
                runs=runs,
                columns=cols,
            )
        )
    return splits


def write_split_files(
    input_beta: Path,
    data_dir: Path,
    splits: list[SubjectSessionSplit],
    chunk_size: int,
    overwrite: bool,
) -> None:
    beta = np.load(input_beta, mmap_mode="r")
    if beta.ndim != 2:
        raise ValueError(f"Expected 2D beta matrix, got {beta.shape}")

    n_vox, n_trials = beta.shape
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        cols = split.columns
        if cols.size == 0:
            continue
        if int(cols.max()) >= n_trials:
            raise ValueError(
                f"{split.label} includes trial index {int(cols.max())} outside [0, {n_trials - 1}]"
            )
        out_path = data_dir / f"selected_beta_trials_{split.label}.npy"
        if out_path.exists() and not overwrite:
            print(f"Keeping existing split file: {out_path}")
            continue

        out_mm = open_memmap(
            out_path,
            mode="w+",
            dtype=beta.dtype,
            shape=(n_vox, cols.size),
        )
        step = int(max(1, chunk_size))
        for start in range(0, n_vox, step):
            stop = min(start + step, n_vox)
            out_mm[start:stop, :] = beta[start:stop, :][:, cols]
        out_mm.flush()
        del out_mm
        print(
            f"Saved {split.label}: {out_path} | shape=({n_vox}, {cols.size}) "
            f"runs={split.runs}",
            flush=True,
        )

    index_map = {split.label: split.columns for split in splits}
    np.savez(data_dir / "selected_beta_trials_subject_session_column_indices.npz", **index_map)


def run_connectivity_pipeline(
    data_dir: Path,
    voxel_indices_path: Path,
    out_dir: Path,
    advanced_metrics: str,
    voxel_weight_img: Path | None,
) -> None:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "group_analysis" / "roi_edge_connectivity.py"),
        "--data-dir",
        str(data_dir),
        "--beta-pattern",
        "selected_beta_trials_sub-*_ses-*.npy",
        "--voxel-indices-path",
        str(voxel_indices_path),
        "--out-dir",
        str(out_dir),
        "--advanced-metrics-out-subdir",
        "advanced_metrics",
        "--advanced-metrics",
        str(advanced_metrics),
    ]
    if voxel_weight_img is not None:
        cmd.extend(["--voxel-weight-img", str(voxel_weight_img)])
    print("Running per-subject/session ROI-edge pipeline:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def _write_heatmap(matrix: np.ndarray, out_png: Path, title: str, cbar_label: str = "Value") -> None:
    vals = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1e-6
    fig, ax = plt.subplots(figsize=(8.5, 6.8))
    im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("ROI")
    ax.set_ylabel("ROI")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def average_over_subjects_per_session(advanced_root: Path, out_dir: Path) -> None:
    summary_csv = advanced_root / "metric_run_summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing advanced metric summary CSV: {summary_csv}")

    run_df = pd.read_csv(summary_csv)
    labels = sorted(set(run_df["label"].astype(str).tolist()))
    metrics = sorted(set(run_df["metric"].astype(str).tolist()))

    by_session: dict[int, list[str]] = {1: [], 2: []}
    for label in labels:
        m = LABEL_RE.match(label)
        if not m:
            continue
        ses = int(m.group(2))
        if ses in by_session:
            by_session[ses].append(label)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []

    for ses in [1, 2]:
        ses_labels = sorted(by_session[ses])
        if not ses_labels:
            continue
        ses_out = out_dir / f"session{ses}"
        ses_out.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            mats: list[np.ndarray] = []
            src_labels: list[str] = []
            labels_path = None
            for label in ses_labels:
                metric_dir = advanced_root / label / metric
                m_path = metric_dir / f"{metric}.npy"
                l_path = metric_dir / f"{metric}_connectome.labels.txt"
                if not m_path.exists():
                    continue
                mats.append(np.asarray(np.load(m_path), dtype=np.float64))
                src_labels.append(label)
                if labels_path is None and l_path.exists():
                    labels_path = l_path

            if not mats:
                continue
            shape0 = mats[0].shape
            if any(mat.shape != shape0 for mat in mats):
                raise RuntimeError(
                    f"Shape mismatch for session {ses}, metric {metric}: {[m.shape for m in mats]}"
                )

            stacked = np.stack(mats, axis=0)
            mean_mat = np.nanmean(stacked, axis=0)
            std_mat = np.nanstd(stacked, axis=0)

            metric_out = ses_out / metric
            metric_out.mkdir(parents=True, exist_ok=True)

            np.save(metric_out / f"{metric}_mean.npy", mean_mat.astype(np.float32, copy=False))
            np.save(metric_out / f"{metric}_std.npy", std_mat.astype(np.float32, copy=False))

            pd.DataFrame(mean_mat).to_csv(metric_out / f"{metric}_mean.csv", index=False)
            pd.DataFrame(std_mat).to_csv(metric_out / f"{metric}_std.csv", index=False)

            if labels_path is not None:
                shutil.copy2(labels_path, metric_out / f"{metric}_connectome.labels.txt")

            _write_heatmap(
                mean_mat,
                metric_out / f"{metric}_mean.png",
                title=f"{metric} session{ses} mean across subjects",
                cbar_label=f"{metric} mean",
            )
            _write_heatmap(
                std_mat,
                metric_out / f"{metric}_std.png",
                title=f"{metric} session{ses} std across subjects",
                cbar_label=f"{metric} std",
            )

            meta = {
                "session": ses,
                "metric": metric,
                "n_subjects": int(len(src_labels)),
                "source_labels": src_labels,
                "matrix_shape": [int(shape0[0]), int(shape0[1])],
            }
            (metric_out / f"{metric}_session_average_meta.json").write_text(
                json.dumps(meta, indent=2),
                encoding="utf-8",
            )

            summary_rows.append(
                {
                    "session": ses,
                    "metric": metric,
                    "n_subjects": int(len(src_labels)),
                    "matrix_rows": int(shape0[0]),
                    "matrix_cols": int(shape0[1]),
                    "mean_abs_value": float(np.nanmean(np.abs(mean_mat))),
                    "mean_std_value": float(np.nanmean(std_mat)),
                    "output_dir": str(metric_out),
                }
            )

    if not summary_rows:
        raise RuntimeError("No session-average outputs were produced.")
    pd.DataFrame(summary_rows).to_csv(out_dir / "session_metric_averages_summary.csv", index=False)


def main() -> None:
    args = parse_args()

    input_beta = args.input_beta.expanduser().resolve()
    input_voxel_indices = args.input_voxel_indices.expanduser().resolve()
    voxel_weight_img = args.voxel_weight_img.expanduser().resolve() if args.voxel_weight_img is not None else None
    manifest_path = args.manifest_path.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()

    data_dir = out_root / "data"
    roi_out_dir = out_root / "roi_edge_network"
    avg_out_dir = roi_out_dir / "advanced_metrics_session_average"

    if not input_beta.exists():
        raise FileNotFoundError(f"Input beta matrix not found: {input_beta}")
    if not input_voxel_indices.exists():
        raise FileNotFoundError(f"Voxel-index file not found: {input_voxel_indices}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    beta = np.load(input_beta, mmap_mode="r")
    if beta.ndim != 2:
        raise ValueError(f"Expected 2D beta matrix at {input_beta}, got {beta.shape}")

    n_vox, n_trials = beta.shape
    print(f"Input beta: {input_beta} | shape=({n_vox}, {n_trials})")

    splits = build_subject_session_splits(
        manifest_path=manifest_path,
        n_trials_total=int(n_trials),
        allow_partial_runs=bool(args.allow_partial_runs),
    )
    if not splits:
        raise RuntimeError("No subject-session splits found.")

    split_df = pd.DataFrame(
        [
            {
                "label": split.label,
                "subject": split.subject,
                "session": split.session,
                "runs": ",".join(map(str, split.runs)),
                "n_runs": len(split.runs),
                "n_trials": int(split.columns.size),
            }
            for split in splits
        ]
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(data_dir / "subject_session_split_summary.csv", index=False)
    print(
        "Subject-session splits: "
        f"{len(splits)} labels across {split_df['subject'].nunique()} subjects"
    )

    write_split_files(
        input_beta=input_beta,
        data_dir=data_dir,
        splits=splits,
        chunk_size=int(max(1, args.chunk_size)),
        overwrite=bool(args.overwrite),
    )

    if not (data_dir / "concat_manifest_group.tsv").exists() or args.overwrite:
        shutil.copy2(manifest_path, data_dir / "concat_manifest_group.tsv")
    if not (data_dir / "selected_voxel_indices.npz").exists() or args.overwrite:
        shutil.copy2(input_voxel_indices, data_dir / "selected_voxel_indices.npz")

    if not args.skip_connectivity:
        run_connectivity_pipeline(
            data_dir=data_dir,
            voxel_indices_path=data_dir / "selected_voxel_indices.npz",
            out_dir=roi_out_dir,
            advanced_metrics=args.advanced_metrics,
            voxel_weight_img=voxel_weight_img,
        )
    else:
        print("Skipping connectivity run (--skip-connectivity).")

    if not args.skip_session_average:
        average_over_subjects_per_session(
            advanced_root=roi_out_dir / "advanced_metrics",
            out_dir=avg_out_dir,
        )
        print(f"Saved session-average metric outputs to: {avg_out_dir}")
    else:
        print("Skipping session averages (--skip-session-average).")

    run_manifest = {
        "input_beta": str(input_beta),
        "input_voxel_indices": str(input_voxel_indices),
        "manifest_path": str(manifest_path),
        "out_root": str(out_root),
        "n_subject_session_labels": int(len(splits)),
        "n_unique_subjects": int(split_df["subject"].nunique()),
        "advanced_metrics": str(args.advanced_metrics),
        "voxel_weight_img": str(voxel_weight_img) if voxel_weight_img is not None else None,
        "allow_partial_runs": bool(args.allow_partial_runs),
        "skip_connectivity": bool(args.skip_connectivity),
        "skip_session_average": bool(args.skip_session_average),
    }
    (out_root / "run_manifest_subject_session.json").write_text(
        json.dumps(run_manifest, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
