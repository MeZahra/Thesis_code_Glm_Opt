#!/usr/bin/env python3
"""Split selected beta trials and projection values by GVS condition."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

DEFAULT_GVS_ORDER_PATH = (
    REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "gvs_order_by_subject_session_run.tsv"
)
DEFAULT_BETA_DIR = REPO_ROOT / "results" / "connectivity" / "data"
DEFAULT_PROJECTION_PATH = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "GVS_effects"
    / "data"
    / "projection_voxel_foldavg_sub9_ses1_task0.8_bold0.8_beta0.5_smooth0.2_gamma1_bold_thr90.npy"
)
DEFAULT_MANIFEST_PATH = REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "concat_manifest_group.tsv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "by_gvs"

TRIALS_PER_GVS_BLOCK = 10
BETA_FILE_RE = re.compile(r"^selected_beta_trials_(sub-pd\d+)_ses-(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split subject/session beta trials and group projection data by GVS condition."
    )
    parser.add_argument("--gvs-order-path", type=Path, default=DEFAULT_GVS_ORDER_PATH, help="Path to GVS order TSV.")
    parser.add_argument(
        "--beta-dir",
        type=Path,
        default=DEFAULT_BETA_DIR,
        help="Directory containing selected_beta_trials_sub-pd*_ses-*.npy files.",
    )
    parser.add_argument("--projection-path", type=Path, default=DEFAULT_PROJECTION_PATH, help="Path to projection .npy.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to concat manifest TSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output root for by-subject/session GVS splits.",
    )
    parser.add_argument(
        "--trials-per-block",
        type=int,
        default=TRIALS_PER_GVS_BLOCK,
        help="Number of consecutive trials in each GVS block.",
    )
    return parser.parse_args()


def _load_gvs_orders(path: Path) -> dict[tuple[str, int, int], list[str]]:
    orders: dict[tuple[str, int, int], list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("status", "")).strip().lower() != "ok":
                continue
            subject_number = int(row["subject_number"])
            session_index = int(row["session_index"])
            run = int(row["run"])
            true_stim_order = str(row.get("true_stim_order", "")).strip()
            order_values = [item.strip() for item in true_stim_order.split(",") if item.strip()]
            if not order_values:
                raise ValueError(
                    f"Missing true_stim_order for subject {subject_number} session {session_index} run {run} in {path}."
                )
            key = (f"sub-pd{subject_number:03d}", session_index, run)
            if key in orders:
                raise ValueError(f"Duplicate GVS order entry for {key} in {path}.")
            orders[key] = order_values
    if not orders:
        raise ValueError(f"No usable GVS order rows found in {path}.")
    return orders


def _load_trial_keep_mask(row: dict[str, str], source_trials: int) -> np.ndarray:
    trial_keep_path = str(row.get("trial_keep_path", "") or "").strip()
    if trial_keep_path and trial_keep_path.lower() != "nan":
        keep_mask = np.asarray(np.load(trial_keep_path), dtype=bool).ravel()
        if keep_mask.size != source_trials:
            raise ValueError(
                f"trial_keep length mismatch for {trial_keep_path}: {keep_mask.size} vs {source_trials}"
            )
        return keep_mask
    return np.ones(source_trials, dtype=bool)


def _build_kept_condition_labels(
    order_values: list[str],
    keep_mask: np.ndarray,
    trials_per_block: int,
    context: str,
) -> np.ndarray:
    expected_source_trials = len(order_values) * int(trials_per_block)
    if expected_source_trials != int(keep_mask.size):
        raise ValueError(
            f"{context}: GVS order defines {expected_source_trials} source trials "
            f"({len(order_values)} blocks x {trials_per_block}) but source trial count is {keep_mask.size}."
        )
    source_labels = np.repeat(np.asarray(order_values, dtype=object), int(trials_per_block))
    return source_labels[np.asarray(keep_mask, dtype=bool)]


def _condition_sort_key(value: str) -> tuple[int, float | str, str]:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return (1, text, text)
    return (0, numeric, text)


def _condition_filename_tag(value: str) -> str:
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        cleaned = re.sub(r"[^0-9A-Za-z]+", "-", text).strip("-").lower()
        if not cleaned:
            raise ValueError(f"Cannot derive filename tag for GVS condition {value!r}.")
        return f"gvs-{cleaned}"
    if np.isfinite(numeric) and np.isclose(numeric, round(numeric)):
        return f"gvs-{int(round(numeric)):02d}"
    return f"gvs-{text.replace('.', 'p')}"


def _load_manifest_rows(
    manifest_path: Path,
    gvs_orders: dict[tuple[str, int, int], list[str]],
    trials_per_block: int,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    condition_values: set[str] = set()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            sub_tag = str(raw_row["sub_tag"]).strip()
            ses = int(raw_row["ses"])
            run = int(raw_row["run"])
            offset_start = int(raw_row["offset_start"])
            offset_end = int(raw_row["offset_end"])
            n_trials_kept = int(raw_row["n_trials"])
            source_trials = int(raw_row.get("n_trials_source") or n_trials_kept)
            keep_mask = _load_trial_keep_mask(raw_row, source_trials)
            kept_count = int(np.count_nonzero(keep_mask))
            if kept_count != n_trials_kept:
                raise ValueError(
                    f"Manifest kept trial mismatch for {sub_tag} ses-{ses} run-{run}: "
                    f"manifest={n_trials_kept}, trial_keep={kept_count}"
                )
            if (offset_end - offset_start) != n_trials_kept:
                raise ValueError(
                    f"Manifest offset range mismatch for {sub_tag} ses-{ses} run-{run}: "
                    f"offsets={offset_end - offset_start}, n_trials={n_trials_kept}"
                )

            order_key = (sub_tag, ses, run)
            order_values = gvs_orders.get(order_key)
            if order_values is None:
                raise KeyError(f"Missing GVS order for {sub_tag} ses-{ses} run-{run}.")

            context = f"{sub_tag} ses-{ses} run-{run}"
            kept_condition_labels = _build_kept_condition_labels(order_values, keep_mask, trials_per_block, context)
            condition_values.update(str(value) for value in kept_condition_labels.tolist())

            rows.append(
                {
                    "sub_tag": sub_tag,
                    "ses": ses,
                    "run": run,
                    "offset_start": offset_start,
                    "offset_end": offset_end,
                    "n_trials_kept": n_trials_kept,
                    "n_trials_source": source_trials,
                    "keep_mask": keep_mask,
                    "kept_condition_labels": kept_condition_labels,
                }
            )

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    rows.sort(key=lambda row: int(row["offset_start"]))
    ordered_conditions = sorted(condition_values, key=_condition_sort_key)
    return rows, ordered_conditions


def _gather_session_columns(
    session_rows: list[dict[str, object]],
    n_session_trials: int,
    condition_values: list[str],
) -> dict[str, np.ndarray]:
    columns_by_condition: dict[str, list[int]] = {condition: [] for condition in condition_values}
    session_cursor = 0
    coverage = np.zeros(n_session_trials, dtype=np.int16)

    for row in sorted(session_rows, key=lambda item: int(item["run"])):
        run_trials = int(row["n_trials_kept"])
        kept_condition_labels = np.asarray(row["kept_condition_labels"], dtype=object)
        if kept_condition_labels.size != run_trials:
            raise ValueError(
                f"{row['sub_tag']} ses-{row['ses']} run-{row['run']}: condition-label count "
                f"{kept_condition_labels.size} does not match kept trials {run_trials}."
            )

        run_columns = np.arange(session_cursor, session_cursor + run_trials, dtype=np.int64)
        if run_columns.size and int(run_columns[-1]) >= n_session_trials:
            raise ValueError(
                f"{row['sub_tag']} ses-{row['ses']}: run columns exceed beta-trial width {n_session_trials}."
            )
        for condition in condition_values:
            mask = kept_condition_labels == condition
            if np.any(mask):
                selected_columns = run_columns[mask]
                columns_by_condition[condition].extend(selected_columns.tolist())
                coverage[selected_columns] += 1
        session_cursor += run_trials

    if session_cursor != n_session_trials:
        raise ValueError(
            f"Session trial mismatch: manifest assigns {session_cursor} beta trials, but file has {n_session_trials}."
        )
    if np.any(coverage != 1):
        bad_count = int(np.count_nonzero(coverage != 1))
        raise ValueError(f"Session beta trials are not covered exactly once across GVS conditions ({bad_count} bad columns).")

    return {condition: np.asarray(columns, dtype=np.int64) for condition, columns in columns_by_condition.items()}


def _gather_projection_rows(
    manifest_rows: list[dict[str, object]],
    n_projection_trials: int,
    condition_values: list[str],
) -> dict[tuple[str, int], dict[str, np.ndarray]]:
    rows_by_session_condition: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: {condition: [] for condition in condition_values}
    )
    coverage = np.zeros(n_projection_trials, dtype=np.int16)

    for row in manifest_rows:
        offset_start = int(row["offset_start"])
        offset_end = int(row["offset_end"])
        kept_condition_labels = np.asarray(row["kept_condition_labels"], dtype=object)
        run_rows = np.arange(offset_start, offset_end, dtype=np.int64)
        if kept_condition_labels.size != run_rows.size:
            raise ValueError(
                f"{row['sub_tag']} ses-{row['ses']} run-{row['run']}: manifest offsets "
                f"yield {run_rows.size} rows but condition labels yield {kept_condition_labels.size} rows."
            )
        coverage[run_rows] += 1

        session_key = (str(row["sub_tag"]), int(row["ses"]))
        session_map = rows_by_session_condition[session_key]
        for condition in condition_values:
            mask = kept_condition_labels == condition
            if np.any(mask):
                session_map[condition].extend(run_rows[mask].tolist())

    if np.any(coverage != 1):
        bad_count = int(np.count_nonzero(coverage != 1))
        raise ValueError(f"Projection rows are not covered exactly once across manifest runs ({bad_count} bad rows).")

    return {
        session_key: {condition: np.asarray(indices, dtype=np.int64) for condition, indices in session_map.items()}
        for session_key, session_map in rows_by_session_condition.items()
    }


def _save_beta_splits(
    beta_dir: Path,
    output_dir: Path,
    manifest_rows: list[dict[str, object]],
    condition_values: list[str],
) -> None:
    session_rows_map: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        session_rows_map[(str(row["sub_tag"]), int(row["ses"]))].append(row)

    beta_paths = sorted(beta_dir.glob("selected_beta_trials_sub-pd*_ses-*.npy"))
    if not beta_paths:
        raise FileNotFoundError(f"No beta-trial files found under {beta_dir}.")

    for beta_path in beta_paths:
        match = BETA_FILE_RE.match(beta_path.name)
        if match is None:
            continue
        sub_tag = match.group(1)
        ses = int(match.group(2))
        session_key = (sub_tag, ses)
        session_rows = session_rows_map.get(session_key)
        if not session_rows:
            raise KeyError(f"Missing manifest rows for {sub_tag} ses-{ses}.")

        beta_trials = np.load(beta_path)
        if beta_trials.ndim != 2:
            raise ValueError(f"Expected beta-trial matrix with shape (n_voxels, n_trials), got {beta_trials.shape}.")

        columns_by_condition = _gather_session_columns(session_rows, int(beta_trials.shape[1]), condition_values)
        session_output_dir = output_dir / sub_tag / f"ses-{ses}"
        session_output_dir.mkdir(parents=True, exist_ok=True)

        for condition in condition_values:
            condition_tag = _condition_filename_tag(condition)
            out_path = session_output_dir / f"selected_beta_trials_{condition_tag}.npy"
            subset = np.asarray(beta_trials[:, columns_by_condition[condition]])
            np.save(out_path, subset)
            print(
                f"saved beta subject={sub_tag} session={ses} gvs_type={condition} "
                f"shape={subset.shape} path={out_path}"
            )


def _save_projection_splits(
    projection_path: Path,
    output_dir: Path,
    projection_rows_by_session: dict[tuple[str, int], dict[str, np.ndarray]],
    manifest_rows: list[dict[str, object]],
    condition_values: list[str],
) -> None:
    projection = np.load(projection_path, mmap_mode="r")
    if projection.ndim < 1:
        raise ValueError(f"Projection array must have at least 1 dimension, got {projection.shape}.")

    expected_projection_trials = max(int(row["offset_end"]) for row in manifest_rows)
    if int(projection.shape[0]) != expected_projection_trials:
        raise ValueError(
            f"Projection trial count mismatch: projection has {projection.shape[0]} rows, "
            f"manifest expects {expected_projection_trials}."
        )

    for (sub_tag, ses), rows_by_condition in sorted(projection_rows_by_session.items()):
        session_output_dir = output_dir / sub_tag / f"ses-{ses}"
        session_output_dir.mkdir(parents=True, exist_ok=True)

        for condition in condition_values:
            condition_tag = _condition_filename_tag(condition)
            out_path = session_output_dir / f"projection_voxel_foldavg_{condition_tag}.npy"
            subset = np.asarray(projection[rows_by_condition[condition]])
            np.save(out_path, subset)
            print(
                f"saved projection subject={sub_tag} session={ses} gvs_type={condition} "
                f"shape={subset.shape} path={out_path}"
            )


def main() -> None:
    args = parse_args()

    gvs_order_path = args.gvs_order_path.expanduser().resolve()
    beta_dir = args.beta_dir.expanduser().resolve()
    projection_path = args.projection_path.expanduser().resolve()
    manifest_path = args.manifest_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.trials_per_block <= 0:
        raise ValueError(f"--trials-per-block must be positive, got {args.trials_per_block}.")

    gvs_orders = _load_gvs_orders(gvs_order_path)
    manifest_rows, condition_values = _load_manifest_rows(manifest_path, gvs_orders, args.trials_per_block)
    projection_rows_by_session = _gather_projection_rows(
        manifest_rows,
        max(int(row["offset_end"]) for row in manifest_rows),
        condition_values,
    )

    _save_beta_splits(beta_dir, output_dir, manifest_rows, condition_values)
    _save_projection_splits(projection_path, output_dir, projection_rows_by_session, manifest_rows, condition_values)


if __name__ == "__main__":
    main()
