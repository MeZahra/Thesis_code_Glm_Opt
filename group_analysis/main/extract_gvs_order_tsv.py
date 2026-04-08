#!/usr/bin/env python3
"""Extract true_stim_order for each subject/session/run into a TSV file.

The Windows path provided by the study files is typically:
M:\Data_Masterfile\H20-00572_All-Dressed\PRECISIONSTIM_PD_Data_Results

On this machine that location is mounted at:
/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

MASTER_ROOT_CANDIDATES = [
    Path("/mnt/TeamShare/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results"),
    Path("/Data_Masterfile/H20-00572_All-Dressed/PRECISIONSTIM_PD_Data_Results"),
]
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "gvs_order_by_subject_session_run.tsv"
)
EXPECTED_SESSIONS = ("OFF", "ON")
EXPECTED_RUNS = (1, 2)

SUBJECT_DIR_RE = re.compile(r"^PRECISIONSTIM_PD(?P<subject>\d+)")
RUN_FILE_RE = re.compile(r"^PSPD(?P<subject>\d{3})_(?P<session>OFF|ON)_Run_(?P<run>\d+)(?P<suffix>.*)\.mat$")


def _default_master_root() -> Path:
    for candidate in MASTER_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    return MASTER_ROOT_CANDIDATES[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract true_stim_order from subject/session/run MATLAB files into a TSV."
    )
    parser.add_argument(
        "--master-root",
        type=Path,
        default=_default_master_root(),
        help="Root containing PRECISIONSTIM_PD### subject folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output TSV path.",
    )
    return parser.parse_args()


def iter_subject_dirs(master_root: Path) -> list[tuple[int, Path]]:
    subject_dirs: list[tuple[int, Path]] = []
    for path in sorted(master_root.iterdir()):
        if not path.is_dir():
            continue
        match = SUBJECT_DIR_RE.match(path.name)
        if match is None:
            continue
        subject_dirs.append((int(match.group("subject")), path))
    return subject_dirs


def select_main_run_file(subject_dir: Path, subject_id: str, session: str, run: int) -> tuple[Path | None, str]:
    canonical_name = f"{subject_id}_{session}_Run_{run}.mat"
    canonical_path = subject_dir / canonical_name
    if canonical_path.exists():
        return canonical_path, "exact"

    glob_pattern = f"{subject_id}_{session}_Run_{run}*.mat"
    candidates: list[tuple[str, Path]] = []
    for path in sorted(subject_dir.glob(glob_pattern)):
        match = RUN_FILE_RE.match(path.name)
        if match is None:
            continue
        suffix = match.group("suffix")
        if "BASELINE_TEST" in suffix.upper():
            continue
        candidates.append((suffix, path))

    if not candidates:
        return None, "missing"

    chosen_suffix, chosen_path = min(
        candidates,
        key=lambda item: (len(item[0]), item[0].lower(), item[1].name.lower()),
    )
    if len(candidates) == 1:
        return chosen_path, f"fallback:{chosen_suffix or 'unsuffixed'}"
    return chosen_path, f"fallback_multiple:{chosen_suffix or 'unsuffixed'}"


def load_true_stim_order(path: Path) -> list[str]:
    try:
        with h5py.File(path, "r") as handle:
            if "true_stim_order" not in handle:
                raise KeyError("true_stim_order")
            values = np.asarray(handle["true_stim_order"]).squeeze()
    except OSError:
        mat_data = loadmat(path)
        if "true_stim_order" not in mat_data:
            raise KeyError("true_stim_order")
        values = np.asarray(mat_data["true_stim_order"]).squeeze()

    flat_values = np.atleast_1d(values).reshape(-1)
    order: list[str] = []
    for value in flat_values.tolist():
        numeric = float(value)
        if np.isfinite(numeric) and np.isclose(numeric, round(numeric)):
            order.append(str(int(round(numeric))))
        else:
            order.append(f"{numeric:g}")
    return order


def collect_rows(master_root: Path) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    max_order_len = 0

    for subject_number, subject_dir in iter_subject_dirs(master_root):
        subject_id = f"PSPD{subject_number:03d}"
        for session_index, session in enumerate(EXPECTED_SESSIONS, start=1):
            for run in EXPECTED_RUNS:
                selected_path, file_match = select_main_run_file(subject_dir, subject_id, session, run)
                row: dict[str, str] = {
                    "subject_number": str(subject_number),
                    "subject_id": subject_id,
                    "session": session,
                    "session_index": str(session_index),
                    "run": str(run),
                    "subject_dir": str(subject_dir),
                    "mat_file": "",
                    "mat_path": "",
                    "file_match": file_match,
                    "true_stim_order": "",
                    "n_stim": "",
                    "status": "",
                    "error": "",
                }

                if selected_path is None:
                    row["status"] = "missing_file"
                    row["_order_values"] = []
                    rows.append(row)
                    continue

                row["mat_file"] = selected_path.name
                row["mat_path"] = str(selected_path)

                try:
                    order_values = load_true_stim_order(selected_path)
                except KeyError as exc:
                    row["status"] = "missing_variable"
                    row["error"] = str(exc)
                    row["_order_values"] = []
                except Exception as exc:  # pragma: no cover - defensive logging for dataset issues
                    row["status"] = "read_error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    row["_order_values"] = []
                else:
                    row["status"] = "ok"
                    row["true_stim_order"] = ",".join(order_values)
                    row["n_stim"] = str(len(order_values))
                    row["_order_values"] = order_values
                    max_order_len = max(max_order_len, len(order_values))

                rows.append(row)

    return rows, max_order_len


def write_tsv(rows: list[dict[str, str]], max_order_len: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject_number",
        "subject_id",
        "session",
        "session_index",
        "run",
        "subject_dir",
        "mat_file",
        "mat_path",
        "file_match",
        "true_stim_order",
        "n_stim",
    ]
    fieldnames.extend(f"stim_order_{index}" for index in range(1, max_order_len + 1))
    fieldnames.extend(["status", "error"])

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            output_row = {key: row.get(key, "") for key in fieldnames}
            order_values = row.get("_order_values", [])
            for index, value in enumerate(order_values, start=1):
                output_row[f"stim_order_{index}"] = value
            writer.writerow(output_row)


def main() -> None:
    args = parse_args()
    master_root = args.master_root.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    if not master_root.exists():
        raise FileNotFoundError(
            f"Master root not found: {master_root}. "
            "If needed, point --master-root to the Linux mount of the Windows M: drive."
        )

    rows, max_order_len = collect_rows(master_root)
    write_tsv(rows, max_order_len, output_path)

    status_counts = Counter(row["status"] for row in rows)
    print(f"Master root: {master_root}")
    print(f"Subject folders: {len(iter_subject_dirs(master_root))}")
    print(f"Rows written: {len(rows)}")
    print(f"Output: {output_path}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}: {count}")


if __name__ == "__main__":
    main()
