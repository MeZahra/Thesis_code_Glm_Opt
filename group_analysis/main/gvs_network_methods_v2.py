#!/usr/bin/env python3
"""GVS network methods v2.

This standalone script keeps the existing ROI-aggregation, correlation, and
Fisher-z functional-connectivity estimator used elsewhere in the repository,
but replaces the older primary inference logic with four cleaner tests.

Why the old primary logic is not reused here:
    1. Raw graph distances are not tested against 0 as a null. Distances such as
       Frobenius or spectral distance are non-negative by construction, so 0 is
       not a scientifically valid chance baseline for condition effects.
    2. Ratio-vs-1 is not used as the primary inferential target. Ratio scores can
       be unstable when denominator variability is small and are harder to
       interpret than permutation- or sign-flip-based contrasts.

Null hypotheses used here:
    Method 1:
        Shuffling block labels within run does not reduce or increase the mean
        sham-referenced session-level network distance for an active GVS
        condition beyond chance.
    Method 2:
        Shuffling block labels within run does not change the mean excess
        sham-referenced run-level distance relative to within-condition
        baseline variability.
    Method 3:
        Shuffling OFF-session block labels within run does not make OFF+GVS look
        systematically closer to ON-sham than OFF-sham does.
    Method 4:
        The crossvalidated sham-vs-GVS edge contrast has subject-level mean 0,
        so random sign flips across subjects should reproduce the observed group
        mean under the null.

The script is intentionally structured around reusable block reconstruction and
restricted within-run relabeling helpers so later methods can reuse the same
session/run network plumbing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection

_HERE = Path(__file__).resolve().parent
_GROUP_ANALYSIS_DIR = _HERE.parent
_REPO_ROOT = _GROUP_ANALYSIS_DIR.parent
if str(_GROUP_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from group_analysis.connectivity_new.roi_edge_connectivity_requested import (  # noqa: E402
    _safe_corrcoef_rows,
)
from group_analysis.main.analyze_pairwise_metric_separation import (  # noqa: E402
    _laplacian_spectral_distance,
    _signed_normalized_laplacian_spectrum,
)
from group_analysis.main.gvs_effects_analysis import (  # noqa: E402
    ACTIVE_CONDITION_CODES,
    ACTIVE_CONDITION_FACTORS,
    DEFAULT_GVS_ORDER_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_ROI_IMG,
    DEFAULT_ROI_SUMMARY,
    DEFAULT_SELECTED_VOXELS_PATH,
    DEFAULT_SESSION_BETA_DIR,
    MEDICATION_ORDER,
    MIN_ROI_VOXELS,
    SHAM_CONDITION_CODE,
    build_roi_membership,
    condition_code_from_numeric,
    condition_factor_from_code,
    condition_name_from_code,
    correlation_distance,
    ensure_dir,
    fisher_z_matrix,
    medication_from_session,
    safe_slug,
)
from group_analysis.main.split_data_by_gvs_condition import (  # noqa: E402
    _load_gvs_orders,
    _load_manifest_rows,
)


TRIALS_PER_BLOCK = 10
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "four_methods_v2"
METHOD1_DIRNAME = "method1_sham_referenced_raw_distance"
METHOD2_DIRNAME = "method2_excess_distance_baseline_adjusted"
METHOD3_DIRNAME = "method3_off_to_on_normalization"
METHOD4_DIRNAME = "method4_crossvalidated_sham_contrast"
METHOD_NAMES = ("method1", "method2", "method3", "method4")
METHOD2_BASELINE_MODES = ("mean_within", "sham_only")
METHOD2_BASELINE_ALIASES = {
    "mean_within": "mean_within",
    "full": "mean_within",
    "sham_only": "sham_only",
}
PRIMARY_DISTANCE_METRIC = "laplacian_spectral_distance"
SECONDARY_DISTANCE_METRICS = ("frobenius_norm", "correlation_distance")
SUPPORTED_DISTANCE_METRICS = (
    PRIMARY_DISTANCE_METRIC,
    *SECONDARY_DISTANCE_METRICS,
)
DISTANCE_METRIC_LABELS = {
    "laplacian_spectral_distance": "Laplacian spectral distance",
    "frobenius_norm": "Frobenius norm",
    "correlation_distance": "Correlation distance",
}
METHOD2_BASELINE_LABELS = {
    "mean_within": "Mean within-condition baseline",
    "sham_only": "Sham-only baseline",
}
METHOD_LABELS = {
    "method1": "Method 1",
    "method2": "Method 2",
    "method3": "Method 3",
    "method4": "Method 4",
}
EXACT_SIGN_FLIP_MAX_SUBJECTS = 15


@dataclass(frozen=True)
class SessionBlock:
    subject: str
    session: int
    medication: str
    run: int
    block_index: int
    true_condition_code: str
    trial_columns: np.ndarray
    roi_ts: np.ndarray

    @property
    def block_uid(self) -> str:
        return f"{self.subject}_ses-{self.session}_run-{self.run}_block-{self.block_index}"

    @property
    def n_trials_kept(self) -> int:
        return int(self.trial_columns.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run GVS network methods v2 with restricted within-run block-label "
            "randomization for Methods 1-3 and subject-level sign flips for Method 4."
        )
    )
    parser.add_argument("--gvs-order-path", type=Path, default=DEFAULT_GVS_ORDER_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--session-beta-dir", type=Path, default=DEFAULT_SESSION_BETA_DIR)
    parser.add_argument("--selected-voxels-path", type=Path, default=DEFAULT_SELECTED_VOXELS_PATH)
    parser.add_argument("--roi-img", type=Path, default=DEFAULT_ROI_IMG)
    parser.add_argument("--roi-summary", type=Path, default=DEFAULT_ROI_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trials-per-block", type=int, default=TRIALS_PER_BLOCK)
    parser.add_argument("--min-roi-voxels", type=int, default=MIN_ROI_VOXELS)
    parser.add_argument(
        "--distance-metrics",
        default="laplacian_spectral_distance,frobenius_norm,correlation_distance",
        help=(
            "Comma-separated distance metrics. "
            "Supported: laplacian_spectral_distance,frobenius_norm,correlation_distance"
        ),
    )
    parser.add_argument(
        "--methods",
        default="method1,method2,method3,method4",
        help="Comma-separated method list. Supported: method1,method2,method3,method4",
    )
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seed", dest="seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--baseline-mode",
        default="mean_within",
        help=(
            "Method 2 baseline definition. "
            "'mean_within' uses mean(sham run1-vs-run2, target run1-vs-run2); "
            "'sham_only' uses only sham run1-vs-run2."
        ),
    )
    return parser.parse_args()


def parse_distance_metrics(raw_value: str) -> list[str]:
    values = [item.strip() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one distance metric must be requested.")
    deduped: list[str] = []
    for value in values:
        if value not in SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                f"Unsupported distance metric {value!r}. Supported values: {', '.join(SUPPORTED_DISTANCE_METRICS)}"
            )
        if value not in deduped:
            deduped.append(value)
    return deduped


def parse_methods(raw_value: str) -> list[str]:
    values = [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]
    if not values:
        raise ValueError("At least one method must be requested.")
    deduped: list[str] = []
    for value in values:
        if value not in METHOD_NAMES:
            raise ValueError(f"Unsupported method {value!r}. Supported values: {', '.join(METHOD_NAMES)}")
        if value not in deduped:
            deduped.append(value)
    return deduped


def normalize_baseline_mode(raw_value: str) -> str:
    key = str(raw_value).strip().lower()
    normalized = METHOD2_BASELINE_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Unsupported baseline mode {raw_value!r}. Supported values: mean_within, sham_only"
        )
    return normalized


def _frame_from_rows(rows: list[dict[str, Any]], sort_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    available_columns = [column for column in sort_columns if column in df.columns]
    if available_columns:
        return df.sort_values(available_columns).reset_index(drop=True)
    return df.reset_index(drop=True)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if not df.empty:
        return df
    return pd.DataFrame(columns=columns)


def _compute_roi_timeseries(beta: np.ndarray, roi_members: list[np.ndarray]) -> np.ndarray:
    """ROI aggregation step used by the existing FC pipeline.

    This mirrors the ROI averaging done inside the older connectivity helper, but
    breaks it out so block-level reconstruction computes ROI time series only once.
    """

    beta_array = np.asarray(beta, dtype=np.float64)
    roi_ts = np.full((len(roi_members), beta_array.shape[1]), np.nan, dtype=np.float64)
    if beta_array.shape[1] == 0:
        return roi_ts
    for roi_index, members in enumerate(roi_members):
        roi_ts[roi_index] = np.nanmean(beta_array[members, :], axis=0)
    return roi_ts


def _compute_fc_payload_from_roi_ts(roi_ts: np.ndarray) -> dict[str, np.ndarray]:
    roi_ts_array = np.asarray(roi_ts, dtype=np.float64)
    if roi_ts_array.ndim != 2:
        raise ValueError(f"Expected ROI time series shape (n_rois, n_trials), got {roi_ts_array.shape}.")
    fc_r = _safe_corrcoef_rows(roi_ts_array)
    fc_r = np.clip(fc_r, -1.0, 1.0, out=fc_r)
    np.fill_diagonal(fc_r, 1.0)
    fc_z = fisher_z_matrix(fc_r)
    iu = np.triu_indices(roi_ts_array.shape[0], k=1)
    vec_z = fc_z[iu]
    spectrum = _signed_normalized_laplacian_spectrum(fc_z)
    return {
        "roi_ts": roi_ts_array,
        "r": fc_r,
        "z": fc_z,
        "vec_z": vec_z,
        "spectrum": spectrum,
        "n_trials_fc": np.asarray(roi_ts_array).shape[1],
    }


def _validate_run_block_structure(blocks: list[SessionBlock]) -> None:
    by_run: dict[int, list[SessionBlock]] = defaultdict(list)
    for block in blocks:
        by_run[int(block.run)].append(block)

    expected_condition_set = {SHAM_CONDITION_CODE, *ACTIVE_CONDITION_CODES}
    for run, run_blocks in sorted(by_run.items()):
        if len(run_blocks) != 9:
            raise ValueError(
                f"{run_blocks[0].subject} ses-{run_blocks[0].session} run-{run}: expected 9 blocks, got {len(run_blocks)}."
            )
        run_condition_set = {block.true_condition_code for block in run_blocks}
        if run_condition_set != expected_condition_set:
            raise ValueError(
                f"{run_blocks[0].subject} ses-{run_blocks[0].session} run-{run}: "
                f"expected condition set {sorted(expected_condition_set)}, got {sorted(run_condition_set)}."
            )


def reconstruct_session_blocks(
    manifest_rows: list[dict[str, object]],
    gvs_orders: dict[tuple[str, int, int], list[str]],
    session_beta_dir: Path,
    roi_members: list[np.ndarray],
    trials_per_block: int,
) -> tuple[dict[tuple[str, int], tuple[SessionBlock, ...]], pd.DataFrame, pd.DataFrame]:
    """Reconstruct fixed run/block containers from manifest rows.

    Sessions that cannot be reconstructed are skipped rather than crashing the
    entire analysis. A skip table is returned so downstream outputs can report
    exactly which subject-session was dropped and why.
    """

    session_rows_map: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        session_rows_map[(str(row["sub_tag"]), int(row["ses"]))].append(row)

    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]] = {}
    inventory_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []

    for (subject, session), session_rows in sorted(session_rows_map.items()):
        beta_path = session_beta_dir / f"selected_beta_trials_{subject}_ses-{session}.npy"
        try:
            if not beta_path.exists():
                raise FileNotFoundError(f"Missing session beta file: {beta_path}")
            beta_trials = np.asarray(np.load(beta_path), dtype=np.float64)
            if beta_trials.ndim != 2:
                raise ValueError(f"Expected 2D beta-trial matrix in {beta_path}, got {beta_trials.shape}.")

            medication = medication_from_session(session)
            session_cursor = 0
            session_blocks: list[SessionBlock] = []
            for row in sorted(session_rows, key=lambda item: int(item["run"])):
                run = int(row["run"])
                run_trials = int(row["n_trials_kept"])
                n_source_trials = int(row["n_trials_source"])
                keep_mask = np.asarray(row["keep_mask"], dtype=bool)
                if keep_mask.size != n_source_trials:
                    raise ValueError(
                        f"{subject} ses-{session} run-{run}: keep-mask length {keep_mask.size} "
                        f"does not match source-trial count {n_source_trials}."
                    )

                order_values = gvs_orders.get((subject, session, run))
                if order_values is None:
                    raise KeyError(f"Missing GVS order for {subject} ses-{session} run-{run}.")
                if len(order_values) != 9:
                    raise ValueError(
                        f"{subject} ses-{session} run-{run}: expected 9 block labels, got {len(order_values)}."
                    )

                source_block_ids = np.repeat(np.arange(1, len(order_values) + 1, dtype=np.int64), int(trials_per_block))
                if source_block_ids.size != n_source_trials:
                    raise ValueError(
                        f"{subject} ses-{session} run-{run}: block definition implies {source_block_ids.size} source trials, "
                        f"but manifest says {n_source_trials}."
                    )

                run_columns = np.arange(session_cursor, session_cursor + run_trials, dtype=np.int64)
                kept_block_ids = source_block_ids[keep_mask]
                if kept_block_ids.size != run_columns.size:
                    raise ValueError(
                        f"{subject} ses-{session} run-{run}: kept block labels yield {kept_block_ids.size} trials "
                        f"but session columns yield {run_columns.size}."
                    )

                for block_index, raw_condition in enumerate(order_values, start=1):
                    condition_code = condition_code_from_numeric(raw_condition)
                    block_columns = run_columns[kept_block_ids == block_index]
                    block_beta = np.asarray(beta_trials[:, block_columns], dtype=np.float64)
                    block_roi_ts = _compute_roi_timeseries(block_beta, roi_members=roi_members)
                    block = SessionBlock(
                        subject=subject,
                        session=int(session),
                        medication=medication,
                        run=int(run),
                        block_index=int(block_index),
                        true_condition_code=condition_code,
                        trial_columns=block_columns,
                        roi_ts=block_roi_ts,
                    )
                    session_blocks.append(block)
                    inventory_rows.append(
                        {
                            "subject": subject,
                            "session": int(session),
                            "medication": medication,
                            "run": int(run),
                            "block_index": int(block_index),
                            "block_uid": block.block_uid,
                            "true_condition_code": condition_code,
                            "true_condition_name": condition_name_from_code(condition_code),
                            "true_condition_factor": condition_factor_from_code(condition_code),
                            "n_trials_kept": block.n_trials_kept,
                            "first_trial_column": int(block_columns[0]) if block_columns.size else float("nan"),
                            "last_trial_column": int(block_columns[-1]) if block_columns.size else float("nan"),
                        }
                    )

                session_cursor += run_trials

            if session_cursor != int(beta_trials.shape[1]):
                raise ValueError(
                    f"{subject} ses-{session}: manifest assigns {session_cursor} kept trials but beta file has {beta_trials.shape[1]}."
                )
            _validate_run_block_structure(session_blocks)
            blocks_by_session[(subject, int(session))] = tuple(
                sorted(session_blocks, key=lambda block: (block.run, block.block_index))
            )
        except Exception as exc:
            skip_rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "medication": medication_from_session(session),
                    "skip_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

    inventory_df = _frame_from_rows(inventory_rows, ["subject", "session", "run", "block_index"])
    skip_df = _ensure_columns(
        _frame_from_rows(skip_rows, ["subject", "session"]),
        ["subject", "session", "medication", "skip_reason"],
    )
    return blocks_by_session, inventory_df, skip_df


def permute_block_labels_within_run(
    blocks: tuple[SessionBlock, ...],
    rng: np.random.Generator,
) -> dict[tuple[int, int], str]:
    relabeled: dict[tuple[int, int], str] = {}
    by_run: dict[int, list[SessionBlock]] = defaultdict(list)
    for block in blocks:
        by_run[int(block.run)].append(block)

    for run, run_blocks in sorted(by_run.items()):
        ordered_blocks = sorted(run_blocks, key=lambda block: block.block_index)
        original_labels = np.asarray([block.true_condition_code for block in ordered_blocks], dtype=object)
        shuffled_labels = original_labels[rng.permutation(original_labels.size)]
        for block, assigned_label in zip(ordered_blocks, shuffled_labels.tolist()):
            relabeled[(int(run), int(block.block_index))] = str(assigned_label)
    return relabeled


def _assigned_condition_code(
    block: SessionBlock,
    relabeled_condition_codes: dict[tuple[int, int], str] | None,
) -> str:
    if relabeled_condition_codes is None:
        return block.true_condition_code
    return str(relabeled_condition_codes.get((block.run, block.block_index), block.true_condition_code))


def _concatenate_block_roi_ts(blocks: list[SessionBlock], n_rois: int) -> np.ndarray:
    if not blocks:
        return np.full((n_rois, 0), np.nan, dtype=np.float64)
    arrays = [np.asarray(block.roi_ts, dtype=np.float64) for block in sorted(blocks, key=lambda item: (item.run, item.block_index))]
    return np.concatenate(arrays, axis=1)


def rebuild_session_condition_networks(
    blocks: tuple[SessionBlock, ...],
    relabeled_condition_codes: dict[tuple[int, int], str] | None = None,
) -> tuple[dict[str, list[SessionBlock]], dict[str, dict[str, np.ndarray]]]:
    if not blocks:
        return {}, {}
    n_rois = int(blocks[0].roi_ts.shape[0])
    grouped_blocks: dict[str, list[SessionBlock]] = defaultdict(list)
    for block in blocks:
        grouped_blocks[_assigned_condition_code(block, relabeled_condition_codes)].append(block)

    payloads = {
        condition_code: _compute_fc_payload_from_roi_ts(_concatenate_block_roi_ts(condition_blocks, n_rois=n_rois))
        for condition_code, condition_blocks in grouped_blocks.items()
    }
    return dict(grouped_blocks), payloads


def rebuild_run_condition_networks(
    blocks: tuple[SessionBlock, ...],
    relabeled_condition_codes: dict[tuple[int, int], str] | None = None,
) -> tuple[dict[tuple[int, str], list[SessionBlock]], dict[tuple[int, str], dict[str, np.ndarray]]]:
    if not blocks:
        return {}, {}
    n_rois = int(blocks[0].roi_ts.shape[0])
    grouped_blocks: dict[tuple[int, str], list[SessionBlock]] = defaultdict(list)
    for block in blocks:
        key = (int(block.run), _assigned_condition_code(block, relabeled_condition_codes))
        grouped_blocks[key].append(block)

    payloads = {
        key: _compute_fc_payload_from_roi_ts(_concatenate_block_roi_ts(condition_blocks, n_rois=n_rois))
        for key, condition_blocks in grouped_blocks.items()
    }
    return dict(grouped_blocks), payloads


def compute_distance_metric(metric_name: str, a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    if metric_name == "correlation_distance":
        return correlation_distance(a["vec_z"], b["vec_z"])
    if metric_name == "frobenius_norm":
        return float(np.linalg.norm(np.nan_to_num(a["z"] - b["z"]), ord="fro"))
    if metric_name == "laplacian_spectral_distance":
        return _laplacian_spectral_distance(a["spectrum"], b["spectrum"])
    raise KeyError(f"Unsupported distance metric: {metric_name}")


def compute_method2_excess_distance(
    run_payloads: dict[tuple[int, str], dict[str, np.ndarray]],
    condition_code: str,
    metric_name: str,
    baseline_mode: str,
) -> dict[str, float] | None:
    """Method 2 score.

    Null hypothesis:
        After within-run block-label randomization, the mean excess-distance
        score is not systematically larger than expected by chance.
    """

    required_keys = [
        (1, SHAM_CONDITION_CODE),
        (2, SHAM_CONDITION_CODE),
        (1, condition_code),
        (2, condition_code),
    ]
    if any(key not in run_payloads for key in required_keys):
        return None

    sham_run1 = run_payloads[(1, SHAM_CONDITION_CODE)]
    sham_run2 = run_payloads[(2, SHAM_CONDITION_CODE)]
    target_run1 = run_payloads[(1, condition_code)]
    target_run2 = run_payloads[(2, condition_code)]

    distance_target_run1_vs_sham_run1 = compute_distance_metric(metric_name, target_run1, sham_run1)
    distance_target_run2_vs_sham_run2 = compute_distance_metric(metric_name, target_run2, sham_run2)
    distance_sham_run1_vs_run2 = compute_distance_metric(metric_name, sham_run1, sham_run2)
    distance_target_run1_vs_run2 = compute_distance_metric(metric_name, target_run1, target_run2)

    target_vs_sham_values = np.asarray(
        [
            distance_target_run1_vs_sham_run1,
            distance_target_run2_vs_sham_run2,
        ],
        dtype=np.float64,
    )
    target_vs_sham_values = target_vs_sham_values[np.isfinite(target_vs_sham_values)]
    if target_vs_sham_values.size == 0:
        return None
    target_vs_sham_mean = float(np.mean(target_vs_sham_values))

    if baseline_mode == "mean_within":
        baseline_values = np.asarray(
            [
                distance_sham_run1_vs_run2,
                distance_target_run1_vs_run2,
            ],
            dtype=np.float64,
        )
    elif baseline_mode == "sham_only":
        baseline_values = np.asarray([distance_sham_run1_vs_run2], dtype=np.float64)
    else:
        raise ValueError(
            f"Unsupported baseline_mode {baseline_mode!r}. Expected one of {', '.join(METHOD2_BASELINE_MODES)}."
        )
    baseline_values = baseline_values[np.isfinite(baseline_values)]
    if baseline_values.size == 0:
        return None
    baseline_mean = float(np.mean(baseline_values))

    return {
        "distance_target_run1_vs_sham_run1": float(distance_target_run1_vs_sham_run1),
        "distance_target_run2_vs_sham_run2": float(distance_target_run2_vs_sham_run2),
        "distance_sham_run1_vs_run2": float(distance_sham_run1_vs_run2),
        "distance_target_run1_vs_run2": float(distance_target_run1_vs_run2),
        "target_vs_sham_mean_distance": target_vs_sham_mean,
        "baseline_mean_distance": baseline_mean,
        "excess_distance_value": float(target_vs_sham_mean - baseline_mean),
    }


def compute_method3_normalization_score(
    off_condition_payloads: dict[str, dict[str, np.ndarray]],
    on_sham_payload: dict[str, np.ndarray],
    condition_code: str,
    metric_name: str,
) -> dict[str, float] | None:
    """Method 3 score.

    Null hypothesis:
        After shuffling only OFF-session block labels within run, OFF+GVS is not
        systematically closer to ON-sham than OFF-sham is.
    """

    off_sham_payload = off_condition_payloads.get(SHAM_CONDITION_CODE)
    off_target_payload = off_condition_payloads.get(condition_code)
    if off_sham_payload is None or off_target_payload is None:
        return None

    distance_off_target_vs_on_sham = compute_distance_metric(metric_name, off_target_payload, on_sham_payload)
    distance_off_sham_vs_on_sham = compute_distance_metric(metric_name, off_sham_payload, on_sham_payload)
    if not np.isfinite(distance_off_target_vs_on_sham) or not np.isfinite(distance_off_sham_vs_on_sham):
        return None

    return {
        "distance_off_target_vs_on_sham": float(distance_off_target_vs_on_sham),
        "distance_off_sham_vs_on_sham": float(distance_off_sham_vs_on_sham),
        "normalization_score": float(distance_off_target_vs_on_sham - distance_off_sham_vs_on_sham),
    }


def collect_observed_method1_outputs(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
    distance_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build observed Method 1 subject-level distances and skip logs.

    Null hypothesis:
        Under within-run block-label randomization, sham-referenced active GVS
        distances are no larger than expected by chance.
    """

    subject_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []

    for (subject, session), blocks in sorted(blocks_by_session.items()):
        medication = medication_from_session(session)
        grouped_blocks, condition_payloads = rebuild_session_condition_networks(blocks)
        sham_payload = condition_payloads.get(SHAM_CONDITION_CODE)

        for condition_code, condition_blocks in sorted(grouped_blocks.items()):
            payload = condition_payloads[condition_code]
            inventory_rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "subject_session": f"{subject}_ses-{session}",
                    "medication": medication,
                    "condition_code": condition_code,
                    "condition_name": condition_name_from_code(condition_code),
                    "condition_factor": condition_factor_from_code(condition_code),
                    "n_blocks_pooled": int(len(condition_blocks)),
                    "n_trials_fc": int(payload["n_trials_fc"]),
                }
            )

        if sham_payload is None:
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    skip_rows.append(
                        {
                            "method": "method1",
                            "subject": subject,
                            "session": int(session),
                            "medication": medication,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "skip_reason": "missing_sham_payload",
                        }
                    )
            continue

        for condition_code in ACTIVE_CONDITION_CODES:
            active_payload = condition_payloads.get(condition_code)
            condition_blocks = grouped_blocks.get(condition_code, [])
            if active_payload is None:
                for metric_name in distance_metrics:
                    skip_rows.append(
                        {
                            "method": "method1",
                            "subject": subject,
                            "session": int(session),
                            "medication": medication,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "skip_reason": "missing_active_condition_payload",
                        }
                    )
                continue
            for metric_name in distance_metrics:
                distance_value = compute_distance_metric(metric_name, active_payload, sham_payload)
                if not np.isfinite(distance_value):
                    skip_rows.append(
                        {
                            "method": "method1",
                            "subject": subject,
                            "session": int(session),
                            "medication": medication,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "skip_reason": "non_finite_distance_value",
                        }
                    )
                    continue
                subject_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "subject_session": f"{subject}_ses-{session}",
                        "medication": medication,
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "distance_metric": metric_name,
                        "distance_value": float(distance_value),
                        "n_trials_active_fc": int(active_payload["n_trials_fc"]),
                        "n_trials_sham_fc": int(sham_payload["n_trials_fc"]),
                        "n_blocks_active": int(len(condition_blocks)),
                        "n_blocks_sham": int(len(grouped_blocks.get(SHAM_CONDITION_CODE, []))),
                    }
                )

    observed_df = _frame_from_rows(
        subject_rows,
        ["distance_metric", "medication", "condition_code", "subject", "session"],
    )
    if not observed_df.empty:
        observed_df["condition_factor"] = pd.Categorical(
            observed_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
        observed_df["medication"] = pd.Categorical(
            observed_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )

    inventory_df = _frame_from_rows(inventory_rows, ["subject", "session", "condition_code"])
    if not inventory_df.empty:
        inventory_df["medication"] = pd.Categorical(
            inventory_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )
    skip_df = _ensure_columns(
        _frame_from_rows(skip_rows, ["subject", "session", "condition_code", "distance_metric"]),
        ["method", "subject", "session", "medication", "condition_code", "distance_metric", "skip_reason"],
    )
    return observed_df, inventory_df, skip_df


def collect_observed_method2_outputs(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
    distance_metrics: list[str],
    baseline_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build observed Method 2 subject-level scores and skip logs.

    Null hypothesis:
        Under within-run block-label randomization, the excess-distance score is
        not systematically larger than expected by chance.
    """

    subject_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []

    for (subject, session), blocks in sorted(blocks_by_session.items()):
        medication = medication_from_session(session)
        grouped_blocks, run_payloads = rebuild_run_condition_networks(blocks)

        for (run, condition_code), condition_blocks in sorted(grouped_blocks.items()):
            payload = run_payloads[(run, condition_code)]
            inventory_rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "subject_session": f"{subject}_ses-{session}",
                    "medication": medication,
                    "run": int(run),
                    "condition_code": condition_code,
                    "condition_name": condition_name_from_code(condition_code),
                    "condition_factor": condition_factor_from_code(condition_code),
                    "n_blocks_pooled": int(len(condition_blocks)),
                    "n_trials_fc": int(payload["n_trials_fc"]),
                }
            )

        for condition_code in ACTIVE_CONDITION_CODES:
            for metric_name in distance_metrics:
                score = compute_method2_excess_distance(
                    run_payloads=run_payloads,
                    condition_code=condition_code,
                    metric_name=metric_name,
                    baseline_mode=baseline_mode,
                )
                if score is None:
                    skip_rows.append(
                        {
                            "method": "method2",
                            "subject": subject,
                            "session": int(session),
                            "medication": medication,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "baseline_mode": baseline_mode,
                            "skip_reason": "missing_required_run_level_payload_or_non_finite_score",
                        }
                    )
                    continue
                subject_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "subject_session": f"{subject}_ses-{session}",
                        "medication": medication,
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "distance_metric": metric_name,
                        "baseline_mode": baseline_mode,
                        **score,
                    }
                )

    observed_df = _frame_from_rows(
        subject_rows,
        ["baseline_mode", "distance_metric", "medication", "condition_code", "subject", "session"],
    )
    if not observed_df.empty:
        observed_df["condition_factor"] = pd.Categorical(
            observed_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
        observed_df["medication"] = pd.Categorical(
            observed_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )

    inventory_df = _frame_from_rows(inventory_rows, ["subject", "session", "run", "condition_code"])
    if not inventory_df.empty:
        inventory_df["medication"] = pd.Categorical(
            inventory_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )
    skip_df = _ensure_columns(
        _frame_from_rows(skip_rows, ["subject", "session", "condition_code", "distance_metric"]),
        ["method", "subject", "session", "medication", "condition_code", "distance_metric", "baseline_mode", "skip_reason"],
    )
    return observed_df, inventory_df, skip_df


def collect_observed_method3_outputs(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
    distance_metrics: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build observed Method 3 subject-level normalization scores and skip logs.

    Null hypothesis:
        After randomizing OFF-session block labels, OFF+GVS does not move closer
        to ON-sham than OFF-sham does.
    """

    subject_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []

    subject_ids = sorted({subject for subject, _ in blocks_by_session})
    for subject in subject_ids:
        off_blocks = blocks_by_session.get((subject, 1))
        on_blocks = blocks_by_session.get((subject, 2))
        if off_blocks is None or on_blocks is None:
            inventory_rows.append(
                {
                    "subject": subject,
                    "has_off_session": bool(off_blocks is not None),
                    "has_on_session": bool(on_blocks is not None),
                    "has_on_sham": False,
                    "n_off_blocks": int(len(off_blocks)) if off_blocks is not None else 0,
                    "n_on_blocks": int(len(on_blocks)) if on_blocks is not None else 0,
                    "n_trials_on_sham_fc": float("nan"),
                }
            )
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    skip_rows.append(
                        {
                            "method": "method3",
                            "subject": subject,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "skip_reason": "missing_off_or_on_session",
                        }
                    )
            continue

        _, off_payloads = rebuild_session_condition_networks(off_blocks)
        _, on_payloads = rebuild_session_condition_networks(on_blocks)
        on_sham_payload = on_payloads.get(SHAM_CONDITION_CODE)
        inventory_rows.append(
            {
                "subject": subject,
                "has_off_session": True,
                "has_on_session": True,
                "has_on_sham": bool(on_sham_payload is not None),
                "n_off_blocks": int(len(off_blocks)),
                "n_on_blocks": int(len(on_blocks)),
                "n_trials_on_sham_fc": int(on_sham_payload["n_trials_fc"]) if on_sham_payload is not None else float("nan"),
            }
        )
        if on_sham_payload is None:
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    skip_rows.append(
                        {
                            "method": "method3",
                            "subject": subject,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "skip_reason": "missing_on_sham_payload",
                        }
                    )
            continue

        for condition_code in ACTIVE_CONDITION_CODES:
            for metric_name in distance_metrics:
                score = compute_method3_normalization_score(
                    off_condition_payloads=off_payloads,
                    on_sham_payload=on_sham_payload,
                    condition_code=condition_code,
                    metric_name=metric_name,
                )
                if score is None:
                    skip_rows.append(
                        {
                            "method": "method3",
                            "subject": subject,
                            "condition_code": condition_code,
                            "distance_metric": metric_name,
                            "skip_reason": "missing_required_off_payload_or_non_finite_score",
                        }
                    )
                    continue
                off_target_payload = off_payloads[condition_code]
                off_sham_payload = off_payloads[SHAM_CONDITION_CODE]
                subject_rows.append(
                    {
                        "subject": subject,
                        "source_session_off": 1,
                        "reference_session_on": 2,
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "distance_metric": metric_name,
                        "n_trials_off_target_fc": int(off_target_payload["n_trials_fc"]),
                        "n_trials_off_sham_fc": int(off_sham_payload["n_trials_fc"]),
                        "n_trials_on_sham_fc": int(on_sham_payload["n_trials_fc"]),
                        **score,
                    }
                )

    observed_df = _frame_from_rows(subject_rows, ["distance_metric", "condition_code", "subject"])
    if not observed_df.empty:
        observed_df["condition_factor"] = pd.Categorical(
            observed_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )

    inventory_df = _frame_from_rows(inventory_rows, ["subject"])
    skip_df = _ensure_columns(
        _frame_from_rows(skip_rows, ["subject", "condition_code", "distance_metric"]),
        ["method", "subject", "condition_code", "distance_metric", "skip_reason"],
    )
    return observed_df, inventory_df, skip_df


def _shared_finite_dot_average(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return float("nan"), 0
    value = float(np.dot(x[valid], y[valid]) / float(n_valid))
    return value, n_valid


def compute_method4_crossvalidated_contrast(
    run_payloads: dict[tuple[int, str], dict[str, np.ndarray]],
    condition_code: str,
) -> dict[str, float] | None:
    """Method 4 score.

    Null hypothesis:
        The crossvalidated sham-vs-GVS edge-vector contrast has group mean 0, so
        subject-level sign flips should reproduce the observed group mean.
    """

    required_keys = [
        (1, SHAM_CONDITION_CODE),
        (2, SHAM_CONDITION_CODE),
        (1, condition_code),
        (2, condition_code),
    ]
    if any(key not in run_payloads for key in required_keys):
        return None

    delta_run1 = np.asarray(
        run_payloads[(1, condition_code)]["vec_z"] - run_payloads[(1, SHAM_CONDITION_CODE)]["vec_z"],
        dtype=np.float64,
    )
    delta_run2 = np.asarray(
        run_payloads[(2, condition_code)]["vec_z"] - run_payloads[(2, SHAM_CONDITION_CODE)]["vec_z"],
        dtype=np.float64,
    )
    contrast_value, n_shared_edges = _shared_finite_dot_average(delta_run1, delta_run2)
    if not np.isfinite(contrast_value) or n_shared_edges <= 0:
        return None

    return {
        "crossvalidated_contrast_value": float(contrast_value),
        "n_shared_edges": int(n_shared_edges),
    }


def collect_observed_method4_outputs(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build observed Method 4 subject-level contrasts and skip logs."""

    subject_rows: list[dict[str, Any]] = []
    skip_rows: list[dict[str, Any]] = []

    for (subject, session), blocks in sorted(blocks_by_session.items()):
        medication = medication_from_session(session)
        _, run_payloads = rebuild_run_condition_networks(blocks)
        for condition_code in ACTIVE_CONDITION_CODES:
            score = compute_method4_crossvalidated_contrast(run_payloads=run_payloads, condition_code=condition_code)
            if score is None:
                skip_rows.append(
                    {
                        "method": "method4",
                        "subject": subject,
                        "session": int(session),
                        "medication": medication,
                        "condition_code": condition_code,
                        "skip_reason": "missing_required_run_payloads_or_non_finite_contrast",
                    }
                )
                continue
            subject_rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "subject_session": f"{subject}_ses-{session}",
                    "medication": medication,
                    "condition_code": condition_code,
                    "condition_name": condition_name_from_code(condition_code),
                    "condition_factor": condition_factor_from_code(condition_code),
                    **score,
                }
            )

    observed_df = _frame_from_rows(subject_rows, ["medication", "condition_code", "subject", "session"])
    if not observed_df.empty:
        observed_df["condition_factor"] = pd.Categorical(
            observed_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
        observed_df["medication"] = pd.Categorical(
            observed_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )

    skip_df = _ensure_columns(
        _frame_from_rows(skip_rows, ["subject", "session", "condition_code"]),
        ["method", "subject", "session", "medication", "condition_code", "skip_reason"],
    )
    return observed_df, skip_df


def run_method1_group_permutations(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
    distance_metrics: list[str],
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []

    for permutation_index in range(int(n_permutations)):
        grouped_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for (_, session), blocks in sorted(blocks_by_session.items()):
            medication = medication_from_session(session)
            relabeled_codes = permute_block_labels_within_run(blocks, rng=rng)
            _, condition_payloads = rebuild_session_condition_networks(blocks, relabeled_condition_codes=relabeled_codes)
            sham_payload = condition_payloads.get(SHAM_CONDITION_CODE)
            if sham_payload is None:
                continue
            for condition_code in ACTIVE_CONDITION_CODES:
                active_payload = condition_payloads.get(condition_code)
                if active_payload is None:
                    continue
                for metric_name in distance_metrics:
                    distance_value = compute_distance_metric(metric_name, active_payload, sham_payload)
                    if np.isfinite(distance_value):
                        grouped_values[(medication, condition_code, metric_name)].append(float(distance_value))

        for medication in MEDICATION_ORDER:
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    key = (medication, condition_code, metric_name)
                    values = np.asarray(grouped_values.get(key, []), dtype=np.float64)
                    rows.append(
                        {
                            "permutation_index": int(permutation_index),
                            "medication": medication,
                            "condition_code": condition_code,
                            "condition_name": condition_name_from_code(condition_code),
                            "condition_factor": condition_factor_from_code(condition_code),
                            "distance_metric": metric_name,
                            "group_statistic_perm_mean_distance": float(np.mean(values)) if values.size else float("nan"),
                            "n_subjects_contributing": int(values.size),
                        }
                    )

    permutation_df = pd.DataFrame(rows).sort_values(
        ["distance_metric", "medication", "condition_code", "permutation_index"]
    ).reset_index(drop=True)
    if not permutation_df.empty:
        permutation_df["condition_factor"] = pd.Categorical(
            permutation_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
        permutation_df["medication"] = pd.Categorical(
            permutation_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )
    return permutation_df


def run_method2_group_permutations(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
    distance_metrics: list[str],
    n_permutations: int,
    random_seed: int,
    baseline_mode: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []

    for permutation_index in range(int(n_permutations)):
        grouped_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for (_, session), blocks in sorted(blocks_by_session.items()):
            medication = medication_from_session(session)
            relabeled_codes = permute_block_labels_within_run(blocks, rng=rng)
            _, run_payloads = rebuild_run_condition_networks(blocks, relabeled_condition_codes=relabeled_codes)
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    score = compute_method2_excess_distance(
                        run_payloads=run_payloads,
                        condition_code=condition_code,
                        metric_name=metric_name,
                        baseline_mode=baseline_mode,
                    )
                    if score is None:
                        continue
                    excess_distance = float(score["excess_distance_value"])
                    if np.isfinite(excess_distance):
                        grouped_values[(medication, condition_code, metric_name)].append(excess_distance)

        for medication in MEDICATION_ORDER:
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    key = (medication, condition_code, metric_name)
                    values = np.asarray(grouped_values.get(key, []), dtype=np.float64)
                    rows.append(
                        {
                            "permutation_index": int(permutation_index),
                            "medication": medication,
                            "condition_code": condition_code,
                            "condition_name": condition_name_from_code(condition_code),
                            "condition_factor": condition_factor_from_code(condition_code),
                            "distance_metric": metric_name,
                            "baseline_mode": baseline_mode,
                            "group_statistic_perm_mean_excess_distance": float(np.mean(values)) if values.size else float("nan"),
                            "n_subjects_contributing": int(values.size),
                        }
                    )

    permutation_df = pd.DataFrame(rows).sort_values(
        ["baseline_mode", "distance_metric", "medication", "condition_code", "permutation_index"]
    ).reset_index(drop=True)
    if not permutation_df.empty:
        permutation_df["condition_factor"] = pd.Categorical(
            permutation_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
        permutation_df["medication"] = pd.Categorical(
            permutation_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )
    return permutation_df


def run_method3_group_permutations(
    blocks_by_session: dict[tuple[str, int], tuple[SessionBlock, ...]],
    distance_metrics: list[str],
    n_permutations: int,
    random_seed: int,
) -> pd.DataFrame:
    subject_ids = sorted({subject for subject, _ in blocks_by_session})
    off_blocks_by_subject: dict[str, tuple[SessionBlock, ...]] = {}
    on_sham_by_subject: dict[str, dict[str, np.ndarray]] = {}
    for subject in subject_ids:
        off_blocks = blocks_by_session.get((subject, 1))
        on_blocks = blocks_by_session.get((subject, 2))
        if off_blocks is None or on_blocks is None:
            continue
        _, on_payloads = rebuild_session_condition_networks(on_blocks)
        on_sham_payload = on_payloads.get(SHAM_CONDITION_CODE)
        if on_sham_payload is None:
            continue
        off_blocks_by_subject[subject] = off_blocks
        on_sham_by_subject[subject] = on_sham_payload

    rng = np.random.default_rng(int(random_seed))
    rows: list[dict[str, Any]] = []
    for permutation_index in range(int(n_permutations)):
        grouped_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        for subject in sorted(off_blocks_by_subject):
            off_blocks = off_blocks_by_subject[subject]
            on_sham_payload = on_sham_by_subject[subject]
            relabeled_codes = permute_block_labels_within_run(off_blocks, rng=rng)
            _, off_payloads = rebuild_session_condition_networks(off_blocks, relabeled_condition_codes=relabeled_codes)
            for condition_code in ACTIVE_CONDITION_CODES:
                for metric_name in distance_metrics:
                    score = compute_method3_normalization_score(
                        off_condition_payloads=off_payloads,
                        on_sham_payload=on_sham_payload,
                        condition_code=condition_code,
                        metric_name=metric_name,
                    )
                    if score is None:
                        continue
                    normalization_value = float(score["normalization_score"])
                    if np.isfinite(normalization_value):
                        grouped_values[(condition_code, metric_name)].append(normalization_value)

        for condition_code in ACTIVE_CONDITION_CODES:
            for metric_name in distance_metrics:
                key = (condition_code, metric_name)
                values = np.asarray(grouped_values.get(key, []), dtype=np.float64)
                rows.append(
                    {
                        "permutation_index": int(permutation_index),
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "distance_metric": metric_name,
                        "group_statistic_perm_mean_normalization_score": float(np.mean(values)) if values.size else float("nan"),
                        "n_subjects_contributing": int(values.size),
                    }
                )

    permutation_df = pd.DataFrame(rows).sort_values(
        ["distance_metric", "condition_code", "permutation_index"]
    ).reset_index(drop=True)
    if not permutation_df.empty:
        permutation_df["condition_factor"] = pd.Categorical(
            permutation_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
    return permutation_df


def _iter_exact_sign_flips(n_subjects: int):
    for signs in itertools.product((-1.0, 1.0), repeat=n_subjects):
        if all(sign > 0 for sign in signs):
            continue
        yield np.asarray(signs, dtype=np.float64)


def run_method4_sign_flip_null(
    observed_df: pd.DataFrame,
    n_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))

    if observed_df.empty:
        return pd.DataFrame(
            columns=[
                "flip_index",
                "medication",
                "condition_code",
                "condition_name",
                "condition_factor",
                "null_type",
                "group_statistic_flip_mean_contrast",
                "n_subjects_contributing",
            ]
        )

    group_columns = ["medication", "condition_code", "condition_name", "condition_factor"]
    for group_key, group_df in observed_df.groupby(group_columns, observed=True, sort=True):
        medication, condition_code, condition_name, condition_factor = group_key
        values = group_df["crossvalidated_contrast_value"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        n_subjects = int(values.size)
        if n_subjects == 0:
            continue

        if n_subjects <= EXACT_SIGN_FLIP_MAX_SUBJECTS:
            null_type = "exact_sign_flip"
            sign_iter = _iter_exact_sign_flips(n_subjects)
        else:
            null_type = "monte_carlo_sign_flip"

            def _random_iter():
                for _ in range(int(n_permutations)):
                    yield rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=n_subjects, replace=True)

            sign_iter = _random_iter()

        for flip_index, signs in enumerate(sign_iter):
            rows.append(
                {
                    "flip_index": int(flip_index),
                    "medication": medication,
                    "condition_code": condition_code,
                    "condition_name": condition_name,
                    "condition_factor": condition_factor,
                    "null_type": null_type,
                    "group_statistic_flip_mean_contrast": float(np.mean(values * signs)),
                    "n_subjects_contributing": n_subjects,
                }
            )

    null_df = pd.DataFrame(rows).sort_values(
        ["medication", "condition_code", "flip_index"]
    ).reset_index(drop=True)
    if not null_df.empty:
        null_df["condition_factor"] = pd.Categorical(
            null_df["condition_factor"],
            categories=ACTIVE_CONDITION_FACTORS,
            ordered=True,
        )
        null_df["medication"] = pd.Categorical(
            null_df["medication"],
            categories=MEDICATION_ORDER,
            ordered=True,
        )
    return null_df


def summarize_method1_group_statistics(
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> pd.DataFrame:
    if observed_df.empty:
        return pd.DataFrame(
            columns=[
                "medication",
                "condition_code",
                "condition_name",
                "condition_factor",
                "distance_metric",
                "t_obs_mean_distance",
                "n_subjects_observed",
                "n_permutations_effective",
                "null_mean",
                "null_std",
                "null_q025",
                "null_q975",
                "empirical_p_value",
                "q_value_fdr",
                "significant_fdr",
            ]
        )

    observed_stats = (
        observed_df.groupby(["medication", "condition_code", "condition_name", "condition_factor", "distance_metric"], observed=True)
        .agg(
            t_obs_mean_distance=("distance_value", "mean"),
            n_subjects_observed=("distance_value", lambda values: int(np.count_nonzero(np.isfinite(values)))),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for row in observed_stats.itertuples(index=False):
        null_df = permutation_df.loc[
            (permutation_df["medication"] == row.medication)
            & (permutation_df["condition_code"] == row.condition_code)
            & (permutation_df["distance_metric"] == row.distance_metric)
        ]
        null_values = null_df["group_statistic_perm_mean_distance"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(null_values)
        null_values = null_values[finite_mask]
        if null_values.size:
            empirical_p = float((1 + np.count_nonzero(null_values >= float(row.t_obs_mean_distance))) / (null_values.size + 1))
            null_mean = float(np.mean(null_values))
            null_std = float(np.std(null_values, ddof=1)) if null_values.size >= 2 else float("nan")
            null_q025 = float(np.quantile(null_values, 0.025))
            null_q975 = float(np.quantile(null_values, 0.975))
        else:
            empirical_p = float("nan")
            null_mean = float("nan")
            null_std = float("nan")
            null_q025 = float("nan")
            null_q975 = float("nan")

        rows.append(
            {
                "medication": row.medication,
                "condition_code": row.condition_code,
                "condition_name": row.condition_name,
                "condition_factor": row.condition_factor,
                "distance_metric": row.distance_metric,
                "t_obs_mean_distance": float(row.t_obs_mean_distance),
                "n_subjects_observed": int(row.n_subjects_observed),
                "n_permutations_effective": int(null_values.size),
                "null_mean": null_mean,
                "null_std": null_std,
                "null_q025": null_q025,
                "null_q975": null_q975,
                "empirical_p_value": empirical_p,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values(
        ["distance_metric", "medication", "condition_code"]
    ).reset_index(drop=True)

    q_values = np.full(stats_df.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(stats_df.shape[0], dtype=bool)
    for _, idx in stats_df.groupby(["medication", "distance_metric"], observed=True).groups.items():
        idx_array = np.asarray(list(idx), dtype=np.int64)
        p_values = stats_df.loc[idx_array, "empirical_p_value"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(p_values)
        if not np.any(finite_mask):
            continue
        sig_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
        q_values[idx_array[finite_mask]] = q_valid
        significant[idx_array[finite_mask]] = sig_valid

    stats_df["q_value_fdr"] = q_values
    stats_df["significant_fdr"] = significant
    stats_df["condition_factor"] = pd.Categorical(
        stats_df["condition_factor"],
        categories=ACTIVE_CONDITION_FACTORS,
        ordered=True,
    )
    stats_df["medication"] = pd.Categorical(
        stats_df["medication"],
        categories=MEDICATION_ORDER,
        ordered=True,
    )
    return stats_df


def summarize_method2_group_statistics(
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> pd.DataFrame:
    if observed_df.empty:
        return pd.DataFrame(
            columns=[
                "medication",
                "condition_code",
                "condition_name",
                "condition_factor",
                "distance_metric",
                "baseline_mode",
                "t_obs_mean_excess_distance",
                "n_subjects_observed",
                "n_permutations_effective",
                "null_mean",
                "null_std",
                "null_q025",
                "null_q975",
                "empirical_p_value",
                "q_value_fdr",
                "significant_fdr",
            ]
        )

    observed_stats = (
        observed_df.groupby(
            ["medication", "condition_code", "condition_name", "condition_factor", "distance_metric", "baseline_mode"],
            observed=True,
        )
        .agg(
            t_obs_mean_excess_distance=("excess_distance_value", "mean"),
            n_subjects_observed=("excess_distance_value", lambda values: int(np.count_nonzero(np.isfinite(values)))),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for row in observed_stats.itertuples(index=False):
        null_df = permutation_df.loc[
            (permutation_df["medication"] == row.medication)
            & (permutation_df["condition_code"] == row.condition_code)
            & (permutation_df["distance_metric"] == row.distance_metric)
            & (permutation_df["baseline_mode"] == row.baseline_mode)
        ]
        null_values = null_df["group_statistic_perm_mean_excess_distance"].to_numpy(dtype=np.float64)
        null_values = null_values[np.isfinite(null_values)]
        if null_values.size:
            empirical_p = float(
                (1 + np.count_nonzero(null_values >= float(row.t_obs_mean_excess_distance))) / (null_values.size + 1)
            )
            null_mean = float(np.mean(null_values))
            null_std = float(np.std(null_values, ddof=1)) if null_values.size >= 2 else float("nan")
            null_q025 = float(np.quantile(null_values, 0.025))
            null_q975 = float(np.quantile(null_values, 0.975))
        else:
            empirical_p = float("nan")
            null_mean = float("nan")
            null_std = float("nan")
            null_q025 = float("nan")
            null_q975 = float("nan")

        rows.append(
            {
                "medication": row.medication,
                "condition_code": row.condition_code,
                "condition_name": row.condition_name,
                "condition_factor": row.condition_factor,
                "distance_metric": row.distance_metric,
                "baseline_mode": row.baseline_mode,
                "t_obs_mean_excess_distance": float(row.t_obs_mean_excess_distance),
                "n_subjects_observed": int(row.n_subjects_observed),
                "n_permutations_effective": int(null_values.size),
                "null_mean": null_mean,
                "null_std": null_std,
                "null_q025": null_q025,
                "null_q975": null_q975,
                "empirical_p_value": empirical_p,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values(
        ["baseline_mode", "distance_metric", "medication", "condition_code"]
    ).reset_index(drop=True)

    q_values = np.full(stats_df.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(stats_df.shape[0], dtype=bool)
    for _, idx in stats_df.groupby(["baseline_mode", "medication", "distance_metric"], observed=True).groups.items():
        idx_array = np.asarray(list(idx), dtype=np.int64)
        p_values = stats_df.loc[idx_array, "empirical_p_value"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(p_values)
        if not np.any(finite_mask):
            continue
        sig_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
        q_values[idx_array[finite_mask]] = q_valid
        significant[idx_array[finite_mask]] = sig_valid

    stats_df["q_value_fdr"] = q_values
    stats_df["significant_fdr"] = significant
    stats_df["condition_factor"] = pd.Categorical(
        stats_df["condition_factor"],
        categories=ACTIVE_CONDITION_FACTORS,
        ordered=True,
    )
    stats_df["medication"] = pd.Categorical(
        stats_df["medication"],
        categories=MEDICATION_ORDER,
        ordered=True,
    )
    return stats_df


def summarize_method3_group_statistics(
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
) -> pd.DataFrame:
    if observed_df.empty:
        return pd.DataFrame(
            columns=[
                "condition_code",
                "condition_name",
                "condition_factor",
                "distance_metric",
                "t_obs_mean_normalization_score",
                "n_subjects_observed",
                "n_permutations_effective",
                "null_mean",
                "null_std",
                "null_q025",
                "null_q975",
                "empirical_p_value_normalization_one_sided",
                "q_value_fdr",
                "significant_fdr",
            ]
        )

    observed_stats = (
        observed_df.groupby(
            ["condition_code", "condition_name", "condition_factor", "distance_metric"],
            observed=True,
        )
        .agg(
            t_obs_mean_normalization_score=("normalization_score", "mean"),
            n_subjects_observed=("normalization_score", lambda values: int(np.count_nonzero(np.isfinite(values)))),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for row in observed_stats.itertuples(index=False):
        null_df = permutation_df.loc[
            (permutation_df["condition_code"] == row.condition_code)
            & (permutation_df["distance_metric"] == row.distance_metric)
        ]
        null_values = null_df["group_statistic_perm_mean_normalization_score"].to_numpy(dtype=np.float64)
        null_values = null_values[np.isfinite(null_values)]
        if null_values.size:
            empirical_p = float(
                (1 + np.count_nonzero(null_values <= float(row.t_obs_mean_normalization_score))) / (null_values.size + 1)
            )
            null_mean = float(np.mean(null_values))
            null_std = float(np.std(null_values, ddof=1)) if null_values.size >= 2 else float("nan")
            null_q025 = float(np.quantile(null_values, 0.025))
            null_q975 = float(np.quantile(null_values, 0.975))
        else:
            empirical_p = float("nan")
            null_mean = float("nan")
            null_std = float("nan")
            null_q025 = float("nan")
            null_q975 = float("nan")

        rows.append(
            {
                "condition_code": row.condition_code,
                "condition_name": row.condition_name,
                "condition_factor": row.condition_factor,
                "distance_metric": row.distance_metric,
                "t_obs_mean_normalization_score": float(row.t_obs_mean_normalization_score),
                "n_subjects_observed": int(row.n_subjects_observed),
                "n_permutations_effective": int(null_values.size),
                "null_mean": null_mean,
                "null_std": null_std,
                "null_q025": null_q025,
                "null_q975": null_q975,
                "empirical_p_value_normalization_one_sided": empirical_p,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values(["distance_metric", "condition_code"]).reset_index(drop=True)
    q_values = np.full(stats_df.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(stats_df.shape[0], dtype=bool)
    for _, idx in stats_df.groupby(["distance_metric"], observed=True).groups.items():
        idx_array = np.asarray(list(idx), dtype=np.int64)
        p_values = stats_df.loc[idx_array, "empirical_p_value_normalization_one_sided"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(p_values)
        if not np.any(finite_mask):
            continue
        sig_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
        q_values[idx_array[finite_mask]] = q_valid
        significant[idx_array[finite_mask]] = sig_valid

    stats_df["q_value_fdr"] = q_values
    stats_df["significant_fdr"] = significant
    stats_df["condition_factor"] = pd.Categorical(
        stats_df["condition_factor"],
        categories=ACTIVE_CONDITION_FACTORS,
        ordered=True,
    )
    return stats_df


def summarize_method4_group_statistics(
    observed_df: pd.DataFrame,
    null_df: pd.DataFrame,
) -> pd.DataFrame:
    if observed_df.empty:
        return pd.DataFrame(
            columns=[
                "medication",
                "condition_code",
                "condition_name",
                "condition_factor",
                "t_obs_mean_contrast",
                "n_subjects_observed",
                "n_sign_flips_effective",
                "null_type",
                "null_mean",
                "null_std",
                "null_q025",
                "null_q975",
                "empirical_p_value",
                "q_value_fdr",
                "significant_fdr",
            ]
        )

    observed_stats = (
        observed_df.groupby(
            ["medication", "condition_code", "condition_name", "condition_factor"],
            observed=True,
        )
        .agg(
            t_obs_mean_contrast=("crossvalidated_contrast_value", "mean"),
            n_subjects_observed=("crossvalidated_contrast_value", lambda values: int(np.count_nonzero(np.isfinite(values)))),
        )
        .reset_index()
    )

    rows: list[dict[str, Any]] = []
    for row in observed_stats.itertuples(index=False):
        group_null = null_df.loc[
            (null_df["medication"] == row.medication)
            & (null_df["condition_code"] == row.condition_code)
        ]
        null_values = group_null["group_statistic_flip_mean_contrast"].to_numpy(dtype=np.float64)
        null_values = null_values[np.isfinite(null_values)]
        null_type_values = group_null["null_type"].dropna().astype(str).unique().tolist()
        null_type = null_type_values[0] if null_type_values else ""
        if null_values.size:
            empirical_p = float((1 + np.count_nonzero(null_values >= float(row.t_obs_mean_contrast))) / (null_values.size + 1))
            null_mean = float(np.mean(null_values))
            null_std = float(np.std(null_values, ddof=1)) if null_values.size >= 2 else float("nan")
            null_q025 = float(np.quantile(null_values, 0.025))
            null_q975 = float(np.quantile(null_values, 0.975))
        else:
            empirical_p = float("nan")
            null_mean = float("nan")
            null_std = float("nan")
            null_q025 = float("nan")
            null_q975 = float("nan")
        rows.append(
            {
                "medication": row.medication,
                "condition_code": row.condition_code,
                "condition_name": row.condition_name,
                "condition_factor": row.condition_factor,
                "t_obs_mean_contrast": float(row.t_obs_mean_contrast),
                "n_subjects_observed": int(row.n_subjects_observed),
                "n_sign_flips_effective": int(null_values.size),
                "null_type": null_type,
                "null_mean": null_mean,
                "null_std": null_std,
                "null_q025": null_q025,
                "null_q975": null_q975,
                "empirical_p_value": empirical_p,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values(["medication", "condition_code"]).reset_index(drop=True)
    q_values = np.full(stats_df.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(stats_df.shape[0], dtype=bool)
    for _, idx in stats_df.groupby(["medication"], observed=True).groups.items():
        idx_array = np.asarray(list(idx), dtype=np.int64)
        p_values = stats_df.loc[idx_array, "empirical_p_value"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(p_values)
        if not np.any(finite_mask):
            continue
        sig_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
        q_values[idx_array[finite_mask]] = q_valid
        significant[idx_array[finite_mask]] = sig_valid
    stats_df["q_value_fdr"] = q_values
    stats_df["significant_fdr"] = significant
    stats_df["condition_factor"] = pd.Categorical(
        stats_df["condition_factor"],
        categories=ACTIVE_CONDITION_FACTORS,
        ordered=True,
    )
    stats_df["medication"] = pd.Categorical(
        stats_df["medication"],
        categories=MEDICATION_ORDER,
        ordered=True,
    )
    return stats_df


def plot_method1_by_condition(
    observed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metric: str,
    out_path: Path,
) -> None:
    metric_df = observed_df.loc[observed_df["distance_metric"] == distance_metric].copy()
    if metric_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(ACTIVE_CONDITION_FACTORS), dtype=np.float64)
    offsets = {"OFF": -0.12, "ON": 0.12}
    colors = {"OFF": "#4c78a8", "ON": "#f58518"}
    mean_by_medication: dict[str, np.ndarray] = {}
    sem_by_medication: dict[str, np.ndarray] = {}

    for medication in MEDICATION_ORDER:
        medication_df = metric_df.loc[metric_df["medication"] == medication]
        means: list[float] = []
        sems: list[float] = []
        for condition_factor in ACTIVE_CONDITION_FACTORS:
            values = medication_df.loc[
                medication_df["condition_factor"] == condition_factor,
                "distance_value",
            ].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            sems.append(float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size >= 2 else float("nan"))
        mean_array = np.asarray(means, dtype=np.float64)
        sem_array = np.asarray(sems, dtype=np.float64)
        mean_by_medication[medication] = mean_array
        sem_by_medication[medication] = sem_array
        ax.errorbar(
            x_positions + offsets[medication],
            mean_array,
            yerr=sem_array,
            fmt="o-",
            color=colors[medication],
            linewidth=1.5,
            markersize=5,
            capsize=3,
            label=medication,
        )

    sig_df = stats_df.loc[
        (stats_df["distance_metric"] == distance_metric)
        & stats_df["significant_fdr"].fillna(False)
        & stats_df["condition_factor"].isin(ACTIVE_CONDITION_FACTORS)
    ].copy()
    if not sig_df.empty:
        data_extent: list[float] = []
        for medication in MEDICATION_ORDER:
            means = mean_by_medication.get(medication)
            sems = sem_by_medication.get(medication)
            if means is None or sems is None:
                continue
            for idx in range(len(ACTIVE_CONDITION_FACTORS)):
                if np.isfinite(means[idx]):
                    data_extent.append(float(means[idx]))
                if np.isfinite(means[idx]) and np.isfinite(sems[idx]):
                    data_extent.append(float(means[idx] + sems[idx]))
        y_span = max(data_extent) - min(data_extent) if len(data_extent) >= 2 else 1.0
        y_pad = max(0.05, 0.06 * y_span)
        star_offset = 0.55 * y_pad
        top_margin = 1.6 * y_pad
        star_tops: list[float] = []
        for row in sig_df.itertuples(index=False):
            condition_idx = ACTIVE_CONDITION_FACTORS.index(str(row.condition_factor))
            mean_array = mean_by_medication.get(str(row.medication))
            sem_array = sem_by_medication.get(str(row.medication))
            if mean_array is None or not np.isfinite(mean_array[condition_idx]):
                continue
            sem_value = sem_array[condition_idx] if sem_array is not None and np.isfinite(sem_array[condition_idx]) else 0.0
            star_y = float(mean_array[condition_idx] + sem_value + star_offset)
            ax.text(
                x_positions[condition_idx] + offsets[str(row.medication)],
                star_y,
                "*",
                ha="center",
                va="bottom",
                color=colors[str(row.medication)],
                fontsize=16,
                fontweight="bold",
            )
            star_tops.append(star_y)
        if star_tops:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, max(ymax, max(star_tops) + top_margin))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ACTIVE_CONDITION_FACTORS, rotation=45, ha="right")
    ax.set_xlabel("Active GVS condition")
    ax.set_ylabel(DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric))
    ax.set_title(f"Method 1 observed distance to sham\n{DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric)}")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method2_by_condition(
    observed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metric: str,
    baseline_mode: str,
    out_path: Path,
) -> None:
    metric_df = observed_df.loc[
        (observed_df["distance_metric"] == distance_metric)
        & (observed_df["baseline_mode"] == baseline_mode)
    ].copy()
    if metric_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(ACTIVE_CONDITION_FACTORS), dtype=np.float64)
    offsets = {"OFF": -0.12, "ON": 0.12}
    colors = {"OFF": "#4c78a8", "ON": "#f58518"}
    mean_by_medication: dict[str, np.ndarray] = {}
    sem_by_medication: dict[str, np.ndarray] = {}

    for medication in MEDICATION_ORDER:
        medication_df = metric_df.loc[metric_df["medication"] == medication]
        means: list[float] = []
        sems: list[float] = []
        for condition_factor in ACTIVE_CONDITION_FACTORS:
            values = medication_df.loc[
                medication_df["condition_factor"] == condition_factor,
                "excess_distance_value",
            ].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            sems.append(float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size >= 2 else float("nan"))
        mean_array = np.asarray(means, dtype=np.float64)
        sem_array = np.asarray(sems, dtype=np.float64)
        mean_by_medication[medication] = mean_array
        sem_by_medication[medication] = sem_array
        ax.errorbar(
            x_positions + offsets[medication],
            mean_array,
            yerr=sem_array,
            fmt="o-",
            color=colors[medication],
            linewidth=1.5,
            markersize=5,
            capsize=3,
            label=medication,
        )

    sig_df = stats_df.loc[
        (stats_df["distance_metric"] == distance_metric)
        & (stats_df["baseline_mode"] == baseline_mode)
        & stats_df["significant_fdr"].fillna(False)
        & stats_df["condition_factor"].isin(ACTIVE_CONDITION_FACTORS)
    ].copy()
    if not sig_df.empty:
        data_extent: list[float] = []
        for medication in MEDICATION_ORDER:
            means = mean_by_medication.get(medication)
            sems = sem_by_medication.get(medication)
            if means is None or sems is None:
                continue
            for idx in range(len(ACTIVE_CONDITION_FACTORS)):
                if np.isfinite(means[idx]):
                    data_extent.append(float(means[idx]))
                if np.isfinite(means[idx]) and np.isfinite(sems[idx]):
                    data_extent.append(float(means[idx] + sems[idx]))
        y_span = max(data_extent) - min(data_extent) if len(data_extent) >= 2 else 1.0
        y_pad = max(0.05, 0.06 * y_span)
        star_offset = 0.55 * y_pad
        top_margin = 1.6 * y_pad
        star_tops: list[float] = []
        for row in sig_df.itertuples(index=False):
            condition_idx = ACTIVE_CONDITION_FACTORS.index(str(row.condition_factor))
            mean_array = mean_by_medication.get(str(row.medication))
            sem_array = sem_by_medication.get(str(row.medication))
            if mean_array is None or not np.isfinite(mean_array[condition_idx]):
                continue
            sem_value = sem_array[condition_idx] if sem_array is not None and np.isfinite(sem_array[condition_idx]) else 0.0
            star_y = float(mean_array[condition_idx] + sem_value + star_offset)
            ax.text(
                x_positions[condition_idx] + offsets[str(row.medication)],
                star_y,
                "*",
                ha="center",
                va="bottom",
                color=colors[str(row.medication)],
                fontsize=16,
                fontweight="bold",
            )
            star_tops.append(star_y)
        if star_tops:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, max(ymax, max(star_tops) + top_margin))

    ax.axhline(0.0, color="#6e6e6e", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ACTIVE_CONDITION_FACTORS, rotation=45, ha="right")
    ax.set_xlabel("Active GVS condition")
    ax.set_ylabel(f"Excess distance\n{DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric)}")
    ax.set_title(
        "Method 2 excess-distance score\n"
        f"{DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric)} | {METHOD2_BASELINE_LABELS[baseline_mode]}"
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method3_by_condition(
    observed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metric: str,
    out_path: Path,
) -> None:
    metric_df = observed_df.loc[observed_df["distance_metric"] == distance_metric].copy()
    if metric_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    x_positions = np.arange(len(ACTIVE_CONDITION_FACTORS), dtype=np.float64)
    means: list[float] = []
    sems: list[float] = []
    for condition_factor in ACTIVE_CONDITION_FACTORS:
        values = metric_df.loc[metric_df["condition_factor"] == condition_factor, "normalization_score"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        means.append(float(np.mean(values)) if values.size else float("nan"))
        sems.append(float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size >= 2 else float("nan"))
    mean_array = np.asarray(means, dtype=np.float64)
    sem_array = np.asarray(sems, dtype=np.float64)

    ax.errorbar(
        x_positions,
        mean_array,
        yerr=sem_array,
        fmt="o-",
        color="#54a24b",
        linewidth=1.6,
        markersize=5,
        capsize=3,
    )

    sig_df = stats_df.loc[
        (stats_df["distance_metric"] == distance_metric)
        & stats_df["significant_fdr"].fillna(False)
        & stats_df["condition_factor"].isin(ACTIVE_CONDITION_FACTORS)
    ].copy()
    if not sig_df.empty:
        data_extent: list[float] = []
        for idx in range(len(ACTIVE_CONDITION_FACTORS)):
            if np.isfinite(mean_array[idx]):
                data_extent.append(float(mean_array[idx]))
            if np.isfinite(mean_array[idx]) and np.isfinite(sem_array[idx]):
                data_extent.append(float(mean_array[idx] + sem_array[idx]))
        y_span = max(data_extent) - min(data_extent) if len(data_extent) >= 2 else 1.0
        y_pad = max(0.05, 0.06 * y_span)
        star_offset = 0.55 * y_pad
        top_margin = 1.6 * y_pad
        star_tops: list[float] = []
        for row in sig_df.itertuples(index=False):
            condition_idx = ACTIVE_CONDITION_FACTORS.index(str(row.condition_factor))
            if not np.isfinite(mean_array[condition_idx]):
                continue
            sem_value = sem_array[condition_idx] if np.isfinite(sem_array[condition_idx]) else 0.0
            star_y = float(mean_array[condition_idx] + sem_value + star_offset)
            ax.text(
                x_positions[condition_idx],
                star_y,
                "*",
                ha="center",
                va="bottom",
                color="#54a24b",
                fontsize=16,
                fontweight="bold",
            )
            star_tops.append(star_y)
        if star_tops:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, max(ymax, max(star_tops) + top_margin))

    ax.axhline(0.0, color="#6e6e6e", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ACTIVE_CONDITION_FACTORS, rotation=45, ha="right")
    ax.set_xlabel("Active GVS condition")
    ax.set_ylabel(f"Normalization score\n{DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric)}")
    ax.set_title(
        "Method 3 OFF-to-ON normalization\n"
        f"{DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric)} | lower is more normalization"
    )
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_method4_by_condition(
    observed_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if observed_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(ACTIVE_CONDITION_FACTORS), dtype=np.float64)
    offsets = {"OFF": -0.12, "ON": 0.12}
    colors = {"OFF": "#4c78a8", "ON": "#f58518"}
    mean_by_medication: dict[str, np.ndarray] = {}
    sem_by_medication: dict[str, np.ndarray] = {}

    for medication in MEDICATION_ORDER:
        medication_df = observed_df.loc[observed_df["medication"] == medication]
        means: list[float] = []
        sems: list[float] = []
        for condition_factor in ACTIVE_CONDITION_FACTORS:
            values = medication_df.loc[
                medication_df["condition_factor"] == condition_factor,
                "crossvalidated_contrast_value",
            ].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            sems.append(float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size >= 2 else float("nan"))
        mean_array = np.asarray(means, dtype=np.float64)
        sem_array = np.asarray(sems, dtype=np.float64)
        mean_by_medication[medication] = mean_array
        sem_by_medication[medication] = sem_array
        ax.errorbar(
            x_positions + offsets[medication],
            mean_array,
            yerr=sem_array,
            fmt="o-",
            color=colors[medication],
            linewidth=1.5,
            markersize=5,
            capsize=3,
            label=medication,
        )

    sig_df = stats_df.loc[
        stats_df["significant_fdr"].fillna(False)
        & stats_df["condition_factor"].isin(ACTIVE_CONDITION_FACTORS)
    ].copy()
    if not sig_df.empty:
        data_extent: list[float] = []
        for medication in MEDICATION_ORDER:
            means = mean_by_medication.get(medication)
            sems = sem_by_medication.get(medication)
            if means is None or sems is None:
                continue
            for idx in range(len(ACTIVE_CONDITION_FACTORS)):
                if np.isfinite(means[idx]):
                    data_extent.append(float(means[idx]))
                if np.isfinite(means[idx]) and np.isfinite(sems[idx]):
                    data_extent.append(float(means[idx] + sems[idx]))
        y_span = max(data_extent) - min(data_extent) if len(data_extent) >= 2 else 1.0
        y_pad = max(0.05, 0.06 * y_span)
        star_offset = 0.55 * y_pad
        top_margin = 1.6 * y_pad
        star_tops: list[float] = []
        for row in sig_df.itertuples(index=False):
            condition_idx = ACTIVE_CONDITION_FACTORS.index(str(row.condition_factor))
            mean_array = mean_by_medication.get(str(row.medication))
            sem_array = sem_by_medication.get(str(row.medication))
            if mean_array is None or not np.isfinite(mean_array[condition_idx]):
                continue
            sem_value = sem_array[condition_idx] if sem_array is not None and np.isfinite(sem_array[condition_idx]) else 0.0
            star_y = float(mean_array[condition_idx] + sem_value + star_offset)
            ax.text(
                x_positions[condition_idx] + offsets[str(row.medication)],
                star_y,
                "*",
                ha="center",
                va="bottom",
                color=colors[str(row.medication)],
                fontsize=16,
                fontweight="bold",
            )
            star_tops.append(star_y)
        if star_tops:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, max(ymax, max(star_tops) + top_margin))

    ax.axhline(0.0, color="#6e6e6e", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ACTIVE_CONDITION_FACTORS, rotation=45, ha="right")
    ax.set_xlabel("Active GVS condition")
    ax.set_ylabel("Crossvalidated contrast")
    ax.set_title("Method 4 crossvalidated sham-vs-GVS contrast")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_manifest_payload(
    args: argparse.Namespace,
    roi_labels: list[str],
    block_inventory_df: pd.DataFrame,
    condition_inventory_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metrics: list[str],
) -> dict[str, Any]:
    return {
        "script": str(Path(__file__).resolve()),
        "implemented_methods": ["method1_sham_referenced_raw_distance"],
        "gvs_order_path": str(args.gvs_order_path.expanduser().resolve()),
        "manifest_path": str(args.manifest_path.expanduser().resolve()),
        "session_beta_dir": str(args.session_beta_dir.expanduser().resolve()),
        "selected_voxels_path": str(args.selected_voxels_path.expanduser().resolve()),
        "roi_img": str(args.roi_img.expanduser().resolve()),
        "roi_summary": str(args.roi_summary.expanduser().resolve()),
        "out_dir": str(args.out_dir.expanduser().resolve()),
        "trials_per_block": int(args.trials_per_block),
        "min_roi_voxels": int(args.min_roi_voxels),
        "distance_metrics": distance_metrics,
        "primary_distance_metric": PRIMARY_DISTANCE_METRIC,
        "secondary_distance_metrics": list(SECONDARY_DISTANCE_METRICS),
        "n_permutations_requested": int(args.n_permutations),
        "seed": int(args.seed),
        "sham_condition_code": SHAM_CONDITION_CODE,
        "active_condition_codes": ACTIVE_CONDITION_CODES,
        "n_rois": int(len(roi_labels)),
        "n_session_blocks": int(block_inventory_df.shape[0]),
        "n_observed_condition_networks": int(condition_inventory_df.shape[0]),
        "n_subject_level_method1_rows": int(observed_df.shape[0]),
        "n_group_permutation_rows": int(permutation_df.shape[0]),
        "n_group_stat_rows": int(stats_df.shape[0]),
    }


def build_method2_manifest_payload(
    args: argparse.Namespace,
    roi_labels: list[str],
    run_inventory_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metrics: list[str],
) -> dict[str, Any]:
    return {
        "script": str(Path(__file__).resolve()),
        "implemented_methods": ["method2_excess_distance_baseline_adjusted"],
        "gvs_order_path": str(args.gvs_order_path.expanduser().resolve()),
        "manifest_path": str(args.manifest_path.expanduser().resolve()),
        "session_beta_dir": str(args.session_beta_dir.expanduser().resolve()),
        "selected_voxels_path": str(args.selected_voxels_path.expanduser().resolve()),
        "roi_img": str(args.roi_img.expanduser().resolve()),
        "roi_summary": str(args.roi_summary.expanduser().resolve()),
        "out_dir": str(args.out_dir.expanduser().resolve()),
        "trials_per_block": int(args.trials_per_block),
        "min_roi_voxels": int(args.min_roi_voxels),
        "distance_metrics": distance_metrics,
        "primary_distance_metric": PRIMARY_DISTANCE_METRIC,
        "secondary_distance_metrics": list(SECONDARY_DISTANCE_METRICS),
        "n_permutations_requested": int(args.n_permutations),
        "seed": int(args.seed),
        "baseline_mode": normalize_baseline_mode(args.baseline_mode),
        "sham_condition_code": SHAM_CONDITION_CODE,
        "active_condition_codes": ACTIVE_CONDITION_CODES,
        "n_rois": int(len(roi_labels)),
        "n_observed_run_condition_networks": int(run_inventory_df.shape[0]),
        "n_subject_level_method2_rows": int(observed_df.shape[0]),
        "n_group_permutation_rows": int(permutation_df.shape[0]),
        "n_group_stat_rows": int(stats_df.shape[0]),
    }


def build_method3_manifest_payload(
    args: argparse.Namespace,
    roi_labels: list[str],
    subject_inventory_df: pd.DataFrame,
    observed_df: pd.DataFrame,
    permutation_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metrics: list[str],
) -> dict[str, Any]:
    return {
        "script": str(Path(__file__).resolve()),
        "implemented_methods": ["method3_off_to_on_normalization"],
        "gvs_order_path": str(args.gvs_order_path.expanduser().resolve()),
        "manifest_path": str(args.manifest_path.expanduser().resolve()),
        "session_beta_dir": str(args.session_beta_dir.expanduser().resolve()),
        "selected_voxels_path": str(args.selected_voxels_path.expanduser().resolve()),
        "roi_img": str(args.roi_img.expanduser().resolve()),
        "roi_summary": str(args.roi_summary.expanduser().resolve()),
        "out_dir": str(args.out_dir.expanduser().resolve()),
        "trials_per_block": int(args.trials_per_block),
        "min_roi_voxels": int(args.min_roi_voxels),
        "distance_metrics": distance_metrics,
        "primary_distance_metric": PRIMARY_DISTANCE_METRIC,
        "secondary_distance_metrics": list(SECONDARY_DISTANCE_METRICS),
        "n_permutations_requested": int(args.n_permutations),
        "seed": int(args.seed),
        "sham_condition_code": SHAM_CONDITION_CODE,
        "active_condition_codes": ACTIVE_CONDITION_CODES,
        "n_rois": int(len(roi_labels)),
        "n_subject_inventory_rows": int(subject_inventory_df.shape[0]),
        "n_subject_level_method3_rows": int(observed_df.shape[0]),
        "n_group_permutation_rows": int(permutation_df.shape[0]),
        "n_group_stat_rows": int(stats_df.shape[0]),
    }


def build_method4_manifest_payload(
    args: argparse.Namespace,
    roi_labels: list[str],
    observed_df: pd.DataFrame,
    skip_df: pd.DataFrame,
    null_df: pd.DataFrame,
    stats_df: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "script": str(Path(__file__).resolve()),
        "implemented_methods": ["method4_crossvalidated_sham_contrast"],
        "gvs_order_path": str(args.gvs_order_path.expanduser().resolve()),
        "manifest_path": str(args.manifest_path.expanduser().resolve()),
        "session_beta_dir": str(args.session_beta_dir.expanduser().resolve()),
        "selected_voxels_path": str(args.selected_voxels_path.expanduser().resolve()),
        "roi_img": str(args.roi_img.expanduser().resolve()),
        "roi_summary": str(args.roi_summary.expanduser().resolve()),
        "out_dir": str(args.out_dir.expanduser().resolve()),
        "trials_per_block": int(args.trials_per_block),
        "min_roi_voxels": int(args.min_roi_voxels),
        "n_permutations_requested": int(args.n_permutations),
        "seed": int(args.seed),
        "sham_condition_code": SHAM_CONDITION_CODE,
        "active_condition_codes": ACTIVE_CONDITION_CODES,
        "n_rois": int(len(roi_labels)),
        "n_subject_level_method4_rows": int(observed_df.shape[0]),
        "n_method4_skip_rows": int(skip_df.shape[0]),
        "n_sign_flip_rows": int(null_df.shape[0]),
        "n_group_stat_rows": int(stats_df.shape[0]),
    }


def write_report(
    out_root: Path,
    selected_methods: list[str],
    args: argparse.Namespace,
    method_stats: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# GVS Network Methods v2",
        "",
        "This report summarizes the inferential targets saved by the standalone GVS methods script.",
        "",
        "## Analysis choices",
        f"- Methods run: {', '.join(selected_methods)}",
        f"- Distance metrics for Methods 1-3: {', '.join(parse_distance_metrics(args.distance_metrics))}",
        f"- Method 2 baseline mode: {normalize_baseline_mode(args.baseline_mode)}",
        f"- Randomization/sign-flip draws requested: {int(args.n_permutations)}",
        f"- Seed: {int(args.seed)}",
        "",
        "## Interpretation",
        "- Method 1: positive values mean the active condition moved the session-level network farther from sham than expected under within-run block-label randomization.",
        "- Method 2: positive values mean the active condition exceeded the within-condition baseline variability more than expected under within-run block-label randomization.",
        "- Method 3: negative values mean OFF+GVS moved closer to ON-sham than OFF-sham did. Lower one-sided p-values support normalization.",
        "- Method 4: positive values mean the sham-vs-GVS edge contrast replicated across runs. Sign-flip nulls are centered at zero by construction.",
        "",
    ]

    for method_name in selected_methods:
        stats_df = method_stats.get(method_name)
        lines.append(f"## {METHOD_LABELS[method_name]}")
        if stats_df is None or stats_df.empty:
            lines.append("- No analyzable cells were available.")
            lines.append("")
            continue

        if method_name == "method1":
            top_df = stats_df.sort_values(["q_value_fdr", "empirical_p_value"]).head(6)
            for row in top_df.itertuples(index=False):
                lines.append(
                    f"- {row.distance_metric} | {row.medication} | {row.condition_factor}: "
                    f"mean distance={row.t_obs_mean_distance:.4g}, p={row.empirical_p_value:.4g}, q={row.q_value_fdr:.4g}"
                )
        elif method_name == "method2":
            top_df = stats_df.sort_values(["q_value_fdr", "empirical_p_value"]).head(6)
            for row in top_df.itertuples(index=False):
                lines.append(
                    f"- {row.distance_metric} | {row.medication} | {row.condition_factor}: "
                    f"mean excess={row.t_obs_mean_excess_distance:.4g}, p={row.empirical_p_value:.4g}, q={row.q_value_fdr:.4g}"
                )
        elif method_name == "method3":
            top_df = stats_df.sort_values(["q_value_fdr", "empirical_p_value_normalization_one_sided"]).head(6)
            for row in top_df.itertuples(index=False):
                lines.append(
                    f"- {row.distance_metric} | {row.condition_factor}: "
                    f"mean normalization={row.t_obs_mean_normalization_score:.4g}, "
                    f"p={row.empirical_p_value_normalization_one_sided:.4g}, q={row.q_value_fdr:.4g}"
                )
        elif method_name == "method4":
            top_df = stats_df.sort_values(["q_value_fdr", "empirical_p_value"]).head(6)
            for row in top_df.itertuples(index=False):
                lines.append(
                    f"- {row.medication} | {row.condition_factor}: "
                    f"mean contrast={row.t_obs_mean_contrast:.4g}, p={row.empirical_p_value:.4g}, q={row.q_value_fdr:.4g}"
                )
        lines.append("")

    (out_root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.trials_per_block <= 0:
        raise ValueError(f"--trials-per-block must be positive, got {args.trials_per_block}.")
    if args.n_permutations <= 0:
        raise ValueError(f"--n-permutations must be positive, got {args.n_permutations}.")

    selected_methods = parse_methods(args.methods)
    distance_metrics = parse_distance_metrics(args.distance_metrics)
    baseline_mode = normalize_baseline_mode(args.baseline_mode)

    out_root = ensure_dir(args.out_dir.expanduser().resolve())
    common_dir = ensure_dir(out_root / "common")

    gvs_orders = _load_gvs_orders(args.gvs_order_path.expanduser().resolve())
    manifest_rows, _ = _load_manifest_rows(
        args.manifest_path.expanduser().resolve(),
        gvs_orders,
        int(args.trials_per_block),
    )

    roi_members, roi_labels, roi_meta_df = build_roi_membership(
        roi_img_path=args.roi_img.expanduser().resolve(),
        roi_summary_path=args.roi_summary.expanduser().resolve(),
        selected_voxels_path=args.selected_voxels_path.expanduser().resolve(),
        min_roi_voxels=int(args.min_roi_voxels),
    )
    roi_meta_df.to_csv(common_dir / "roi_nodes.csv", index=False)

    blocks_by_session, block_inventory_df, session_skip_df = reconstruct_session_blocks(
        manifest_rows=manifest_rows,
        gvs_orders=gvs_orders,
        session_beta_dir=args.session_beta_dir.expanduser().resolve(),
        roi_members=roi_members,
        trials_per_block=int(args.trials_per_block),
    )
    block_inventory_df.to_csv(common_dir / "session_block_inventory.csv", index=False)
    session_skip_df.to_csv(common_dir / "session_reconstruction_skips.csv", index=False)

    method_stats: dict[str, pd.DataFrame] = {}

    if "method1" in selected_methods:
        method1_dir = ensure_dir(out_root / METHOD1_DIRNAME)
        tables_dir = ensure_dir(method1_dir / "tables")
        plots_dir = ensure_dir(method1_dir / "plots")

        observed_df, condition_inventory_df, method1_skip_df = collect_observed_method1_outputs(
            blocks_by_session=blocks_by_session,
            distance_metrics=distance_metrics,
        )
        condition_inventory_df.to_csv(tables_dir / "observed_session_condition_inventory.csv", index=False)
        method1_skip_df.to_csv(tables_dir / "method1_skipped_cells.csv", index=False)
        observed_df.to_csv(tables_dir / "method1_subject_observed_distances.csv", index=False)

        permutation_df = run_method1_group_permutations(
            blocks_by_session=blocks_by_session,
            distance_metrics=distance_metrics,
            n_permutations=int(args.n_permutations),
            random_seed=int(args.seed),
        )
        permutation_df.to_csv(tables_dir / "method1_group_permutation_null.csv", index=False)

        stats_df = summarize_method1_group_statistics(observed_df=observed_df, permutation_df=permutation_df)
        stats_df.to_csv(tables_dir / "method1_group_statistics.csv", index=False)
        method_stats["method1"] = stats_df

        for metric_name in distance_metrics:
            plot_method1_by_condition(
                observed_df=observed_df,
                stats_df=stats_df,
                distance_metric=metric_name,
                out_path=plots_dir / f"method1_{safe_slug(metric_name)}_by_condition.png",
            )

        manifest_payload = build_manifest_payload(
            args=args,
            roi_labels=roi_labels,
            block_inventory_df=block_inventory_df,
            condition_inventory_df=condition_inventory_df,
            observed_df=observed_df,
            permutation_df=permutation_df,
            stats_df=stats_df,
            distance_metrics=distance_metrics,
        )
        (method1_dir / "analysis_manifest.json").write_text(
            json.dumps(manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved Method 1 outputs to: {method1_dir}", flush=True)

    if "method2" in selected_methods:
        method2_dir = ensure_dir(out_root / METHOD2_DIRNAME / f"baseline_{safe_slug(baseline_mode)}")
        method2_tables_dir = ensure_dir(method2_dir / "tables")
        method2_plots_dir = ensure_dir(method2_dir / "plots")

        method2_observed_df, run_inventory_df, method2_skip_df = collect_observed_method2_outputs(
            blocks_by_session=blocks_by_session,
            distance_metrics=distance_metrics,
            baseline_mode=baseline_mode,
        )
        run_inventory_df.to_csv(method2_tables_dir / "observed_run_condition_inventory.csv", index=False)
        method2_skip_df.to_csv(method2_tables_dir / "method2_skipped_cells.csv", index=False)
        method2_observed_df.to_csv(method2_tables_dir / "method2_subject_observed_excess_distances.csv", index=False)

        method2_permutation_df = run_method2_group_permutations(
            blocks_by_session=blocks_by_session,
            distance_metrics=distance_metrics,
            n_permutations=int(args.n_permutations),
            random_seed=int(args.seed),
            baseline_mode=baseline_mode,
        )
        method2_permutation_df.to_csv(method2_tables_dir / "method2_group_permutation_null.csv", index=False)

        method2_stats_df = summarize_method2_group_statistics(
            observed_df=method2_observed_df,
            permutation_df=method2_permutation_df,
        )
        method2_stats_df.to_csv(method2_tables_dir / "method2_group_statistics.csv", index=False)
        method_stats["method2"] = method2_stats_df

        for metric_name in distance_metrics:
            plot_method2_by_condition(
                observed_df=method2_observed_df,
                stats_df=method2_stats_df,
                distance_metric=metric_name,
                baseline_mode=baseline_mode,
                out_path=method2_plots_dir / f"method2_{safe_slug(metric_name)}_by_condition.png",
            )

        method2_manifest_payload = build_method2_manifest_payload(
            args=args,
            roi_labels=roi_labels,
            run_inventory_df=run_inventory_df,
            observed_df=method2_observed_df,
            permutation_df=method2_permutation_df,
            stats_df=method2_stats_df,
            distance_metrics=distance_metrics,
        )
        (method2_dir / "analysis_manifest.json").write_text(
            json.dumps(method2_manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved Method 2 outputs to: {method2_dir}", flush=True)

    if "method3" in selected_methods:
        method3_dir = ensure_dir(out_root / METHOD3_DIRNAME)
        method3_tables_dir = ensure_dir(method3_dir / "tables")
        method3_plots_dir = ensure_dir(method3_dir / "plots")

        method3_observed_df, method3_subject_inventory_df, method3_skip_df = collect_observed_method3_outputs(
            blocks_by_session=blocks_by_session,
            distance_metrics=distance_metrics,
        )
        method3_subject_inventory_df.to_csv(method3_tables_dir / "method3_subject_inventory.csv", index=False)
        method3_skip_df.to_csv(method3_tables_dir / "method3_skipped_cells.csv", index=False)
        method3_observed_df.to_csv(method3_tables_dir / "method3_subject_normalization_scores.csv", index=False)

        method3_permutation_df = run_method3_group_permutations(
            blocks_by_session=blocks_by_session,
            distance_metrics=distance_metrics,
            n_permutations=int(args.n_permutations),
            random_seed=int(args.seed),
        )
        method3_permutation_df.to_csv(method3_tables_dir / "method3_group_permutation_null.csv", index=False)

        method3_stats_df = summarize_method3_group_statistics(
            observed_df=method3_observed_df,
            permutation_df=method3_permutation_df,
        )
        method3_stats_df.to_csv(method3_tables_dir / "method3_group_statistics.csv", index=False)
        method_stats["method3"] = method3_stats_df

        for metric_name in distance_metrics:
            plot_method3_by_condition(
                observed_df=method3_observed_df,
                stats_df=method3_stats_df,
                distance_metric=metric_name,
                out_path=method3_plots_dir / f"method3_{safe_slug(metric_name)}_by_condition.png",
            )

        method3_manifest_payload = build_method3_manifest_payload(
            args=args,
            roi_labels=roi_labels,
            subject_inventory_df=method3_subject_inventory_df,
            observed_df=method3_observed_df,
            permutation_df=method3_permutation_df,
            stats_df=method3_stats_df,
            distance_metrics=distance_metrics,
        )
        (method3_dir / "analysis_manifest.json").write_text(
            json.dumps(method3_manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved Method 3 outputs to: {method3_dir}", flush=True)

    if "method4" in selected_methods:
        method4_dir = ensure_dir(out_root / METHOD4_DIRNAME)
        method4_tables_dir = ensure_dir(method4_dir / "tables")
        method4_plots_dir = ensure_dir(method4_dir / "plots")

        method4_observed_df, method4_skip_df = collect_observed_method4_outputs(blocks_by_session=blocks_by_session)
        method4_skip_df.to_csv(method4_tables_dir / "method4_skipped_cells.csv", index=False)
        method4_observed_df.to_csv(method4_tables_dir / "method4_subject_crossvalidated_contrasts.csv", index=False)

        method4_null_df = run_method4_sign_flip_null(
            observed_df=method4_observed_df,
            n_permutations=int(args.n_permutations),
            seed=int(args.seed),
        )
        method4_null_df.to_csv(method4_tables_dir / "method4_group_signflip_null.csv", index=False)

        method4_stats_df = summarize_method4_group_statistics(
            observed_df=method4_observed_df,
            null_df=method4_null_df,
        )
        method4_stats_df.to_csv(method4_tables_dir / "method4_group_statistics.csv", index=False)
        method_stats["method4"] = method4_stats_df

        plot_method4_by_condition(
            observed_df=method4_observed_df,
            stats_df=method4_stats_df,
            out_path=method4_plots_dir / "method4_crossvalidated_contrast_by_condition.png",
        )

        method4_manifest_payload = build_method4_manifest_payload(
            args=args,
            roi_labels=roi_labels,
            observed_df=method4_observed_df,
            skip_df=method4_skip_df,
            null_df=method4_null_df,
            stats_df=method4_stats_df,
        )
        (method4_dir / "analysis_manifest.json").write_text(
            json.dumps(method4_manifest_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"Saved Method 4 outputs to: {method4_dir}", flush=True)

    write_report(
        out_root=out_root,
        selected_methods=selected_methods,
        args=args,
        method_stats=method_stats,
    )
    print(f"Saved report to: {out_root / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
