#!/usr/bin/env python3
"""Quantify GVS-condition effects on projection and ROI-network summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection

_HERE = Path(__file__).resolve().parent
_GROUP_ANALYSIS_DIR = _HERE.parent
_REPO_ROOT = _GROUP_ANALYSIS_DIR.parent
if str(_GROUP_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from group_analysis.connectivity_new.roi_edge_connectivity_requested import (  # noqa: E402
    ALWAYS_EXCLUDED_ROI_PATTERNS,
    _load_roi_names,
    _load_selected_ijk,
    _safe_corrcoef_rows,
)
from group_analysis.main.analyze_pairwise_metric_separation import (  # noqa: E402
    _laplacian_spectral_distance,
    _signed_normalized_laplacian_spectrum,
)
from group_analysis.main.split_data_by_gvs_condition import (  # noqa: E402
    _load_gvs_orders,
    _load_manifest_rows,
)


TRIALS_PER_BLOCK = 10
RT_COLUMN_INDEX = 1
MIN_CORR_TRIALS = 3
MIN_CONSECUTIVE_PAIRS = 1
MIN_ROI_VOXELS = 5
SHAM_CONDITION_CODE = "gvs-01"
ACTIVE_CONDITION_CODES = [f"gvs-{index:02d}" for index in range(2, 10)]
ACTIVE_CONDITION_FACTORS = [f"GVS{index}" for index in range(2, 10)]
APPROACH2_CONDITION_FACTORS = ["sham", *ACTIVE_CONDITION_FACTORS]
PLOT_CONDITION_ORDER = ["sham", *[f"GVS{index}" for index in range(2, 10)]]
MEDICATION_ORDER = ["OFF", "ON"]

DEFAULT_GVS_ORDER_PATH = (
    _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "gvs_order_by_subject_session_run.tsv"
)
DEFAULT_MANIFEST_PATH = (
    _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "concat_manifest_group.tsv"
)
DEFAULT_PROJECTION_GLOB = "projection_voxel_foldavg_*.npy"
DEFAULT_BY_GVS_DIR = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data" / "by_gvs"
DEFAULT_PROJECTION_ROOT = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "data"
DEFAULT_SESSION_BETA_DIR = _REPO_ROOT / "results" / "connectivity" / "data"
DEFAULT_SELECTED_VOXELS_PATH = _REPO_ROOT / "results" / "connectivity" / "data" / "selected_voxel_indices.npz"
DEFAULT_ROI_IMG = _REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_rois_fitted.nii.gz"
DEFAULT_ROI_SUMMARY = _REPO_ROOT / "results" / "connectivity" / "atlas figure" / "created_roi_summary.json"
DEFAULT_BEHAVIOR_ROOT = Path("/Data/zahra/behaviour")
DEFAULT_GLM_ROOT = Path("/Data/zahra/results_glm")
DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "connectivity" / "GVS_effects"


@dataclass(frozen=True)
class ModelResult:
    name: str
    engine: str
    formula: str
    n_obs: int
    n_groups: int
    converged: bool | None
    aic: float | None
    bic: float | None
    table: pd.DataFrame
    summary_text: str
    fit_object: Any | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run subject/session/condition-level GVS analyses for the blinded vigour network "
            "and save outputs under results/connectivity/GVS_effects."
        )
    )
    parser.add_argument("--gvs-order-path", type=Path, default=DEFAULT_GVS_ORDER_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--projection-root", type=Path, default=DEFAULT_PROJECTION_ROOT)
    parser.add_argument("--projection-glob", default=DEFAULT_PROJECTION_GLOB)
    parser.add_argument("--by-gvs-dir", type=Path, default=DEFAULT_BY_GVS_DIR)
    parser.add_argument("--session-beta-dir", type=Path, default=DEFAULT_SESSION_BETA_DIR)
    parser.add_argument("--selected-voxels-path", type=Path, default=DEFAULT_SELECTED_VOXELS_PATH)
    parser.add_argument("--roi-img", type=Path, default=DEFAULT_ROI_IMG)
    parser.add_argument("--roi-summary", type=Path, default=DEFAULT_ROI_SUMMARY)
    parser.add_argument("--behavior-root", type=Path, default=DEFAULT_BEHAVIOR_ROOT)
    parser.add_argument("--glm-root", type=Path, default=DEFAULT_GLM_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--trials-per-block", type=int, default=TRIALS_PER_BLOCK)
    parser.add_argument("--min-roi-voxels", type=int, default=MIN_ROI_VOXELS)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def medication_from_session(session: int) -> str:
    return "OFF" if int(session) == 1 else "ON"


def condition_code_from_numeric(value: str | int | float) -> str:
    number = int(round(float(value)))
    return f"gvs-{number:02d}"


def condition_name_from_code(code: str) -> str:
    return f"GVS{int(code.split('-')[-1])}"


def condition_factor_from_code(code: str) -> str:
    if code == SHAM_CONDITION_CODE:
        return "sham"
    return condition_name_from_code(code)


def safe_slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text)).strip("_")


def fisher_z_value(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    clipped = float(np.clip(value, -0.999999, 0.999999))
    return float(np.arctanh(clipped))


def fisher_z_matrix(matrix: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(matrix, dtype=np.float64), -0.999999, 0.999999)
    out = np.arctanh(clipped)
    np.fill_diagonal(out, 0.0)
    out[~np.isfinite(out)] = np.nan
    return out


def _mad(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def adjacent_diff_ratio_sum(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    x_prev = array[:-1]
    x_next = array[1:]
    denom = (x_prev ** 2) + (x_next ** 2)
    valid = np.isfinite(denom) & (~np.isclose(denom, 0.0))
    if not np.any(valid):
        return float("nan")
    numerator = (x_prev - x_next) ** 2
    return float(np.sum(numerator[valid] / denom[valid]))


def correlation_distance(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan")
    x_valid = x[valid]
    y_valid = y[valid]
    if np.isclose(np.std(x_valid), 0.0) or np.isclose(np.std(y_valid), 0.0):
        return float("nan")
    corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
    if not np.isfinite(corr):
        return float("nan")
    return float(1.0 - corr)


def cohen_dz(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 2:
        return float("nan")
    std_value = float(np.std(array, ddof=1))
    if np.isclose(std_value, 0.0):
        return float("nan")
    return float(np.mean(array) / std_value)


def find_projection_path(projection_root: Path, pattern: str) -> Path:
    candidates = sorted(projection_root.glob(pattern))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one projection file matching {pattern!r} under {projection_root}, "
            f"found {len(candidates)}."
        )
    return candidates[0]


def load_behavior_run(behavior_root: Path, subject: str, session: int, run: int) -> np.ndarray:
    subject_digits = int(subject.split("sub-pd")[-1])
    path = behavior_root / f"PSPD{subject_digits:03d}_ses_{int(session)}_run_{int(run)}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing behavior file: {path}")
    values = np.asarray(np.load(path), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError(f"Unexpected behavior shape in {path}: {values.shape}")
    return values


def load_trial_metric_mean(glm_root: Path, subject: str, session: int, run: int) -> float:
    path = glm_root / subject / f"ses-{int(session)}" / "GLMOutputs-mni-std" / f"trial_metric_run{int(run)}.npy"
    if not path.exists():
        return float("nan")
    values = np.asarray(np.load(path), dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def build_trial_metadata(
    manifest_rows: list[dict[str, Any]],
    gvs_orders: dict[tuple[str, int, int], list[str]],
    projection: np.ndarray,
    behavior_root: Path,
    glm_root: Path,
    trials_per_block: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for row in manifest_rows:
        subject = str(row["sub_tag"])
        session = int(row["ses"])
        run = int(row["run"])
        subject_session = f"{subject}_ses-{session}"
        medication = medication_from_session(session)
        keep_mask = np.asarray(row["keep_mask"], dtype=bool)
        offset_start = int(row["offset_start"])
        offset_end = int(row["offset_end"])
        y_values = np.asarray(projection[offset_start:offset_end], dtype=np.float64).ravel()
        order_values = gvs_orders[(subject, session, run)]
        n_source_trials = int(row["n_trials_source"])
        if keep_mask.size != n_source_trials:
            raise ValueError(
                f"Keep-mask/source-trial mismatch for {subject} session {session} run {run}: "
                f"{keep_mask.size} vs {n_source_trials}"
            )

        source_conditions = np.repeat(np.asarray(order_values, dtype=object), int(trials_per_block))
        if source_conditions.size != n_source_trials:
            raise ValueError(
                f"GVS order/source trial mismatch for {subject} session {session} run {run}: "
                f"{source_conditions.size} vs {n_source_trials}"
            )
        source_block_order = np.repeat(
            np.arange(1, len(order_values) + 1, dtype=np.int64),
            int(trials_per_block),
        )
        source_trial_in_block = np.tile(
            np.arange(1, int(trials_per_block) + 1, dtype=np.int64),
            len(order_values),
        )
        condition_codes = np.asarray(
            [condition_code_from_numeric(value) for value in source_conditions[keep_mask]],
            dtype=object,
        )
        block_order = source_block_order[keep_mask]
        trial_in_block = source_trial_in_block[keep_mask]

        behavior = load_behavior_run(behavior_root, subject, session, run)
        if behavior.shape[0] < n_source_trials:
            raise ValueError(
                f"Behavior file too short for {subject} session {session} run {run}: "
                f"{behavior.shape[0]} vs {n_source_trials}"
            )
        behavior = behavior[:n_source_trials, :]
        kept_behavior = behavior[keep_mask, :]
        if kept_behavior.shape[0] != y_values.size:
            raise ValueError(
                f"Behavior/projection mismatch for {subject} session {session} run {run}: "
                f"{kept_behavior.shape[0]} vs {y_values.size}"
            )

        inv_rt = np.asarray(kept_behavior[:, RT_COLUMN_INDEX], dtype=np.float64)
        rt = np.full(inv_rt.shape, np.nan, dtype=np.float64)
        valid_inv_rt = np.isfinite(inv_rt) & (~np.isclose(inv_rt, 0.0))
        rt[valid_inv_rt] = 1.0 / inv_rt[valid_inv_rt]
        run_trial_metric_mean = load_trial_metric_mean(glm_root, subject, session, run)

        for idx in range(y_values.size):
            condition_code = str(condition_codes[idx])
            rows.append(
                {
                    "subject": subject,
                    "subject_session": subject_session,
                    "session": session,
                    "medication": medication,
                    "run": run,
                    "condition_code": condition_code,
                    "condition_name": condition_name_from_code(condition_code),
                    "condition_factor": condition_factor_from_code(condition_code),
                    "is_sham": bool(condition_code == SHAM_CONDITION_CODE),
                    "block_order": int(block_order[idx]),
                    "trial_in_block": int(trial_in_block[idx]),
                    "block_uid": f"{subject_session}_run-{run}_block-{int(block_order[idx])}",
                    "y": float(y_values[idx]),
                    "inv_rt": float(inv_rt[idx]) if np.isfinite(inv_rt[idx]) else float("nan"),
                    "rt": float(rt[idx]) if np.isfinite(rt[idx]) else float("nan"),
                    "run_trial_metric_mean": run_trial_metric_mean,
                    "mean_fd": float("nan"),
                }
            )

    trial_df = pd.DataFrame(rows).sort_values(
        ["subject", "session", "run", "block_order", "trial_in_block"]
    )
    trial_df["condition_factor"] = pd.Categorical(
        trial_df["condition_factor"],
        categories=PLOT_CONDITION_ORDER,
        ordered=True,
    )
    trial_df["medication"] = pd.Categorical(
        trial_df["medication"],
        categories=MEDICATION_ORDER,
        ordered=True,
    )
    return trial_df.reset_index(drop=True)


def summarize_condition_metrics(trial_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = [
        "subject",
        "subject_session",
        "session",
        "medication",
        "condition_code",
        "condition_name",
        "condition_factor",
        "is_sham",
    ]
    for key, group in trial_df.groupby(group_cols, sort=True, observed=True):
        (
            subject,
            subject_session,
            session,
            medication,
            condition_code,
            condition_name,
            condition_factor,
            is_sham,
        ) = key
        group = group.sort_values(["run", "block_order", "trial_in_block"]).reset_index(drop=True)
        y = np.asarray(group["y"], dtype=np.float64)
        y_finite = y[np.isfinite(y)]
        rt_pair_mask = np.isfinite(group["y"].to_numpy(dtype=np.float64)) & np.isfinite(
            group["rt"].to_numpy(dtype=np.float64)
        )
        y_pair = group.loc[rt_pair_mask, "y"].to_numpy(dtype=np.float64)
        rt_pair = group.loc[rt_pair_mask, "rt"].to_numpy(dtype=np.float64)

        y_rt_r = float("nan")
        y_rt_p = float("nan")
        if (
            y_pair.size >= MIN_CORR_TRIALS
            and not np.isclose(np.std(y_pair), 0.0)
            and not np.isclose(np.std(rt_pair), 0.0)
        ):
            corr = stats.pearsonr(y_pair, rt_pair)
            y_rt_r = float(corr.statistic)
            y_rt_p = float(corr.pvalue)

        lag_prev_parts: list[np.ndarray] = []
        lag_next_parts: list[np.ndarray] = []
        error_parts: list[np.ndarray] = []
        adjacent_ratio_terms: list[float] = []
        n_blocks_with_pairs = 0
        for _, block in group.groupby("block_uid", sort=True):
            block_y = block["y"].to_numpy(dtype=np.float64)
            block_y = block_y[np.isfinite(block_y)]
            if block_y.size < 2:
                continue
            n_blocks_with_pairs += 1
            lag_prev_parts.append(block_y[:-1])
            lag_next_parts.append(block_y[1:])
            error_parts.append(block_y[1:] - block_y[:-1])
            adjacent_ratio_terms.append(adjacent_diff_ratio_sum(block_y))

        if lag_prev_parts:
            lag_prev = np.concatenate(lag_prev_parts)
            lag_next = np.concatenate(lag_next_parts)
            lag_errors = np.concatenate(error_parts)
            lag1_corr = float("nan")
            if (
                lag_prev.size >= 2
                and not np.isclose(np.std(lag_prev), 0.0)
                and not np.isclose(np.std(lag_next), 0.0)
            ):
                lag1_corr = float(np.corrcoef(lag_prev, lag_next)[0, 1])
            consecutive_rmse = float(np.sqrt(np.mean(lag_errors ** 2)))
            consecutive_mad = float(_mad(lag_errors))
            n_consecutive_pairs = int(lag_errors.size)
        else:
            lag1_corr = float("nan")
            consecutive_rmse = float("nan")
            consecutive_mad = float("nan")
            n_consecutive_pairs = 0

        rows.append(
            {
                "subject": subject,
                "subject_session": subject_session,
                "session": int(session),
                "medication": medication,
                "condition_code": condition_code,
                "condition_name": condition_name,
                "condition_factor": condition_factor,
                "is_sham": bool(is_sham),
                "n_trials_total": int(group.shape[0]),
                "n_trials_y": int(y_finite.size),
                "n_trials_y_rt": int(y_pair.size),
                "mean_y": float(np.mean(y_finite)) if y_finite.size else float("nan"),
                "std_y": float(np.std(y_finite)) if y_finite.size else float("nan"),
                "mean_rt": float(np.mean(rt_pair)) if rt_pair.size else float("nan"),
                "y_rt_r": y_rt_r,
                "y_rt_p": y_rt_p,
                "y_rt_fisher_z": fisher_z_value(y_rt_r),
                "consecutive_rmse_y": consecutive_rmse,
                "consecutive_mad_y": consecutive_mad,
                "consecutive_lag1_corr_y": lag1_corr,
                "adjacent_diff_ratio_sum_y": (
                    float(np.nansum(adjacent_ratio_terms)) if adjacent_ratio_terms else float("nan")
                ),
                "n_consecutive_pairs_y": n_consecutive_pairs,
                "n_blocks_with_pairs": int(n_blocks_with_pairs),
                "mean_run": float(np.mean(group["run"].to_numpy(dtype=np.float64))),
                "mean_block_order": float(np.mean(group["block_order"].to_numpy(dtype=np.float64))),
                "mean_trial_metric": float(np.mean(group["run_trial_metric_mean"].to_numpy(dtype=np.float64))),
                "mean_fd": float("nan"),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(
        ["subject", "session", "condition_code"]
    )
    summary_df["condition_factor"] = pd.Categorical(
        summary_df["condition_factor"],
        categories=PLOT_CONDITION_ORDER,
        ordered=True,
    )
    summary_df["medication"] = pd.Categorical(
        summary_df["medication"],
        categories=MEDICATION_ORDER,
        ordered=True,
    )

    scale = 1e5
    for column in ("mean_y", "consecutive_rmse_y", "consecutive_mad_y"):
        summary_df[f"{column}_scaled"] = summary_df[column] * scale
    return summary_df.reset_index(drop=True)


def build_primary_delta_table(summary_df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    key_cols = ["subject", "session", "medication"]
    sham_df = (
        summary_df[summary_df["condition_code"] == SHAM_CONDITION_CODE][key_cols + metric_columns]
        .rename(columns={column: f"{column}_sham" for column in metric_columns})
        .reset_index(drop=True)
    )
    active_df = summary_df[summary_df["condition_code"] != SHAM_CONDITION_CODE].copy()
    merged = active_df.merge(sham_df, on=key_cols, how="left")
    for column in metric_columns:
        merged[f"delta_vs_sham_{column}"] = merged[column] - merged[f"{column}_sham"]
    return merged


def _fit_model_frame(
    data: pd.DataFrame,
    response_col: str,
    formula_rhs: str,
    model_name: str,
) -> ModelResult:
    model_df = data.copy()
    model_df = model_df.dropna(subset=[response_col, "subject"]).reset_index(drop=True)
    formula = f"{response_col} ~ {formula_rhs}"
    if model_df.empty:
        empty_table = pd.DataFrame(columns=["term", "coef", "std_err", "z_or_t", "p_value", "ci_low", "ci_high"])
        return ModelResult(
            name=model_name,
            engine="none",
            formula=formula,
            n_obs=0,
            n_groups=0,
            converged=None,
            aic=None,
            bic=None,
            table=empty_table,
            summary_text="No usable rows for model fitting.\n",
            fit_object=None,
        )

    n_groups = int(model_df["subject"].nunique())
    warnings.simplefilter("ignore")
    mixed_result = None
    mixed_error: Exception | None = None
    for method in ("lbfgs", "powell"):
        try:
            mixed_model = smf.mixedlm(formula, data=model_df, groups=model_df["subject"])
            mixed_result = mixed_model.fit(reml=False, method=method, maxiter=400, disp=False)
            break
        except Exception as exc:  # pragma: no cover - model solver failures are data dependent
            mixed_error = exc

    if mixed_result is not None:
        conf = mixed_result.conf_int()
        table = pd.DataFrame(
            {
                "term": mixed_result.params.index,
                "coef": mixed_result.params.to_numpy(dtype=np.float64),
                "std_err": mixed_result.bse.to_numpy(dtype=np.float64),
                "z_or_t": mixed_result.tvalues.to_numpy(dtype=np.float64),
                "p_value": mixed_result.pvalues.to_numpy(dtype=np.float64),
                "ci_low": conf.iloc[:, 0].to_numpy(dtype=np.float64),
                "ci_high": conf.iloc[:, 1].to_numpy(dtype=np.float64),
            }
        )
        return ModelResult(
            name=model_name,
            engine="mixedlm",
            formula=formula,
            n_obs=int(model_df.shape[0]),
            n_groups=n_groups,
            converged=bool(getattr(mixed_result, "converged", True)),
            aic=float(mixed_result.aic) if np.isfinite(mixed_result.aic) else None,
            bic=float(mixed_result.bic) if np.isfinite(mixed_result.bic) else None,
            table=table,
            summary_text=str(mixed_result.summary()),
            fit_object=mixed_result,
        )

    ols_result = smf.ols(formula, data=model_df).fit(
        cov_type="cluster",
        cov_kwds={"groups": model_df["subject"]},
    )
    conf = ols_result.conf_int()
    table = pd.DataFrame(
        {
            "term": ols_result.params.index,
            "coef": ols_result.params.to_numpy(dtype=np.float64),
            "std_err": ols_result.bse.to_numpy(dtype=np.float64),
            "z_or_t": ols_result.tvalues.to_numpy(dtype=np.float64),
            "p_value": ols_result.pvalues.to_numpy(dtype=np.float64),
            "ci_low": conf.iloc[:, 0].to_numpy(dtype=np.float64),
            "ci_high": conf.iloc[:, 1].to_numpy(dtype=np.float64),
        }
    )
    summary_text = str(ols_result.summary())
    if mixed_error is not None:
        summary_text = f"MixedLM failed: {type(mixed_error).__name__}: {mixed_error}\n\n{summary_text}"
    return ModelResult(
        name=model_name,
        engine="ols_cluster_subject",
        formula=formula,
        n_obs=int(model_df.shape[0]),
        n_groups=n_groups,
        converged=None,
        aic=float(ols_result.aic) if np.isfinite(ols_result.aic) else None,
        bic=float(ols_result.bic) if np.isfinite(ols_result.bic) else None,
        table=table,
        summary_text=summary_text,
        fit_object=ols_result,
    )


def _linear_model_contrast(
    fit: Any | None,
    weights: dict[str, float],
) -> dict[str, float]:
    row = {
        "estimate": float("nan"),
        "se": float("nan"),
        "z_value": float("nan"),
        "p_value_two_sided": float("nan"),
    }
    if fit is None:
        return row

    params = getattr(fit, "fe_params", None)
    if params is None:
        params = getattr(fit, "params", None)
    cov_params = getattr(fit, "cov_params", None)
    if params is None or cov_params is None:
        return row

    param_names = list(params.index)
    if not param_names:
        return row

    covariance = cov_params()
    if isinstance(covariance, pd.DataFrame):
        cov_matrix = covariance.loc[param_names, param_names].to_numpy(dtype=np.float64)
    else:
        cov_matrix = np.asarray(covariance, dtype=np.float64)
        cov_matrix = cov_matrix[: len(param_names), : len(param_names)]
    beta = params.to_numpy(dtype=np.float64)

    contrast = np.zeros(len(param_names), dtype=np.float64)
    for idx, name in enumerate(param_names):
        contrast[idx] = float(weights.get(name, 0.0))

    estimate = float(np.dot(contrast, beta))
    variance = max(float(np.dot(contrast, cov_matrix @ contrast)), 0.0)
    se = math.sqrt(variance)
    if se <= 1e-12:
        z_value = float("nan")
        p_value = float("nan")
    else:
        z_value = float(estimate / se)
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))

    row.update(
        {
            "estimate": estimate,
            "se": float(se),
            "z_value": z_value,
            "p_value_two_sided": p_value,
        }
    )
    return row


def build_condition_on_vs_off_contrasts(
    fit: Any | None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    medication_term = 'C(medication, Treatment(reference="OFF"))[T.ON]'
    rows: list[dict[str, Any]] = []

    for condition_factor in PLOT_CONDITION_ORDER:
        weights = {medication_term: 1.0}
        if condition_factor != "sham":
            weights[
                f'{medication_term}:C(condition_factor, Treatment(reference="sham"))[T.{condition_factor}]'
            ] = 1.0
        contrast = _linear_model_contrast(fit, weights)
        rows.append(
            {
                "condition_factor": condition_factor,
                "on_minus_off": float(contrast["estimate"]),
                "se": float(contrast["se"]),
                "z_value": float(contrast["z_value"]),
                "p_value": float(contrast["p_value_two_sided"]),
            }
        )

    contrast_df = pd.DataFrame(rows)
    valid_mask = np.isfinite(contrast_df["p_value"].to_numpy(dtype=np.float64))
    q_values = np.full(contrast_df.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(contrast_df.shape[0], dtype=bool)
    if np.any(valid_mask):
        significant_valid, q_values_valid = fdrcorrection(
            contrast_df.loc[valid_mask, "p_value"].to_numpy(dtype=np.float64)
        )
        q_values[valid_mask] = q_values_valid
        significant[valid_mask] = significant_valid
    contrast_df["q_value_fdr"] = q_values
    contrast_df["significant_fdr"] = significant

    if out_path is not None:
        contrast_df.to_csv(out_path, index=False)
    return contrast_df


def fit_primary_models(
    trial_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    consecutive_rmse_contrasts = pd.DataFrame()
    formula_trial = (
        'C(medication, Treatment(reference="OFF")) '
        '* C(condition_factor, Treatment(reference="sham")) + C(run) + block_order'
    )
    trial_model = _fit_model_frame(
        data=trial_df.rename(columns={"y": "y_raw"}).assign(y_scaled=lambda df: df["y_raw"] * 1e5),
        response_col="y_scaled",
        formula_rhs=formula_trial,
        model_name="trial_level_mean_expression",
    )
    model_dir = ensure_dir(out_dir / "models")
    (model_dir / f"{trial_model.name}_summary.txt").write_text(trial_model.summary_text, encoding="utf-8")
    trial_model.table.to_csv(model_dir / f"{trial_model.name}_coefficients.csv", index=False)
    for row in trial_model.table.to_dict(orient="records"):
        model_rows.append(
            {
                "model_name": trial_model.name,
                "engine": trial_model.engine,
                "formula": trial_model.formula,
                "n_obs": trial_model.n_obs,
                "n_groups": trial_model.n_groups,
                "converged": trial_model.converged,
                "aic": trial_model.aic,
                "bic": trial_model.bic,
                **row,
            }
        )

    condition_models = [
        ("mean_y_scaled", "condition_level_mean_expression"),
        ("consecutive_rmse_y_scaled", "condition_level_consecutive_rmse"),
        ("consecutive_mad_y_scaled", "condition_level_consecutive_mad"),
        ("y_rt_fisher_z", "condition_level_y_rt_fisher_z"),
        ("consecutive_lag1_corr_y", "condition_level_consecutive_lag1_corr"),
    ]
    formula_condition = (
        'C(medication, Treatment(reference="OFF")) '
        '* C(condition_factor, Treatment(reference="sham")) + mean_run + mean_block_order'
    )
    for response_col, model_name in condition_models:
        result = _fit_model_frame(
            data=summary_df,
            response_col=response_col,
            formula_rhs=formula_condition,
            model_name=model_name,
        )
        (model_dir / f"{result.name}_summary.txt").write_text(result.summary_text, encoding="utf-8")
        result.table.to_csv(model_dir / f"{result.name}_coefficients.csv", index=False)
        if result.name == "condition_level_consecutive_rmse":
            consecutive_rmse_contrasts = build_condition_on_vs_off_contrasts(
                fit=result.fit_object,
                out_path=model_dir / f"{result.name}_on_vs_off_contrasts.csv",
            )
        for row in result.table.to_dict(orient="records"):
            model_rows.append(
                {
                    "model_name": result.name,
                    "engine": result.engine,
                    "formula": result.formula,
                    "n_obs": result.n_obs,
                    "n_groups": result.n_groups,
                    "converged": result.converged,
                    "aic": result.aic,
                    "bic": result.bic,
                    **row,
                }
            )
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(out_dir / "primary_model_coefficients.csv", index=False)
    return model_df, consecutive_rmse_contrasts


def run_sham_delta_tests(
    data: pd.DataFrame,
    metric_columns: list[str],
    condition_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for medication in MEDICATION_ORDER:
        medication_df = data[data["medication"] == medication].copy()
        for metric in metric_columns:
            metric_rows: list[dict[str, Any]] = []
            for condition in sorted(
                medication_df[condition_column].dropna().unique().tolist(),
                key=lambda value: (value != "sham", str(value)),
            ):
                if condition == "sham":
                    continue
                values = medication_df.loc[
                    medication_df[condition_column] == condition, metric
                ].to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                if values.size == 0:
                    metric_rows.append(
                        {
                            "medication": medication,
                            "metric": metric,
                            "condition_factor": condition,
                            "n": 0,
                            "mean_delta": np.nan,
                            "std_delta": np.nan,
                            "t_stat": np.nan,
                            "p_value": np.nan,
                            "cohen_dz": np.nan,
                        }
                    )
                    continue
                if values.size >= 2:
                    t_res = stats.ttest_1samp(values, popmean=0.0, nan_policy="omit")
                    t_stat = float(t_res.statistic)
                    p_value = float(t_res.pvalue)
                else:
                    t_stat = float("nan")
                    p_value = float("nan")
                metric_rows.append(
                    {
                        "medication": medication,
                        "metric": metric,
                        "condition_factor": condition,
                        "n": int(values.size),
                        "mean_delta": float(np.mean(values)),
                        "std_delta": float(np.std(values, ddof=1)) if values.size >= 2 else float("nan"),
                        "t_stat": t_stat,
                        "p_value": p_value,
                        "cohen_dz": cohen_dz(values),
                    }
                )
            metric_df = pd.DataFrame(metric_rows)
            finite_mask = np.isfinite(metric_df["p_value"].to_numpy(dtype=np.float64))
            q_values = np.full(metric_df.shape[0], np.nan, dtype=np.float64)
            sig = np.zeros(metric_df.shape[0], dtype=bool)
            if np.any(finite_mask):
                sig_valid, q_valid = fdrcorrection(metric_df.loc[finite_mask, "p_value"], alpha=0.05)
                q_values[finite_mask] = q_valid
                sig[finite_mask] = sig_valid
            metric_df["q_value_fdr"] = q_values
            metric_df["significant_fdr"] = sig
            rows.extend(metric_df.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_roi_membership(
    roi_img_path: Path,
    roi_summary_path: Path,
    selected_voxels_path: Path,
    min_roi_voxels: int,
) -> tuple[list[np.ndarray], list[str], pd.DataFrame]:
    roi_img = nib.load(str(roi_img_path))
    roi_data = np.asarray(roi_img.get_fdata(), dtype=np.int32)
    selected_ijk = _load_selected_ijk(selected_voxels_path, roi_data.shape)
    x, y, z = selected_ijk.T
    roi_ids = roi_data[x, y, z]
    roi_names = _load_roi_names(roi_summary_path)
    coords_mm = nib.affines.apply_affine(roi_img.affine, selected_ijk)
    unique_ids, counts = np.unique(roi_ids[roi_ids > 0], return_counts=True)
    exclude_patterns = [pattern.lower() for pattern in ALWAYS_EXCLUDED_ROI_PATTERNS]

    members: list[np.ndarray] = []
    labels: list[str] = []
    meta_rows: list[dict[str, Any]] = []
    for roi_id, count in zip(unique_ids.tolist(), counts.tolist()):
        if int(count) < int(max(1, min_roi_voxels)):
            continue
        label = roi_names.get(int(roi_id), f"ROI_{int(roi_id)}")
        if any(pattern in label.lower() for pattern in exclude_patterns):
            continue
        member_idx = np.flatnonzero(roi_ids == int(roi_id)).astype(np.int64, copy=False)
        if member_idx.size < int(max(1, min_roi_voxels)):
            continue
        members.append(member_idx)
        labels.append(label)
        centroid = np.mean(coords_mm[member_idx], axis=0)
        meta_rows.append(
            {
                "roi_id": int(roi_id),
                "roi_name": label,
                "n_selected_voxels": int(member_idx.size),
                "x_mm": float(centroid[0]),
                "y_mm": float(centroid[1]),
                "z_mm": float(centroid[2]),
            }
        )
    meta_df = pd.DataFrame(meta_rows).sort_values("roi_name").reset_index(drop=True)
    return members, labels, meta_df


def compute_roi_fc_payload(
    beta: np.ndarray,
    roi_members: list[np.ndarray],
) -> dict[str, np.ndarray]:
    roi_ts = np.full((len(roi_members), beta.shape[1]), np.nan, dtype=np.float64)
    for idx, members in enumerate(roi_members):
        roi_ts[idx] = np.nanmean(beta[members, :], axis=0)
    fc_r = _safe_corrcoef_rows(roi_ts)
    fc_r = np.clip(fc_r, -1.0, 1.0, out=fc_r)
    np.fill_diagonal(fc_r, 1.0)
    fc_z = fisher_z_matrix(fc_r)
    iu = np.triu_indices(len(roi_members), k=1)
    vec_r = fc_r[iu]
    vec_z = fc_z[iu]
    spectrum = _signed_normalized_laplacian_spectrum(fc_z)
    return {
        "roi_ts": roi_ts,
        "r": fc_r,
        "z": fc_z,
        "vec_r": vec_r,
        "vec_z": vec_z,
        "spectrum": spectrum,
    }


def compute_fc_outputs(
    by_gvs_dir: Path,
    roi_members: list[np.ndarray],
    roi_labels: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, int, str], dict[str, np.ndarray]]]:
    matrices: dict[tuple[str, int, str], dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []

    matrix_dir = ensure_dir(out_dir / "matrices")
    for subject_dir in sorted(path for path in by_gvs_dir.iterdir() if path.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            session = int(session_dir.name.split("-")[-1])
            medication = medication_from_session(session)
            out_session_dir = ensure_dir(matrix_dir / subject / session_dir.name)
            for beta_path in sorted(session_dir.glob("selected_beta_trials_gvs-*.npy")):
                condition_code = beta_path.stem.replace("selected_beta_trials_", "")
                beta = np.asarray(np.load(beta_path), dtype=np.float64)
                if beta.ndim != 2:
                    raise ValueError(f"Expected 2D beta matrix in {beta_path}, got {beta.shape}")
                payload = compute_roi_fc_payload(beta=beta, roi_members=roi_members)
                roi_ts = payload["roi_ts"]
                fc_r = payload["r"]
                fc_z = payload["z"]

                prefix = out_session_dir / condition_code
                np.save(prefix.with_name(f"{condition_code}_roi_timeseries.npy"), roi_ts.astype(np.float32, copy=False))
                np.save(prefix.with_name(f"{condition_code}_fc_r.npy"), fc_r.astype(np.float32, copy=False))
                np.save(prefix.with_name(f"{condition_code}_fc_z.npy"), fc_z.astype(np.float32, copy=False))

                vec_r = payload["vec_r"]
                vec_z = payload["vec_z"]
                matrices[(subject, session, condition_code)] = {
                    "r": fc_r,
                    "z": fc_z,
                    "vec_z": vec_z,
                    "spectrum": payload["spectrum"],
                }
                rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "medication": medication,
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "n_trials_fc": int(beta.shape[1]),
                        "n_rois": int(len(roi_members)),
                        "n_edges": int(vec_r.size),
                        "mean_fc_r": float(np.nanmean(vec_r)) if np.any(np.isfinite(vec_r)) else float("nan"),
                        "mean_abs_fc_r": float(np.nanmean(np.abs(vec_r))) if np.any(np.isfinite(vec_r)) else float("nan"),
                        "mean_fc_z": float(np.nanmean(vec_z)) if np.any(np.isfinite(vec_z)) else float("nan"),
                    }
                )

    fc_df = pd.DataFrame(rows).sort_values(["subject", "session", "condition_code"]).reset_index(drop=True)
    fc_df["condition_factor"] = pd.Categorical(fc_df["condition_factor"], categories=PLOT_CONDITION_ORDER, ordered=True)
    fc_df["medication"] = pd.Categorical(fc_df["medication"], categories=MEDICATION_ORDER, ordered=True)
    return fc_df, matrices


def _gather_run_condition_columns(
    session_rows: list[dict[str, object]],
    n_session_trials: int,
) -> dict[tuple[int, str], np.ndarray]:
    columns_by_run_condition: dict[tuple[int, str], np.ndarray] = {}
    session_cursor = 0
    coverage = np.zeros(n_session_trials, dtype=np.int16)

    for row in sorted(session_rows, key=lambda item: int(item["run"])):
        run = int(row["run"])
        run_trials = int(row["n_trials_kept"])
        kept_condition_labels = np.asarray(row["kept_condition_labels"], dtype=object)
        if kept_condition_labels.size != run_trials:
            raise ValueError(
                f"{row['sub_tag']} ses-{row['ses']} run-{run}: condition-label count "
                f"{kept_condition_labels.size} does not match kept trials {run_trials}."
            )
        run_columns = np.arange(session_cursor, session_cursor + run_trials, dtype=np.int64)
        if run_columns.size and int(run_columns[-1]) >= n_session_trials:
            raise ValueError(
                f"{row['sub_tag']} ses-{row['ses']} run-{run}: run columns exceed session-trial width {n_session_trials}."
            )
        for condition_value in np.unique(kept_condition_labels):
            condition_code = condition_code_from_numeric(str(condition_value))
            condition_mask = kept_condition_labels == condition_value
            selected_columns = run_columns[condition_mask]
            if selected_columns.size == 0:
                continue
            coverage[selected_columns] += 1
            columns_by_run_condition[(run, condition_code)] = selected_columns
        session_cursor += run_trials

    if session_cursor != n_session_trials:
        raise ValueError(
            f"Session trial mismatch: manifest assigns {session_cursor} beta trials, but session file has {n_session_trials}."
        )
    if np.any(coverage != 1):
        bad_count = int(np.count_nonzero(coverage != 1))
        raise ValueError(f"Session beta trials are not covered exactly once across run-condition cells ({bad_count} bad columns).")
    return columns_by_run_condition


def compute_run_level_fc_outputs(
    session_beta_dir: Path,
    manifest_rows: list[dict[str, object]],
    roi_members: list[np.ndarray],
    roi_labels: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, int, int, str], dict[str, np.ndarray]]]:
    session_rows_map: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        session_rows_map[(str(row["sub_tag"]), int(row["ses"]))].append(row)

    run_level_dir = ensure_dir(out_dir / "run_level")
    matrix_dir = ensure_dir(run_level_dir / "matrices")
    rows: list[dict[str, Any]] = []
    matrices: dict[tuple[str, int, int, str], dict[str, np.ndarray]] = {}

    for (subject, session), session_rows in sorted(session_rows_map.items()):
        beta_path = session_beta_dir / f"selected_beta_trials_{subject}_ses-{session}.npy"
        if not beta_path.exists():
            raise FileNotFoundError(f"Missing session beta file for run-level FC reconstruction: {beta_path}")
        beta_trials = np.asarray(np.load(beta_path), dtype=np.float64)
        if beta_trials.ndim != 2:
            raise ValueError(f"Expected 2D beta-trial matrix in {beta_path}, got {beta_trials.shape}")

        columns_by_run_condition = _gather_run_condition_columns(
            session_rows=session_rows,
            n_session_trials=int(beta_trials.shape[1]),
        )
        medication = medication_from_session(session)
        for (run, condition_code), columns in sorted(columns_by_run_condition.items()):
            subset = np.asarray(beta_trials[:, columns], dtype=np.float64)
            payload = compute_roi_fc_payload(beta=subset, roi_members=roi_members)
            out_run_dir = ensure_dir(matrix_dir / subject / f"ses-{session}" / f"run-{run}")
            prefix = out_run_dir / condition_code
            np.save(prefix.with_name(f"{condition_code}_roi_timeseries.npy"), payload["roi_ts"].astype(np.float32, copy=False))
            np.save(prefix.with_name(f"{condition_code}_fc_r.npy"), payload["r"].astype(np.float32, copy=False))
            np.save(prefix.with_name(f"{condition_code}_fc_z.npy"), payload["z"].astype(np.float32, copy=False))

            matrices[(subject, int(session), int(run), condition_code)] = {
                "r": payload["r"],
                "z": payload["z"],
                "vec_z": payload["vec_z"],
                "spectrum": payload["spectrum"],
            }
            rows.append(
                {
                    "subject": subject,
                    "session": int(session),
                    "run": int(run),
                    "medication": medication,
                    "condition_code": condition_code,
                    "condition_name": condition_name_from_code(condition_code),
                    "condition_factor": condition_factor_from_code(condition_code),
                    "n_trials_fc": int(subset.shape[1]),
                    "n_rois": int(len(roi_labels)),
                    "n_edges": int(payload["vec_r"].size),
                    "mean_fc_r": float(np.nanmean(payload["vec_r"])) if np.any(np.isfinite(payload["vec_r"])) else float("nan"),
                    "mean_abs_fc_r": (
                        float(np.nanmean(np.abs(payload["vec_r"]))) if np.any(np.isfinite(payload["vec_r"])) else float("nan")
                    ),
                    "mean_fc_z": float(np.nanmean(payload["vec_z"])) if np.any(np.isfinite(payload["vec_z"])) else float("nan"),
                }
            )

    fc_df = pd.DataFrame(rows).sort_values(["subject", "session", "run", "condition_code"]).reset_index(drop=True)
    fc_df["condition_factor"] = pd.Categorical(fc_df["condition_factor"], categories=PLOT_CONDITION_ORDER, ordered=True)
    fc_df["medication"] = pd.Categorical(fc_df["medication"], categories=MEDICATION_ORDER, ordered=True)
    return fc_df, matrices


def summarize_fc_deltas(
    matrices: dict[tuple[str, int, str], dict[str, np.ndarray]],
    roi_labels: list[str],
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    iu = np.triu_indices(len(roi_labels), k=1)
    rows: list[dict[str, Any]] = []
    edge_delta_by_contrast: dict[tuple[str, str], list[np.ndarray]] = {}
    mean_delta_by_contrast: dict[tuple[str, str], np.ndarray] = {}

    for (subject, session, condition_code), payload in sorted(matrices.items()):
        if condition_code == SHAM_CONDITION_CODE:
            continue
        sham_key = (subject, session, SHAM_CONDITION_CODE)
        if sham_key not in matrices:
            continue
        sham_payload = matrices[sham_key]
        medication = medication_from_session(session)
        condition_factor = condition_factor_from_code(condition_code)
        corr_dist = correlation_distance(payload["vec_z"], sham_payload["vec_z"])
        fro_value = float(np.linalg.norm(np.nan_to_num(payload["z"] - sham_payload["z"]), ord="fro"))
        spectrum_active = payload["spectrum"]
        spectrum_sham = sham_payload["spectrum"]
        lap_value = _laplacian_spectral_distance(spectrum_active, spectrum_sham)
        delta_vec = np.asarray(payload["vec_z"] - sham_payload["vec_z"], dtype=np.float64)
        edge_delta_by_contrast.setdefault((medication, condition_factor), []).append(delta_vec)
        rows.append(
            {
                "subject": subject,
                "session": int(session),
                "medication": medication,
                "condition_code": condition_code,
                "condition_name": condition_name_from_code(condition_code),
                "condition_factor": condition_factor,
                "correlation_distance_vs_sham": corr_dist,
                "frobenius_norm_vs_sham": fro_value,
                "laplacian_spectral_distance_vs_sham": lap_value,
            }
        )

    edge_level_dir = ensure_dir(out_dir / "edge_level")
    mean_delta_heatmaps_dir = ensure_dir(edge_level_dir / "mean_delta_heatmaps")
    summary_rows: list[dict[str, Any]] = []
    for (medication, condition_factor), matrices_list in sorted(edge_delta_by_contrast.items()):
        delta_stack = np.vstack(matrices_list).astype(np.float64, copy=False)
        mean_delta = np.nanmean(delta_stack, axis=0)
        mean_delta_by_contrast[(medication, condition_factor)] = mean_delta
        t_res = stats.ttest_1samp(delta_stack, popmean=0.0, axis=0, nan_policy="omit")
        p_values = np.asarray(t_res.pvalue, dtype=np.float64)
        t_values = np.asarray(t_res.statistic, dtype=np.float64)
        q_values = np.full(p_values.shape, np.nan, dtype=np.float64)
        significant = np.zeros(p_values.shape, dtype=bool)
        finite_mask = np.isfinite(p_values)
        if np.any(finite_mask):
            significant_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
            q_values[finite_mask] = q_valid
            significant[finite_mask] = significant_valid

        sd_delta = np.nanstd(delta_stack, axis=0, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            dz = np.nanmean(delta_stack, axis=0) / sd_delta
        edge_df = pd.DataFrame(
            {
                "edge_id": np.arange(mean_delta.size, dtype=int),
                "roi_i": [roi_labels[i] for i in iu[0]],
                "roi_j": [roi_labels[j] for j in iu[1]],
                "mean_delta_z_vs_sham": mean_delta,
                "std_delta_z_vs_sham": sd_delta,
                "t_stat": t_values,
                "p_value": p_values,
                "q_value_fdr": q_values,
                "cohen_dz": dz,
                "significant_fdr": significant,
                "n_pairs": int(delta_stack.shape[0]),
            }
        ).sort_values(["q_value_fdr", "p_value", "mean_delta_z_vs_sham"], ascending=[True, True, False])
        edge_path = edge_level_dir / f"{safe_slug(medication)}__{safe_slug(condition_factor)}__edge_tests.csv"
        edge_df.to_csv(edge_path, index=False)

        matrix = np.zeros((len(roi_labels), len(roi_labels)), dtype=np.float64)
        matrix[iu] = mean_delta
        matrix[(iu[1], iu[0])] = mean_delta
        vmax = float(np.nanmax(np.abs(mean_delta))) if np.any(np.isfinite(mean_delta)) else 0.1
        vmax = max(vmax, 1e-6)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Mean FC delta vs sham\n{medication} | {condition_factor}")
        ax.set_xticks(np.arange(len(roi_labels)))
        ax.set_yticks(np.arange(len(roi_labels)))
        ax.set_xticklabels(roi_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(roi_labels, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Mean delta Fisher z")
        fig.tight_layout()
        fig.savefig(
            mean_delta_heatmaps_dir / f"{safe_slug(medication)}__{safe_slug(condition_factor)}__mean_delta_heatmap.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)

        top_edge = edge_df.iloc[0] if not edge_df.empty else None
        summary_rows.append(
            {
                "medication": medication,
                "condition_factor": condition_factor,
                "n_subject_session_pairs": int(delta_stack.shape[0]),
                "n_significant_edges_fdr": int(np.count_nonzero(significant)),
                "min_q_value_fdr": float(np.nanmin(q_values)) if np.any(np.isfinite(q_values)) else float("nan"),
                "top_edge": (
                    f"{top_edge['roi_i']} -- {top_edge['roi_j']}" if top_edge is not None else ""
                ),
                "top_edge_mean_delta_z": (
                    float(top_edge["mean_delta_z_vs_sham"]) if top_edge is not None else float("nan")
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["medication", "condition_factor"]
    )
    summary_df.to_csv(edge_level_dir / "edge_level_summary.csv", index=False)

    network_df = pd.DataFrame(rows).sort_values(
        ["medication", "condition_factor", "subject", "session"]
    ).reset_index(drop=True)
    return network_df, mean_delta_by_contrast


def compute_distance_metric(metric_name: str, a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float:
    if metric_name == "correlation_distance":
        return correlation_distance(a["vec_z"], b["vec_z"])
    if metric_name == "frobenius_norm":
        return float(np.linalg.norm(np.nan_to_num(a["z"] - b["z"]), ord="fro"))
    if metric_name == "laplacian_spectral_distance":
        return _laplacian_spectral_distance(a["spectrum"], b["spectrum"])
    raise KeyError(f"Unsupported distance metric: {metric_name}")


def compute_ratio_score(numerator_terms: list[float], denominator_terms: list[float]) -> float:
    numerator_array = np.asarray(numerator_terms, dtype=np.float64)
    denominator_array = np.asarray(denominator_terms, dtype=np.float64)
    numerator_array = numerator_array[np.isfinite(numerator_array)]
    denominator_array = np.abs(denominator_array[np.isfinite(denominator_array)])
    if numerator_array.size == 0 or denominator_array.size == 0:
        return float("nan")
    # Compare mean between-condition distance against mean within-condition distance
    # so that a null case with matched distances is centered at ratio_score == 1.
    numerator = float(np.mean(numerator_array))
    denominator = float(np.mean(denominator_array))
    if not np.isfinite(numerator) or not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return float("nan")
    return float(numerator / denominator)


def summarize_runmatched_fc_scores(
    matrices: dict[tuple[str, int, int, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    metric_names = ["correlation_distance", "frobenius_norm", "laplacian_spectral_distance"]
    rows: list[dict[str, Any]] = []
    session_keys = sorted({(subject, session) for subject, session, _, _ in matrices})
    for subject, session in session_keys:
        medication = medication_from_session(session)
        for condition_code in ACTIVE_CONDITION_CODES:
            required_keys = [
                (subject, session, 1, SHAM_CONDITION_CODE),
                (subject, session, 2, SHAM_CONDITION_CODE),
                (subject, session, 1, condition_code),
                (subject, session, 2, condition_code),
            ]
            if any(key not in matrices for key in required_keys):
                continue
            sham_run1 = matrices[(subject, session, 1, SHAM_CONDITION_CODE)]
            sham_run2 = matrices[(subject, session, 2, SHAM_CONDITION_CODE)]
            gvs_run1 = matrices[(subject, session, 1, condition_code)]
            gvs_run2 = matrices[(subject, session, 2, condition_code)]
            for metric_name in metric_names:
                distance_gvs1_sham1 = compute_distance_metric(metric_name, gvs_run1, sham_run1)
                distance_gvs2_sham2 = compute_distance_metric(metric_name, gvs_run2, sham_run2)
                distance_sham1_sham2 = compute_distance_metric(metric_name, sham_run1, sham_run2)
                distance_gvs1_gvs2 = compute_distance_metric(metric_name, gvs_run1, gvs_run2)
                score = compute_ratio_score(
                    numerator_terms=[distance_gvs1_sham1, distance_gvs2_sham2],
                    denominator_terms=[distance_sham1_sham2, distance_gvs1_gvs2],
                )
                rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "medication": medication,
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "distance_metric": metric_name,
                        "distance_run1_target_vs_sham": distance_gvs1_sham1,
                        "distance_run2_target_vs_sham": distance_gvs2_sham2,
                        "distance_sham_run1_vs_run2": distance_sham1_sham2,
                        "distance_target_run1_vs_run2": distance_gvs1_gvs2,
                        "ratio_score": score,
                        "log_ratio_score": float(np.log(score)) if np.isfinite(score) and score > 0.0 else float("nan"),
                    }
                )
    df = pd.DataFrame(rows).sort_values(
        ["distance_metric", "medication", "condition_factor", "subject", "session"]
    ).reset_index(drop=True)
    if not df.empty:
        df["condition_factor"] = pd.Categorical(df["condition_factor"], categories=ACTIVE_CONDITION_FACTORS, ordered=True)
        df["medication"] = pd.Categorical(df["medication"], categories=MEDICATION_ORDER, ordered=True)
    return df


def summarize_off_to_on_sham_scores(
    matrices: dict[tuple[str, int, int, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    metric_names = ["correlation_distance", "frobenius_norm", "laplacian_spectral_distance"]
    rows: list[dict[str, Any]] = []
    subject_keys = sorted({subject for subject, _, _, _ in matrices})
    for subject in subject_keys:
        for condition_code in ACTIVE_CONDITION_CODES:
            required_keys = [
                (subject, 1, 1, SHAM_CONDITION_CODE),
                (subject, 1, 2, SHAM_CONDITION_CODE),
                (subject, 2, 1, SHAM_CONDITION_CODE),
                (subject, 2, 2, SHAM_CONDITION_CODE),
                (subject, 1, 1, condition_code),
                (subject, 1, 2, condition_code),
            ]
            if any(key not in matrices for key in required_keys):
                continue
            sham_off_run1 = matrices[(subject, 1, 1, SHAM_CONDITION_CODE)]
            sham_off_run2 = matrices[(subject, 1, 2, SHAM_CONDITION_CODE)]
            sham_on_run1 = matrices[(subject, 2, 1, SHAM_CONDITION_CODE)]
            sham_on_run2 = matrices[(subject, 2, 2, SHAM_CONDITION_CODE)]
            gvs_off_run1 = matrices[(subject, 1, 1, condition_code)]
            gvs_off_run2 = matrices[(subject, 1, 2, condition_code)]
            for metric_name in metric_names:
                distance_gvs1_sham_on1 = compute_distance_metric(metric_name, gvs_off_run1, sham_on_run1)
                distance_gvs2_sham_on2 = compute_distance_metric(metric_name, gvs_off_run2, sham_on_run2)
                distance_sham_off = compute_distance_metric(metric_name, sham_off_run1, sham_off_run2)
                distance_sham_on = compute_distance_metric(metric_name, sham_on_run1, sham_on_run2)
                distance_gvs_off = compute_distance_metric(metric_name, gvs_off_run1, gvs_off_run2)
                score = compute_ratio_score(
                    numerator_terms=[distance_gvs1_sham_on1, distance_gvs2_sham_on2],
                    denominator_terms=[distance_sham_off, distance_sham_on, distance_gvs_off],
                )
                rows.append(
                    {
                        "subject": subject,
                        "source_session_off": 1,
                        "reference_session_on": 2,
                        "condition_code": condition_code,
                        "condition_name": condition_name_from_code(condition_code),
                        "condition_factor": condition_factor_from_code(condition_code),
                        "distance_metric": metric_name,
                        "distance_run1_off_target_vs_on_sham": distance_gvs1_sham_on1,
                        "distance_run2_off_target_vs_on_sham": distance_gvs2_sham_on2,
                        "distance_sham_off_run1_vs_run2": distance_sham_off,
                        "distance_sham_on_run1_vs_run2": distance_sham_on,
                        "distance_target_off_run1_vs_run2": distance_gvs_off,
                        "ratio_score": score,
                        "log_ratio_score": float(np.log(score)) if np.isfinite(score) and score > 0.0 else float("nan"),
                    }
                )
        sham_required_keys = [
            (subject, 1, 1, SHAM_CONDITION_CODE),
            (subject, 1, 2, SHAM_CONDITION_CODE),
            (subject, 2, 1, SHAM_CONDITION_CODE),
            (subject, 2, 2, SHAM_CONDITION_CODE),
        ]
        if any(key not in matrices for key in sham_required_keys):
            continue
        sham_off_run1 = matrices[(subject, 1, 1, SHAM_CONDITION_CODE)]
        sham_off_run2 = matrices[(subject, 1, 2, SHAM_CONDITION_CODE)]
        sham_on_run1 = matrices[(subject, 2, 1, SHAM_CONDITION_CODE)]
        sham_on_run2 = matrices[(subject, 2, 2, SHAM_CONDITION_CODE)]
        for metric_name in metric_names:
            distance_sham_off1_vs_on1 = compute_distance_metric(metric_name, sham_off_run1, sham_on_run1)
            distance_sham_off2_vs_on2 = compute_distance_metric(metric_name, sham_off_run2, sham_on_run2)
            distance_sham_off = compute_distance_metric(metric_name, sham_off_run1, sham_off_run2)
            distance_sham_on = compute_distance_metric(metric_name, sham_on_run1, sham_on_run2)
            score = compute_ratio_score(
                numerator_terms=[distance_sham_off1_vs_on1, distance_sham_off2_vs_on2],
                denominator_terms=[distance_sham_off, distance_sham_on],
            )
            rows.append(
                {
                    "subject": subject,
                    "source_session_off": 1,
                    "reference_session_on": 2,
                    "condition_code": SHAM_CONDITION_CODE,
                    "condition_name": condition_name_from_code(SHAM_CONDITION_CODE),
                    "condition_factor": "sham",
                    "distance_metric": metric_name,
                    "distance_run1_off_target_vs_on_sham": distance_sham_off1_vs_on1,
                    "distance_run2_off_target_vs_on_sham": distance_sham_off2_vs_on2,
                    "distance_sham_off_run1_vs_run2": distance_sham_off,
                    "distance_sham_on_run1_vs_run2": distance_sham_on,
                    "distance_target_off_run1_vs_run2": distance_sham_off,
                    "ratio_score": score,
                    "log_ratio_score": float(np.log(score)) if np.isfinite(score) and score > 0.0 else float("nan"),
                }
            )
    df = pd.DataFrame(rows).sort_values(["distance_metric", "condition_factor", "subject"]).reset_index(drop=True)
    if not df.empty:
        df["condition_factor"] = pd.Categorical(df["condition_factor"], categories=APPROACH2_CONDITION_FACTORS, ordered=True)
    return df


def summarize_ratio_score_stats(
    data: pd.DataFrame,
    strata_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [*strata_columns, "distance_metric", "condition_factor"]
    if data.empty:
        return pd.DataFrame(
            columns=[
                *strata_columns,
                "distance_metric",
                "condition_factor",
                "n",
                "mean_ratio_score",
                "median_ratio_score",
                "geometric_mean_ratio_score",
                "mean_log_ratio_score",
                "std_log_ratio_score",
                "t_stat_log_ratio_vs_zero",
                "p_value_ttest",
                "wilcoxon_stat_log_ratio_vs_zero",
                "p_value_wilcoxon",
                "cohen_dz_log_ratio",
                "n_ratio_gt_1",
                "fraction_ratio_gt_1",
                "q_value_fdr",
                "significant_fdr",
            ]
        )

    for keys, group_df in data.groupby(group_columns, sort=True, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {column: value for column, value in zip(group_columns, keys)}
        ratio_values = group_df["ratio_score"].to_numpy(dtype=np.float64)
        ratio_values = ratio_values[np.isfinite(ratio_values) & (ratio_values > 0.0)]
        log_values = np.log(ratio_values)
        if log_values.size >= 2:
            t_res = stats.ttest_1samp(log_values, popmean=0.0, nan_policy="omit")
            t_stat = float(t_res.statistic)
            p_t = float(t_res.pvalue)
            try:
                wilcoxon_res = stats.wilcoxon(log_values, alternative="two-sided", zero_method="wilcox")
                wilcoxon_stat = float(wilcoxon_res.statistic)
                p_wilcoxon = float(wilcoxon_res.pvalue)
            except ValueError:
                wilcoxon_stat = float("nan")
                p_wilcoxon = float("nan")
            std_log = float(np.std(log_values, ddof=1))
        elif log_values.size == 1:
            t_stat = float("nan")
            p_t = float("nan")
            wilcoxon_stat = float("nan")
            p_wilcoxon = float("nan")
            std_log = float("nan")
        else:
            t_stat = float("nan")
            p_t = float("nan")
            wilcoxon_stat = float("nan")
            p_wilcoxon = float("nan")
            std_log = float("nan")

        rows.append(
            {
                **record,
                "n": int(log_values.size),
                "mean_ratio_score": float(np.mean(ratio_values)) if ratio_values.size else float("nan"),
                "median_ratio_score": float(np.median(ratio_values)) if ratio_values.size else float("nan"),
                "geometric_mean_ratio_score": float(np.exp(np.mean(log_values))) if log_values.size else float("nan"),
                "mean_log_ratio_score": float(np.mean(log_values)) if log_values.size else float("nan"),
                "std_log_ratio_score": std_log,
                "t_stat_log_ratio_vs_zero": t_stat,
                "p_value_ttest": p_t,
                "wilcoxon_stat_log_ratio_vs_zero": wilcoxon_stat,
                "p_value_wilcoxon": p_wilcoxon,
                "cohen_dz_log_ratio": cohen_dz(log_values),
                "n_ratio_gt_1": int(np.count_nonzero(ratio_values > 1.0)),
                "fraction_ratio_gt_1": float(np.mean(ratio_values > 1.0)) if ratio_values.size else float("nan"),
            }
        )

    stats_df = pd.DataFrame(rows)
    fdr_group_columns = [*strata_columns, "distance_metric"]
    q_values = np.full(stats_df.shape[0], np.nan, dtype=np.float64)
    significant = np.zeros(stats_df.shape[0], dtype=bool)
    for _, idx in stats_df.groupby(fdr_group_columns, sort=True, observed=True).groups.items():
        idx_array = np.asarray(list(idx), dtype=np.int64)
        p_values = stats_df.loc[idx_array, "p_value_wilcoxon"].to_numpy(dtype=np.float64)
        finite_mask = np.isfinite(p_values)
        if not np.any(finite_mask):
            continue
        sig_valid, q_valid = fdrcorrection(p_values[finite_mask], alpha=0.05)
        q_values[idx_array[finite_mask]] = q_valid
        significant[idx_array[finite_mask]] = sig_valid
    stats_df["q_value_fdr"] = q_values
    stats_df["significant_fdr"] = significant
    return stats_df.sort_values([*fdr_group_columns, "condition_factor"]).reset_index(drop=True)


def plot_metric_by_condition(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_path: Path,
    significance_df: pd.DataFrame | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(PLOT_CONDITION_ORDER), dtype=np.float64)
    offsets = {"OFF": -0.12, "ON": 0.12}
    colors = {"OFF": "#4c78a8", "ON": "#f58518"}
    mean_by_medication: dict[str, np.ndarray] = {}
    sem_by_medication: dict[str, np.ndarray] = {}

    for medication in MEDICATION_ORDER:
        med_df = df[df["medication"] == medication]
        means = []
        sems = []
        for condition in PLOT_CONDITION_ORDER:
            values = med_df.loc[med_df["condition_factor"] == condition, metric].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else np.nan)
            sems.append(float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size >= 2 else np.nan)
        means_arr = np.asarray(means, dtype=np.float64)
        sems_arr = np.asarray(sems, dtype=np.float64)
        mean_by_medication[medication] = means_arr
        sem_by_medication[medication] = sems_arr
        ax.errorbar(
            x_positions + offsets[medication],
            means_arr,
            yerr=sems_arr,
            fmt="o-",
            color=colors[medication],
            linewidth=1.5,
            markersize=5,
            capsize=3,
            label=medication,
        )

    if significance_df is not None and not significance_df.empty:
        sig_df = significance_df.copy()
        sig_df = sig_df.loc[
            sig_df["significant_fdr"].fillna(False)
            & sig_df["condition_factor"].isin(PLOT_CONDITION_ORDER)
            & (sig_df["condition_factor"] != "sham")
        ].copy()
        if not sig_df.empty:
            data_extent: list[float] = []
            for medication in MEDICATION_ORDER:
                means_arr = mean_by_medication[medication]
                sems_arr = sem_by_medication[medication]
                for idx in range(len(PLOT_CONDITION_ORDER)):
                    if np.isfinite(means_arr[idx]):
                        data_extent.append(float(means_arr[idx]))
                    if np.isfinite(means_arr[idx]) and np.isfinite(sems_arr[idx]):
                        data_extent.append(float(means_arr[idx] + sems_arr[idx]))
            y_span = max(data_extent) - min(data_extent) if len(data_extent) >= 2 else 1.0
            y_pad = max(0.2, 0.05 * y_span)
            star_offset = 0.55 * y_pad
            top_margin = 1.6 * y_pad
            star_tops: list[float] = []
            for row in sig_df.itertuples(index=False):
                condition_idx = PLOT_CONDITION_ORDER.index(str(row.condition_factor))
                on_mean = mean_by_medication["ON"][condition_idx]
                on_sem = sem_by_medication["ON"][condition_idx]
                if not np.isfinite(on_mean):
                    continue
                star_y = float(on_mean + (on_sem if np.isfinite(on_sem) else 0.0) + star_offset)
                ax.text(
                    x_positions[condition_idx] + offsets["ON"],
                    star_y,
                    "*",
                    ha="center",
                    va="bottom",
                    color=colors["ON"],
                    fontsize=16,
                    fontweight="bold",
                )
                star_tops.append(star_y)
            if star_tops:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, max(ymax, max(star_tops) + top_margin))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(PLOT_CONDITION_ORDER, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("GVS condition")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_scores_by_condition(
    score_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    distance_metric: str,
    ylabel: str,
    out_path: Path,
    group_column: str,
    group_order: list[str],
    condition_order: list[str],
) -> None:
    metric_df = score_df.loc[score_df["distance_metric"] == distance_metric].copy()
    if metric_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x_positions = np.arange(len(condition_order), dtype=np.float64)
    if len(group_order) >= 2:
        offsets = np.linspace(-0.12, 0.12, num=len(group_order), dtype=np.float64)
    else:
        offsets = np.asarray([0.0], dtype=np.float64)
    color_map = {
        "OFF": "#4c78a8",
        "ON": "#f58518",
        "OFF_vs_ON_sham": "#54a24b",
    }

    mean_by_group: dict[str, np.ndarray] = {}
    sem_by_group: dict[str, np.ndarray] = {}
    for group_index, group_value in enumerate(group_order):
        group_df = metric_df.loc[metric_df[group_column] == group_value].copy()
        means: list[float] = []
        sems: list[float] = []
        for condition in condition_order:
            values = group_df.loc[group_df["condition_factor"] == condition, "ratio_score"].to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            means.append(float(np.mean(values)) if values.size else float("nan"))
            sems.append(float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size >= 2 else float("nan"))
        means_arr = np.asarray(means, dtype=np.float64)
        sems_arr = np.asarray(sems, dtype=np.float64)
        mean_by_group[str(group_value)] = means_arr
        sem_by_group[str(group_value)] = sems_arr
        ax.errorbar(
            x_positions + offsets[group_index],
            means_arr,
            yerr=sems_arr,
            fmt="o-",
            color=color_map.get(str(group_value), "#4c4c4c"),
            linewidth=1.5,
            markersize=5,
            capsize=3,
            label=str(group_value),
        )

    if not stats_df.empty:
        sig_df = stats_df.loc[
            (stats_df["distance_metric"] == distance_metric)
            & stats_df["condition_factor"].isin(condition_order)
            & stats_df["significant_fdr"].fillna(False)
        ].copy()
        if not sig_df.empty:
            data_extent: list[float] = []
            for group_value in group_order:
                means_arr = mean_by_group.get(str(group_value))
                sems_arr = sem_by_group.get(str(group_value))
                if means_arr is None or sems_arr is None:
                    continue
                for idx in range(len(condition_order)):
                    if np.isfinite(means_arr[idx]):
                        data_extent.append(float(means_arr[idx]))
                    if np.isfinite(means_arr[idx]) and np.isfinite(sems_arr[idx]):
                        data_extent.append(float(means_arr[idx] + sems_arr[idx]))
            y_span = max(data_extent) - min(data_extent) if len(data_extent) >= 2 else 1.0
            y_pad = max(0.2, 0.05 * y_span)
            star_offset = 0.55 * y_pad
            top_margin = 1.6 * y_pad
            star_tops: list[float] = []
            for row in sig_df.itertuples(index=False):
                row_group = str(getattr(row, group_column))
                if row_group not in group_order:
                    continue
                condition_idx = condition_order.index(str(row.condition_factor))
                group_idx = group_order.index(row_group)
                mean_arr = mean_by_group.get(row_group)
                sem_arr = sem_by_group.get(row_group)
                if mean_arr is None or not np.isfinite(mean_arr[condition_idx]):
                    continue
                sem_value = sem_arr[condition_idx] if sem_arr is not None and np.isfinite(sem_arr[condition_idx]) else 0.0
                star_y = float(mean_arr[condition_idx] + sem_value + star_offset)
                ax.text(
                    x_positions[condition_idx] + offsets[group_idx],
                    star_y,
                    "*",
                    ha="center",
                    va="bottom",
                    color=color_map.get(row_group, "#4c4c4c"),
                    fontsize=16,
                    fontweight="bold",
                )
                star_tops.append(star_y)
            if star_tops:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(ymin, max(ymax, max(star_tops) + top_margin))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(condition_order, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("GVS condition")
    if len(group_order) > 1:
        ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    trial_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    primary_delta_tests: pd.DataFrame,
    fc_df: pd.DataFrame,
    network_df: pd.DataFrame,
    network_tests: pd.DataFrame,
    run_level_fc_df: pd.DataFrame,
    approach1_stats: pd.DataFrame,
    approach2_stats: pd.DataFrame,
    edge_summary_df: pd.DataFrame,
) -> None:
    report_lines = [
        "# GVS Effects Analysis",
        "",
        "## Data coverage",
        f"- Subjects with trialwise data: {trial_df['subject'].nunique()}",
        f"- Subject-session rows: {summary_df['subject_session'].nunique()}",
        f"- Trial rows in primary analysis: {trial_df.shape[0]}",
        f"- Condition-level primary rows: {summary_df.shape[0]}",
        f"- Condition-level FC rows: {fc_df.shape[0]}",
        f"- Run-level FC rows: {run_level_fc_df.shape[0]}",
        "- Sham was defined as `gvs-01` / `GVS1` based on the requested prompt.",
        "- `meanFD` files were not found under the available data roots, so fitted models omit `meanFD`.",
        "",
        "## Primary sham-delta highlights",
    ]

    top_primary = (
        primary_delta_tests.sort_values(["q_value_fdr", "p_value"], ascending=[True, True])
        .head(10)
        .copy()
    )
    if top_primary.empty:
        report_lines.append("- No sham-delta primary tests were available.")
    else:
        for row in top_primary.itertuples(index=False):
            report_lines.append(
                f"- {row.metric} | {row.medication} | {row.condition_factor}: "
                f"mean delta={row.mean_delta:.4g}, p={row.p_value:.4g}, q={row.q_value_fdr:.4g}"
            )

    report_lines.extend(["", "## FC network-summary highlights"])
    top_network = (
        network_tests.sort_values(["q_value_fdr", "p_value"], ascending=[True, True])
        .head(10)
        .copy()
    )
    if top_network.empty:
        report_lines.append("- No network-summary sham-delta tests were available.")
    else:
        for row in top_network.itertuples(index=False):
            report_lines.append(
                f"- {row.metric} | {row.medication} | {row.condition_factor}: "
                f"mean delta={row.mean_delta:.4g}, p={row.p_value:.4g}, q={row.q_value_fdr:.4g}"
            )

    report_lines.extend(["", "## FC run-paired ratio highlights (Approach 1)"])
    if approach1_stats.empty:
        report_lines.append("- No run-paired FC ratio tests were available.")
    else:
        report_lines.append(
            "- `ratio_score` was defined as mean target-vs-sham distance divided by mean absolute within-condition distance. Significance was tested with a two-sided Wilcoxon signed-rank test on `log(ratio_score)` against 0, which is equivalent to testing `ratio_score` against 1."
        )
        for row in (
            approach1_stats.sort_values(["q_value_fdr", "p_value_wilcoxon"], ascending=[True, True])
            .head(10)
            .itertuples(index=False)
        ):
            report_lines.append(
                f"- {row.distance_metric} | {row.medication} | {row.condition_factor}: "
                f"mean ratio={row.mean_ratio_score:.4g}, geometric mean={row.geometric_mean_ratio_score:.4g}, "
                f"mean log ratio={row.mean_log_ratio_score:.4g}, p={row.p_value_wilcoxon:.4g}, q={row.q_value_fdr:.4g}"
            )

    report_lines.extend(["", "## FC OFF-vs-ON sham ratio highlights (Approach 2)"])
    if approach2_stats.empty:
        report_lines.append("- No OFF-vs-ON sham ratio tests were available.")
    else:
        report_lines.append(
            "- `ratio_score` was defined as mean target-vs-sham distance divided by mean absolute within-condition distance. Significance was tested with a two-sided Wilcoxon signed-rank test on `log(ratio_score)` against 0, which is equivalent to testing `ratio_score` against 1."
        )
        for row in (
            approach2_stats.sort_values(["q_value_fdr", "p_value_wilcoxon"], ascending=[True, True])
            .head(10)
            .itertuples(index=False)
        ):
            report_lines.append(
                f"- {row.distance_metric} | {row.condition_factor}: "
                f"mean ratio={row.mean_ratio_score:.4g}, geometric mean={row.geometric_mean_ratio_score:.4g}, "
                f"mean log ratio={row.mean_log_ratio_score:.4g}, p={row.p_value_wilcoxon:.4g}, q={row.q_value_fdr:.4g}"
            )

    report_lines.extend(["", "## Edge-level highlights"])
    if edge_summary_df.empty:
        report_lines.append("- No edge-level contrasts were available.")
    else:
        for row in edge_summary_df.sort_values(
            ["n_significant_edges_fdr", "min_q_value_fdr"], ascending=[False, True]
        ).head(10).itertuples(index=False):
            report_lines.append(
                f"- {row.medication} | {row.condition_factor}: "
                f"{row.n_significant_edges_fdr} FDR-significant edges, top edge={row.top_edge}"
            )

    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    primary_dir = ensure_dir(out_dir / "primary")
    fc_dir = ensure_dir(out_dir / "fc")

    projection_path = find_projection_path(args.projection_root.expanduser().resolve(), args.projection_glob)
    projection = np.asarray(np.load(projection_path), dtype=np.float64).ravel()
    gvs_orders = _load_gvs_orders(args.gvs_order_path.expanduser().resolve())
    manifest_rows, _ = _load_manifest_rows(
        args.manifest_path.expanduser().resolve(),
        gvs_orders,
        args.trials_per_block,
    )

    trial_df = build_trial_metadata(
        manifest_rows=manifest_rows,
        gvs_orders=gvs_orders,
        projection=projection,
        behavior_root=args.behavior_root.expanduser().resolve(),
        glm_root=args.glm_root.expanduser().resolve(),
        trials_per_block=int(args.trials_per_block),
    )
    trial_df.to_csv(primary_dir / "trial_level_projection_behavior.csv", index=False)

    summary_df = summarize_condition_metrics(trial_df)
    summary_df.to_csv(primary_dir / "session_condition_primary_metrics.csv", index=False)

    primary_metrics = [
        "mean_y_scaled",
        "consecutive_rmse_y_scaled",
        "consecutive_mad_y_scaled",
        "consecutive_lag1_corr_y",
        "y_rt_fisher_z",
    ]
    primary_delta_df = build_primary_delta_table(summary_df, primary_metrics)
    primary_delta_df.to_csv(primary_dir / "session_condition_primary_metrics_delta_vs_sham.csv", index=False)
    primary_delta_tests = run_sham_delta_tests(
        data=primary_delta_df,
        metric_columns=[f"delta_vs_sham_{metric}" for metric in primary_metrics],
        condition_column="condition_factor",
    )
    primary_delta_tests.to_csv(primary_dir / "primary_metric_sham_delta_tests.csv", index=False)

    _, consecutive_rmse_contrasts = fit_primary_models(
        trial_df=trial_df,
        summary_df=summary_df,
        out_dir=primary_dir,
    )

    plot_metric_by_condition(
        summary_df,
        metric="mean_y_scaled",
        ylabel="Mean projection (x1e5)",
        out_path=primary_dir / "mean_projection_by_condition.png",
    )
    plot_metric_by_condition(
        summary_df,
        metric="consecutive_rmse_y_scaled",
        ylabel="Consecutive RMSE (x1e5)",
        out_path=primary_dir / "consecutive_rmse_by_condition.png",
        significance_df=consecutive_rmse_contrasts,
    )
    plot_metric_by_condition(
        summary_df,
        metric="y_rt_fisher_z",
        ylabel="y-RT Fisher z",
        out_path=primary_dir / "y_rt_coupling_by_condition.png",
    )

    roi_members, roi_labels, roi_meta_df = build_roi_membership(
        roi_img_path=args.roi_img.expanduser().resolve(),
        roi_summary_path=args.roi_summary.expanduser().resolve(),
        selected_voxels_path=args.selected_voxels_path.expanduser().resolve(),
        min_roi_voxels=int(args.min_roi_voxels),
    )
    roi_meta_df.to_csv(fc_dir / "roi_nodes.csv", index=False)

    fc_df, matrices = compute_fc_outputs(
        by_gvs_dir=args.by_gvs_dir.expanduser().resolve(),
        roi_members=roi_members,
        roi_labels=roi_labels,
        out_dir=fc_dir,
    )
    fc_df.to_csv(fc_dir / "session_condition_fc_inventory.csv", index=False)

    network_df, _ = summarize_fc_deltas(
        matrices=matrices,
        roi_labels=roi_labels,
        out_dir=fc_dir,
    )
    network_df.to_csv(fc_dir / "network_summary_delta_vs_sham.csv", index=False)

    network_tests = run_sham_delta_tests(
        data=network_df,
        metric_columns=[
            "correlation_distance_vs_sham",
            "frobenius_norm_vs_sham",
            "laplacian_spectral_distance_vs_sham",
        ],
        condition_column="condition_factor",
    )
    network_tests.to_csv(fc_dir / "network_summary_sham_delta_tests.csv", index=False)

    edge_summary_df = pd.read_csv(fc_dir / "edge_level" / "edge_level_summary.csv")

    plot_metric_by_condition(
        network_df,
        metric="correlation_distance_vs_sham",
        ylabel="Correlation distance vs sham",
        out_path=fc_dir / "correlation_distance_by_condition.png",
    )
    plot_metric_by_condition(
        network_df,
        metric="frobenius_norm_vs_sham",
        ylabel="Frobenius norm vs sham",
        out_path=fc_dir / "frobenius_by_condition.png",
    )
    plot_metric_by_condition(
        network_df,
        metric="laplacian_spectral_distance_vs_sham",
        ylabel="Laplacian spectral distance vs sham",
        out_path=fc_dir / "laplacian_distance_by_condition.png",
    )

    run_level_fc_df, run_level_matrices = compute_run_level_fc_outputs(
        session_beta_dir=args.session_beta_dir.expanduser().resolve(),
        manifest_rows=manifest_rows,
        roi_members=roi_members,
        roi_labels=roi_labels,
        out_dir=fc_dir,
    )
    run_level_fc_df.to_csv(fc_dir / "run_level_fc_inventory.csv", index=False)

    approach1_df = summarize_runmatched_fc_scores(run_level_matrices)
    approach1_df.to_csv(fc_dir / "approach1_runpaired_ratio_scores.csv", index=False)
    approach1_stats = summarize_ratio_score_stats(approach1_df, strata_columns=["medication"])
    approach1_stats.to_csv(fc_dir / "approach1_runpaired_ratio_stats.csv", index=False)

    approach2_df = summarize_off_to_on_sham_scores(run_level_matrices)
    approach2_df["comparison_group"] = "OFF_vs_ON_sham"
    approach2_df.to_csv(fc_dir / "approach2_off_vs_on_sham_ratio_scores.csv", index=False)
    approach2_stats = summarize_ratio_score_stats(approach2_df, strata_columns=[])
    approach2_stats["comparison_group"] = "OFF_vs_ON_sham"
    approach2_stats.to_csv(fc_dir / "approach2_off_vs_on_sham_ratio_stats.csv", index=False)

    plot_ratio_scores_by_condition(
        score_df=approach1_df,
        stats_df=approach1_stats,
        distance_metric="correlation_distance",
        ylabel="Run-paired correlation-distance ratio",
        out_path=fc_dir / "approach1_correlation_ratio_by_condition.png",
        group_column="medication",
        group_order=MEDICATION_ORDER,
        condition_order=ACTIVE_CONDITION_FACTORS,
    )
    plot_ratio_scores_by_condition(
        score_df=approach1_df,
        stats_df=approach1_stats,
        distance_metric="frobenius_norm",
        ylabel="Run-paired Frobenius ratio",
        out_path=fc_dir / "approach1_frobenius_ratio_by_condition.png",
        group_column="medication",
        group_order=MEDICATION_ORDER,
        condition_order=ACTIVE_CONDITION_FACTORS,
    )
    plot_ratio_scores_by_condition(
        score_df=approach1_df,
        stats_df=approach1_stats,
        distance_metric="laplacian_spectral_distance",
        ylabel="Run-paired Laplacian-distance ratio",
        out_path=fc_dir / "approach1_laplacian_ratio_by_condition.png",
        group_column="medication",
        group_order=MEDICATION_ORDER,
        condition_order=ACTIVE_CONDITION_FACTORS,
    )
    plot_ratio_scores_by_condition(
        score_df=approach2_df,
        stats_df=approach2_stats,
        distance_metric="correlation_distance",
        ylabel="OFF GVS vs ON sham correlation-distance ratio",
        out_path=fc_dir / "approach2_correlation_ratio_by_condition.png",
        group_column="comparison_group",
        group_order=["OFF_vs_ON_sham"],
        condition_order=APPROACH2_CONDITION_FACTORS,
    )
    plot_ratio_scores_by_condition(
        score_df=approach2_df,
        stats_df=approach2_stats,
        distance_metric="frobenius_norm",
        ylabel="OFF GVS vs ON sham Frobenius ratio",
        out_path=fc_dir / "approach2_frobenius_ratio_by_condition.png",
        group_column="comparison_group",
        group_order=["OFF_vs_ON_sham"],
        condition_order=APPROACH2_CONDITION_FACTORS,
    )
    plot_ratio_scores_by_condition(
        score_df=approach2_df,
        stats_df=approach2_stats,
        distance_metric="laplacian_spectral_distance",
        ylabel="OFF GVS vs ON sham Laplacian-distance ratio",
        out_path=fc_dir / "approach2_laplacian_ratio_by_condition.png",
        group_column="comparison_group",
        group_order=["OFF_vs_ON_sham"],
        condition_order=APPROACH2_CONDITION_FACTORS,
    )

    manifest_payload = {
        "projection_path": str(projection_path),
        "gvs_order_path": str(args.gvs_order_path.expanduser().resolve()),
        "manifest_path": str(args.manifest_path.expanduser().resolve()),
        "behavior_root": str(args.behavior_root.expanduser().resolve()),
        "glm_root": str(args.glm_root.expanduser().resolve()),
        "by_gvs_dir": str(args.by_gvs_dir.expanduser().resolve()),
        "session_beta_dir": str(args.session_beta_dir.expanduser().resolve()),
        "selected_voxels_path": str(args.selected_voxels_path.expanduser().resolve()),
        "roi_img": str(args.roi_img.expanduser().resolve()),
        "roi_summary": str(args.roi_summary.expanduser().resolve()),
        "trials_per_block": int(args.trials_per_block),
        "min_roi_voxels": int(args.min_roi_voxels),
        "sham_condition_code": SHAM_CONDITION_CODE,
        "active_condition_codes": ACTIVE_CONDITION_CODES,
        "n_trial_rows": int(trial_df.shape[0]),
        "n_condition_rows": int(summary_df.shape[0]),
        "n_fc_rows": int(fc_df.shape[0]),
        "n_run_level_fc_rows": int(run_level_fc_df.shape[0]),
        "n_approach1_rows": int(approach1_df.shape[0]),
        "n_approach2_rows": int(approach2_df.shape[0]),
        "n_rois": int(len(roi_labels)),
    }
    (out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    write_report(
        out_dir=out_dir,
        trial_df=trial_df,
        summary_df=summary_df,
        primary_delta_tests=primary_delta_tests,
        fc_df=fc_df,
        network_df=network_df,
        network_tests=network_tests,
        run_level_fc_df=run_level_fc_df,
        approach1_stats=approach1_stats,
        approach2_stats=approach2_stats,
        edge_summary_df=edge_summary_df,
    )

    print(f"Saved primary outputs to: {primary_dir}", flush=True)
    print(f"Saved FC outputs to: {fc_dir}", flush=True)
    print(f"Saved report to: {out_dir / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
