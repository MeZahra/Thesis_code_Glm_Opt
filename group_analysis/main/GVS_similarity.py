#!/usr/bin/env python3
"""Subject-level GVS beta-matrix similarity analysis.

This analysis works directly on ROI-by-trial beta matrices rather than ROI
connectivity matrices. Each condition-specific matrix is derived from the saved
`results/connectivity/GVS_effects/data/by_gvs/.../selected_beta_trials_gvs-XX.npy`
files using the existing selected-network ROI membership.

Primary similarity metric:
    Flattened Pearson correlation after:
    1. averaging selected voxels into ROI rows,
    2. resampling each ROI row to a fixed trial grid,
    3. z-scoring each ROI row.

Additional metrics:
    - flattened cosine similarity on the resampled raw beta matrices
    - Frobenius norm of the difference between row-z-scored matrices
    - RMSE on the row-z-scored matrices
    - eigenvalue-profile correlation
    - eigenvalue-profile L2 distance

Because the ROI beta matrices are rectangular, the spectral metrics are derived
from the eigenvalues of the ROI Gram matrix `X @ X.T / n_trials` after the same
row-wise z-scoring used by the primary metric.

Comparisons requested:
    1. OFF condition vs OFF sham, within subject
    2. ON condition vs ON sham, within subject
    3. OFF condition, including OFF sham, vs ON sham, within subject
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_GROUP_ANALYSIS_DIR = _HERE.parent
_REPO_ROOT = _GROUP_ANALYSIS_DIR.parent
if str(_GROUP_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_GROUP_ANALYSIS_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from group_analysis.main.gvs_effects_analysis import (  # noqa: E402
    ACTIVE_CONDITION_CODES,
    DEFAULT_BY_GVS_DIR,
    DEFAULT_ROI_IMG,
    DEFAULT_ROI_SUMMARY,
    DEFAULT_SELECTED_VOXELS_PATH,
    SHAM_CONDITION_CODE,
    build_roi_membership,
    condition_factor_from_code,
    ensure_dir,
    medication_from_session,
)


DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "results" / "connectivity" / "GVS_effects" / "gvs_similarity"
DEFAULT_TARGET_TRIALS = 20
MIN_ROI_VOXELS = 5
CONDITION_FILE_RE = re.compile(r"^selected_beta_trials_(gvs-\d+)\.npy$")
PRIMARY_METRIC = "flat_pearson_r"
SECONDARY_METRICS = (
    "flat_cosine_similarity",
    "frobenius_norm_diff",
    "zscore_rmse",
    "eigenvalue_profile_correlation",
    "eigenvalue_profile_l2_distance",
)
ALL_CONDITION_CODES = [SHAM_CONDITION_CODE, *ACTIVE_CONDITION_CODES]
HEATMAP_CMAP = "jet"
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "flat_pearson_r": {
        "label": "Flattened Pearson r",
        "vmin": -1.0,
        "vmax": 1.0,
        "higher_is_more_similar": True,
    },
    "flat_cosine_similarity": {
        "label": "Flattened cosine similarity",
        "vmin": -1.0,
        "vmax": 1.0,
        "higher_is_more_similar": True,
    },
    "frobenius_norm_diff": {
        "label": "Frobenius norm difference",
        "vmin": 0.0,
        "vmax": None,
        "higher_is_more_similar": False,
    },
    "zscore_rmse": {
        "label": "Z-scored RMSE",
        "vmin": 0.0,
        "vmax": None,
        "higher_is_more_similar": False,
    },
    "eigenvalue_profile_correlation": {
        "label": "Eigenvalue-profile correlation",
        "vmin": -1.0,
        "vmax": 1.0,
        "higher_is_more_similar": True,
    },
    "eigenvalue_profile_l2_distance": {
        "label": "Eigenvalue-profile L2 distance",
        "vmin": 0.0,
        "vmax": None,
        "higher_is_more_similar": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build subject/session ROI beta matrices from GVS beta splits and "
            "measure sham-reference similarity within and across medication states."
        )
    )
    parser.add_argument("--by-gvs-dir", type=Path, default=DEFAULT_BY_GVS_DIR)
    parser.add_argument("--selected-voxels-path", type=Path, default=DEFAULT_SELECTED_VOXELS_PATH)
    parser.add_argument("--roi-img", type=Path, default=DEFAULT_ROI_IMG)
    parser.add_argument("--roi-summary", type=Path, default=DEFAULT_ROI_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--min-roi-voxels", type=int, default=MIN_ROI_VOXELS)
    parser.add_argument(
        "--target-trials",
        type=int,
        default=DEFAULT_TARGET_TRIALS,
        help="Fixed trial-grid width used when resampling ROI rows before comparison.",
    )
    return parser.parse_args()


def _condition_display_name(code: str) -> str:
    return condition_factor_from_code(code)


def _compute_roi_beta_matrix(beta: np.ndarray, roi_members: list[np.ndarray]) -> np.ndarray:
    beta_array = np.asarray(beta, dtype=np.float64)
    roi_beta = np.full((len(roi_members), beta_array.shape[1]), np.nan, dtype=np.float64)
    if beta_array.shape[1] == 0:
        return roi_beta
    for roi_index, members in enumerate(roi_members):
        roi_beta[roi_index] = np.nanmean(beta_array[members, :], axis=0)
    return roi_beta


def _resample_row(values: np.ndarray, target_trials: int) -> np.ndarray:
    row = np.asarray(values, dtype=np.float64).ravel()
    out = np.full(int(target_trials), np.nan, dtype=np.float64)
    if row.size == 0:
        return out
    finite = np.isfinite(row)
    if not np.any(finite):
        return out
    valid_values = row[finite]
    if valid_values.size == 1:
        out[:] = float(valid_values[0])
        return out
    x_src = np.linspace(0.0, 1.0, row.size, dtype=np.float64)[finite]
    x_dst = np.linspace(0.0, 1.0, int(target_trials), dtype=np.float64)
    out[:] = np.interp(x_dst, x_src, valid_values)
    return out


def _resample_matrix(matrix: np.ndarray, target_trials: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    return np.vstack([_resample_row(row, target_trials=target_trials) for row in array])


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    out = np.full_like(array, np.nan, dtype=np.float64)
    for row_index, row in enumerate(array):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        values = row[finite]
        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=0))
        if not np.isfinite(std_value) or np.isclose(std_value, 0.0):
            out[row_index, finite] = 0.0
            continue
        out[row_index, finite] = (values - mean_value) / std_value
    return out


def _pearson_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or y.size < 3:
        return float("nan")
    x_std = float(np.std(x, ddof=0))
    y_std = float(np.std(y, ddof=0))
    if np.isclose(x_std, 0.0) or np.isclose(y_std, 0.0):
        return float("nan")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else float("nan")


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_norm = float(np.linalg.norm(x))
    y_norm = float(np.linalg.norm(y))
    if np.isclose(x_norm, 0.0) or np.isclose(y_norm, 0.0):
        return float("nan")
    value = float(np.dot(x, y) / (x_norm * y_norm))
    return value if np.isfinite(value) else float("nan")


def _row_gram_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got {array.shape}.")
    finite = np.where(np.isfinite(array), array, 0.0)
    scale = float(max(1, finite.shape[1]))
    gram = (finite @ finite.T) / scale
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.sort(np.asarray(eigvals, dtype=np.float64))[::-1]
    eigvals[~np.isfinite(eigvals)] = np.nan
    return eigvals


def compute_matrix_similarity(
    target_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    target_trials: int,
) -> dict[str, float | int]:
    target_resampled = _resample_matrix(target_matrix, target_trials=target_trials)
    reference_resampled = _resample_matrix(reference_matrix, target_trials=target_trials)
    target_norm = _zscore_rows(target_resampled)
    reference_norm = _zscore_rows(reference_resampled)

    valid = np.isfinite(target_norm) & np.isfinite(reference_norm)
    x = target_norm[valid]
    y = reference_norm[valid]
    raw_valid = np.isfinite(target_resampled) & np.isfinite(reference_resampled)
    x_raw = target_resampled[raw_valid]
    y_raw = reference_resampled[raw_valid]
    target_eig = _row_gram_eigenvalues(target_norm)
    reference_eig = _row_gram_eigenvalues(reference_norm)
    eig_valid = np.isfinite(target_eig) & np.isfinite(reference_eig)
    eig_x = target_eig[eig_valid]
    eig_y = reference_eig[eig_valid]
    if x.size == 0 or y.size == 0:
        return {
            "n_overlap_values": 0,
            "flat_pearson_r": float("nan"),
            "flat_cosine_similarity": float("nan"),
            "frobenius_norm_diff": float("nan"),
            "zscore_rmse": float("nan"),
            "eigenvalue_profile_correlation": float("nan"),
            "eigenvalue_profile_l2_distance": float("nan"),
        }

    diff = x - y
    return {
        "n_overlap_values": int(x.size),
        "flat_pearson_r": _pearson_similarity(x, y),
        "flat_cosine_similarity": _cosine_similarity(x_raw, y_raw),
        "frobenius_norm_diff": float(np.linalg.norm(target_norm - reference_norm, ord="fro")),
        "zscore_rmse": float(np.sqrt(np.mean(diff**2))),
        "eigenvalue_profile_correlation": _pearson_similarity(eig_x, eig_y),
        "eigenvalue_profile_l2_distance": float(np.linalg.norm(eig_x - eig_y, ord=2))
        if eig_x.size and eig_y.size
        else float("nan"),
    }


def load_roi_beta_matrices(
    by_gvs_dir: Path,
    roi_members: list[np.ndarray],
    matrix_dir: Path,
) -> tuple[dict[tuple[str, int, str], np.ndarray], pd.DataFrame]:
    matrices: dict[tuple[str, int, str], np.ndarray] = {}
    inventory_rows: list[dict[str, Any]] = []

    for subject_dir in sorted(path for path in by_gvs_dir.iterdir() if path.is_dir()):
        subject = subject_dir.name
        for session_dir in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            session = int(session_dir.name.split("-")[-1])
            medication = medication_from_session(session)
            output_session_dir = ensure_dir(matrix_dir / subject / session_dir.name)
            for beta_path in sorted(session_dir.glob("selected_beta_trials_gvs-*.npy")):
                match = CONDITION_FILE_RE.match(beta_path.name)
                if match is None:
                    continue
                condition_code = match.group(1)
                beta = np.asarray(np.load(beta_path), dtype=np.float64)
                roi_beta = _compute_roi_beta_matrix(beta=beta, roi_members=roi_members)
                matrices[(subject, session, condition_code)] = roi_beta

                saved_matrix_path = output_session_dir / f"{condition_code}_roi_beta_matrix.npy"
                np.save(saved_matrix_path, roi_beta.astype(np.float32, copy=False))
                inventory_rows.append(
                    {
                        "subject": subject,
                        "session": int(session),
                        "medication": medication,
                        "condition_code": condition_code,
                        "condition_label": _condition_display_name(condition_code),
                        "n_rois": int(roi_beta.shape[0]),
                        "n_trials": int(roi_beta.shape[1]),
                        "matrix_path": str(saved_matrix_path),
                        "source_beta_path": str(beta_path),
                    }
                )

    inventory_df = pd.DataFrame(inventory_rows).sort_values(
        ["subject", "session", "condition_code"]
    ).reset_index(drop=True)
    return matrices, inventory_df


def _comparison_row(
    *,
    subject: str,
    target_session: int,
    target_condition_code: str,
    target_matrix: np.ndarray,
    reference_session: int,
    reference_condition_code: str,
    reference_matrix: np.ndarray,
    comparison_kind: str,
    target_trials: int,
) -> dict[str, Any]:
    similarity = compute_matrix_similarity(
        target_matrix=target_matrix,
        reference_matrix=reference_matrix,
        target_trials=target_trials,
    )
    return {
        "subject": subject,
        "target_session": int(target_session),
        "target_medication": medication_from_session(target_session),
        "target_condition_code": target_condition_code,
        "target_condition_label": _condition_display_name(target_condition_code),
        "reference_session": int(reference_session),
        "reference_medication": medication_from_session(reference_session),
        "reference_condition_code": reference_condition_code,
        "reference_condition_label": _condition_display_name(reference_condition_code),
        "comparison_kind": comparison_kind,
        "n_rois": int(target_matrix.shape[0]),
        "n_trials_target": int(target_matrix.shape[1]),
        "n_trials_reference": int(reference_matrix.shape[1]),
        "target_trials_resampled": int(target_trials),
        **similarity,
    }


def build_within_medication_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    target_trials: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        for session in (1, 2):
            sham_matrix = matrices.get((subject, session, SHAM_CONDITION_CODE))
            if sham_matrix is None:
                continue
            for condition_code in ACTIVE_CONDITION_CODES:
                target_matrix = matrices.get((subject, session, condition_code))
                if target_matrix is None:
                    continue
                rows.append(
                    _comparison_row(
                        subject=subject,
                        target_session=session,
                        target_condition_code=condition_code,
                        target_matrix=target_matrix,
                        reference_session=session,
                        reference_condition_code=SHAM_CONDITION_CODE,
                        reference_matrix=sham_matrix,
                        comparison_kind="within_med_sham_reference",
                        target_trials=target_trials,
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["target_medication", "subject", "target_condition_code"]
    ).reset_index(drop=True)


def build_off_to_on_sham_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
    target_trials: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        on_sham = matrices.get((subject, 2, SHAM_CONDITION_CODE))
        if on_sham is None:
            continue
        for condition_code in ALL_CONDITION_CODES:
            off_matrix = matrices.get((subject, 1, condition_code))
            if off_matrix is None:
                continue
            rows.append(
                _comparison_row(
                    subject=subject,
                    target_session=1,
                    target_condition_code=condition_code,
                    target_matrix=off_matrix,
                    reference_session=2,
                    reference_condition_code=SHAM_CONDITION_CODE,
                    reference_matrix=on_sham,
                    comparison_kind="off_to_on_sham_reference",
                    target_trials=target_trials,
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["subject", "target_condition_code"]).reset_index(drop=True)


def _trial_vectors(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D ROI beta matrix, got {array.shape}.")
    return array.T


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D trial matrix, got {array.shape}.")
    filled = np.where(np.isfinite(array), array, 0.0)
    norms = np.linalg.norm(filled, axis=1, keepdims=True)
    norms = np.where(np.isclose(norms, 0.0), 1.0, norms)
    return filled / norms


def build_trial_distance_to_on_sham_rows(
    matrices: dict[tuple[str, int, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subjects = sorted({subject for subject, _, _ in matrices})
    for subject in subjects:
        on_sham_matrix = matrices.get((subject, 2, SHAM_CONDITION_CODE))
        if on_sham_matrix is None:
            continue
        on_sham_trials = _l2_normalize_rows(_trial_vectors(on_sham_matrix))
        if on_sham_trials.shape[0] == 0:
            continue

        for condition_code in ALL_CONDITION_CODES:
            off_matrix = matrices.get((subject, 1, condition_code))
            if off_matrix is None:
                continue
            off_trials = _l2_normalize_rows(_trial_vectors(off_matrix))
            if off_trials.shape[0] == 0:
                continue

            similarity_matrix = off_trials @ on_sham_trials.T
            distance_matrix = 1.0 - similarity_matrix
            closest_indices = np.argmin(distance_matrix, axis=1)

            for trial_index in range(off_trials.shape[0]):
                trial_sim = similarity_matrix[trial_index]
                trial_dist = distance_matrix[trial_index]
                rows.append(
                    {
                        "subject": subject,
                        "target_session": 1,
                        "target_medication": "OFF",
                        "target_condition_code": condition_code,
                        "target_condition_label": _condition_display_name(condition_code),
                        "reference_session": 2,
                        "reference_medication": "ON",
                        "reference_condition_code": SHAM_CONDITION_CODE,
                        "reference_condition_label": _condition_display_name(SHAM_CONDITION_CODE),
                        "target_trial_index": int(trial_index + 1),
                        "n_rois": int(off_matrix.shape[0]),
                        "n_reference_trials": int(on_sham_trials.shape[0]),
                        "mean_cosine_similarity_to_on_sham": float(np.mean(trial_sim)),
                        "max_cosine_similarity_to_on_sham": float(np.max(trial_sim)),
                        "mean_cosine_distance_to_on_sham": float(np.mean(trial_dist)),
                        "min_cosine_distance_to_on_sham": float(np.min(trial_dist)),
                        "closest_on_sham_trial_index": int(closest_indices[trial_index] + 1),
                    }
                )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["subject", "target_condition_code", "target_trial_index"]
    ).reset_index(drop=True)


def _format_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _render_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    metric_label: str,
    out_path: Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if pivot_df.empty:
        return
    data = pivot_df.to_numpy(dtype=np.float64)
    masked = np.ma.masked_invalid(data)
    fig_w = max(6.0, 1.15 * pivot_df.shape[1] + 2.0)
    fig_h = max(4.0, 0.45 * pivot_df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(pivot_df.shape[1]))
    ax.set_xticklabels(pivot_df.columns.tolist(), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot_df.shape[0]))
    ax.set_yticklabels(pivot_df.index.tolist())
    ax.set_title(title)
    for row_idx in range(pivot_df.shape[0]):
        for col_idx in range(pivot_df.shape[1]):
            value = data[row_idx, col_idx]
            if not np.isfinite(value):
                continue
            text_color = "white" if (vmin is not None and vmax is not None and value < (vmin + vmax) / 2.0) else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(metric_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_metric_heatmap_bundle(
    df: pd.DataFrame,
    tables_dir: Path,
    plots_dir: Path,
    *,
    metric_name: str,
    group_name: str,
    title: str,
    condition_codes: list[str],
) -> None:
    if df.empty:
        return
    if metric_name not in METRIC_SPECS:
        raise KeyError(f"Unknown metric {metric_name!r}.")
    ordered_labels = [_condition_display_name(code) for code in condition_codes]
    subset = df.copy()
    subset["target_condition_label"] = pd.Categorical(
        subset["target_condition_label"],
        categories=ordered_labels,
        ordered=True,
    )
    pivot_df = subset.pivot(index="subject", columns="target_condition_label", values=metric_name)
    pivot_df = pivot_df.reindex(columns=ordered_labels)
    pivot_df = pivot_df.sort_index()
    pivot_df.to_csv(tables_dir / f"{group_name}_{metric_name}_wide.csv")
    metric_spec = METRIC_SPECS[metric_name]
    _render_heatmap(
        pivot_df=pivot_df,
        title=title,
        metric_label=str(metric_spec["label"]),
        out_path=plots_dir / f"{group_name}_{metric_name}_heatmap.png",
        vmin=metric_spec["vmin"],
        vmax=metric_spec["vmax"],
    )


def _write_metric_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    grouped = (
        df.groupby(
            ["comparison_kind", "target_medication", "target_condition_code", "target_condition_label"],
            dropna=False,
            observed=False,
        )[[PRIMARY_METRIC, *SECONDARY_METRICS]]
        .agg(["mean", "std", "count"])
    )
    grouped.columns = ["_".join(part for part in col if part) for col in grouped.columns]
    grouped = grouped.reset_index().sort_values(
        ["comparison_kind", "target_medication", "target_condition_code"]
    )
    grouped.to_csv(out_path, index=False)


def _write_trial_distance_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    grouped = (
        df.groupby(
            ["subject", "target_condition_code", "target_condition_label"],
            dropna=False,
            observed=False,
        )[
            [
                "mean_cosine_similarity_to_on_sham",
                "max_cosine_similarity_to_on_sham",
                "mean_cosine_distance_to_on_sham",
                "min_cosine_distance_to_on_sham",
            ]
        ]
        .agg(["mean", "median", "std", "count"])
    )
    grouped.columns = ["_".join(part for part in col if part) for col in grouped.columns]
    grouped = grouped.reset_index().sort_values(["subject", "target_condition_code"])
    grouped.to_csv(out_path, index=False)


def _plot_trial_distance_boxplots(
    subject_df: pd.DataFrame,
    out_path: Path,
) -> None:
    if subject_df.empty:
        return
    ordered_codes = [code for code in ALL_CONDITION_CODES if code in set(subject_df["target_condition_code"])]
    ordered_labels = [_condition_display_name(code) for code in ordered_codes]
    mean_series = []
    min_series = []
    for code in ordered_codes:
        condition_rows = subject_df.loc[subject_df["target_condition_code"] == code]
        mean_values = condition_rows["mean_cosine_distance_to_on_sham"].to_numpy(dtype=np.float64)
        min_values = condition_rows["min_cosine_distance_to_on_sham"].to_numpy(dtype=np.float64)
        mean_series.append(mean_values[np.isfinite(mean_values)])
        min_series.append(min_values[np.isfinite(min_values)])

    colors = plt.get_cmap("jet")(np.linspace(0.1, 0.9, len(ordered_codes)))
    fig, axes = plt.subplots(1, 2, figsize=(max(10.0, 1.2 * len(ordered_codes) + 4.0), 4.8), sharey=False)
    panels = [
        ("mean_cosine_distance_to_on_sham", "Mean distance to ON sham", mean_series),
        ("min_cosine_distance_to_on_sham", "Min distance to ON sham", min_series),
    ]
    for ax, (_, ylabel, values_list) in zip(axes, panels, strict=True):
        box = ax.boxplot(values_list, patch_artist=True, tick_labels=ordered_labels, widths=0.65, showfliers=False)
        for patch, color in zip(box["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        for median in box["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("OFF condition")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"{subject_df['subject'].iloc[0]}: OFF trial distance to ON sham trials")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_report(
    out_dir: Path,
    roi_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    within_df: pd.DataFrame,
    off_to_on_df: pd.DataFrame,
    trial_distance_df: pd.DataFrame,
    target_trials: int,
) -> None:
    lines: list[str] = []
    lines.append("# GVS Similarity Analysis")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- ROI beta matrices were built from saved `by_gvs` beta splits using {int(roi_df.shape[0])} ROI rows.")
    lines.append(f"- Each ROI row was resampled to `{int(target_trials)}` trial positions before comparison.")
    lines.append("- Primary similarity metric: flattened Pearson correlation after ROI-wise z-scoring.")
    lines.append("- Additional metrics: flattened cosine similarity, Frobenius norm, z-scored RMSE, and eigenvalue-profile comparisons.")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Subject/session/condition beta matrices: {int(inventory_df.shape[0])}")
    lines.append(f"- Unique subjects with at least one beta matrix: {int(inventory_df['subject'].nunique()) if not inventory_df.empty else 0}")
    lines.append(f"- Within-medication sham-reference rows: {int(within_df.shape[0])}")
    lines.append(f"- OFF-vs-ON sham-reference rows: {int(off_to_on_df.shape[0])}")
    lines.append(f"- Trial-level OFF-to-ON sham rows: {int(trial_distance_df.shape[0])}")
    lines.append("")

    def _append_summary_block(title: str, df: pd.DataFrame) -> None:
        lines.append(f"## {title}")
        if df.empty:
            lines.append("- No rows available.")
            lines.append("")
            return
        ascending = not bool(METRIC_SPECS[PRIMARY_METRIC]["higher_is_more_similar"])
        summary = (
            df.groupby("target_condition_label", dropna=False, observed=False)[PRIMARY_METRIC]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=ascending)
        )
        for row in summary.itertuples(index=False):
            lines.append(
                f"- {row.target_condition_label}: mean {PRIMARY_METRIC}={_format_value(row.mean)}, "
                f"sd={_format_value(row.std)}, n={int(row.count)}"
            )
        lines.append("")

    _append_summary_block(
        "Within-Medication OFF vs OFF Sham",
        within_df.loc[within_df["target_medication"] == "OFF"].reset_index(drop=True)
        if not within_df.empty
        else pd.DataFrame(),
    )
    _append_summary_block(
        "Within-Medication ON vs ON Sham",
        within_df.loc[within_df["target_medication"] == "ON"].reset_index(drop=True)
        if not within_df.empty
        else pd.DataFrame(),
    )
    _append_summary_block("OFF Condition vs ON Sham", off_to_on_df)

    lines.append("## Trial-Level OFF Trial Distance vs ON Sham")
    if trial_distance_df.empty:
        lines.append("- No rows available.")
        lines.append("")
    else:
        trial_summary = (
            trial_distance_df.groupby("target_condition_label", dropna=False, observed=False)[
                ["mean_cosine_distance_to_on_sham", "min_cosine_distance_to_on_sham"]
            ]
            .mean()
            .reset_index()
            .sort_values("mean_cosine_distance_to_on_sham")
        )
        for row in trial_summary.itertuples(index=False):
            lines.append(
                f"- {row.target_condition_label}: mean trial mean-distance={_format_value(row.mean_cosine_distance_to_on_sham)}, "
                f"mean trial min-distance={_format_value(row.min_cosine_distance_to_on_sham)}"
            )
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    by_gvs_dir = args.by_gvs_dir.expanduser().resolve()
    selected_voxels_path = args.selected_voxels_path.expanduser().resolve()
    roi_img_path = args.roi_img.expanduser().resolve()
    roi_summary_path = args.roi_summary.expanduser().resolve()
    out_dir = ensure_dir(args.out_dir.expanduser().resolve())
    if int(args.target_trials) <= 0:
        raise ValueError(f"--target-trials must be positive, got {args.target_trials}.")

    roi_members, roi_labels, _ = build_roi_membership(
        roi_img_path=roi_img_path,
        roi_summary_path=roi_summary_path,
        selected_voxels_path=selected_voxels_path,
        min_roi_voxels=int(args.min_roi_voxels),
    )
    if not roi_members:
        raise ValueError("No ROI members passed the requested threshold.")

    common_dir = ensure_dir(out_dir / "common")
    within_dir = ensure_dir(out_dir / "within_med_sham_reference")
    off_to_on_dir = ensure_dir(out_dir / "off_to_on_sham_reference")
    trial_to_on_sham_dir = ensure_dir(out_dir / "trial_distance_to_on_sham")
    matrix_dir = ensure_dir(common_dir / "matrices")
    roi_df = pd.DataFrame(
        {
            "roi_index": np.arange(1, len(roi_labels) + 1, dtype=np.int64),
            "roi_label": roi_labels,
            "n_selected_voxels": [int(members.size) for members in roi_members],
        }
    )
    roi_df.to_csv(common_dir / "roi_nodes.csv", index=False)

    matrices, inventory_df = load_roi_beta_matrices(
        by_gvs_dir=by_gvs_dir,
        roi_members=roi_members,
        matrix_dir=matrix_dir,
    )
    inventory_df.to_csv(common_dir / "matrix_inventory.csv", index=False)

    within_df = build_within_medication_rows(matrices=matrices, target_trials=int(args.target_trials))
    within_tables_dir = ensure_dir(within_dir / "tables")
    within_plots_dir = ensure_dir(within_dir / "plots")
    within_df.to_csv(within_tables_dir / "subject_level_similarity.csv", index=False)
    _write_metric_summary(within_df, within_tables_dir / "condition_summary.csv")
    within_off_df = (
        within_df.loc[within_df["target_medication"] == "OFF"].reset_index(drop=True)
        if not within_df.empty
        else pd.DataFrame()
    )
    within_on_df = (
        within_df.loc[within_df["target_medication"] == "ON"].reset_index(drop=True)
        if not within_df.empty
        else pd.DataFrame()
    )
    for metric_name in (PRIMARY_METRIC, *SECONDARY_METRICS):
        _write_metric_heatmap_bundle(
            df=within_off_df,
            tables_dir=within_tables_dir,
            plots_dir=within_plots_dir,
            metric_name=metric_name,
            group_name="off_vs_off_sham",
            title=f"OFF: active GVS beta-matrix {METRIC_SPECS[metric_name]['label']} vs OFF sham",
            condition_codes=ACTIVE_CONDITION_CODES,
        )
        _write_metric_heatmap_bundle(
            df=within_on_df,
            tables_dir=within_tables_dir,
            plots_dir=within_plots_dir,
            metric_name=metric_name,
            group_name="on_vs_on_sham",
            title=f"ON: active GVS beta-matrix {METRIC_SPECS[metric_name]['label']} vs ON sham",
            condition_codes=ACTIVE_CONDITION_CODES,
        )

    off_to_on_df = build_off_to_on_sham_rows(matrices=matrices, target_trials=int(args.target_trials))
    off_to_on_tables_dir = ensure_dir(off_to_on_dir / "tables")
    off_to_on_plots_dir = ensure_dir(off_to_on_dir / "plots")
    off_to_on_df.to_csv(off_to_on_tables_dir / "subject_level_similarity.csv", index=False)
    _write_metric_summary(off_to_on_df, off_to_on_tables_dir / "condition_summary.csv")
    for metric_name in (PRIMARY_METRIC, *SECONDARY_METRICS):
        _write_metric_heatmap_bundle(
            df=off_to_on_df,
            tables_dir=off_to_on_tables_dir,
            plots_dir=off_to_on_plots_dir,
            metric_name=metric_name,
            group_name="off_condition_vs_on_sham",
            title=f"OFF condition beta-matrix {METRIC_SPECS[metric_name]['label']} vs ON sham",
            condition_codes=ALL_CONDITION_CODES,
        )

    trial_tables_dir = ensure_dir(trial_to_on_sham_dir / "tables")
    trial_plots_dir = ensure_dir(trial_to_on_sham_dir / "plots")
    trial_distance_df = build_trial_distance_to_on_sham_rows(matrices=matrices)
    trial_distance_df.to_csv(trial_tables_dir / "trial_level_similarity.csv", index=False)
    _write_trial_distance_summary(trial_distance_df, trial_tables_dir / "subject_condition_summary.csv")
    for subject, subject_df in trial_distance_df.groupby("subject", sort=True):
        _plot_trial_distance_boxplots(
            subject_df=subject_df.reset_index(drop=True),
            out_path=trial_plots_dir / f"{subject}_trial_distance_to_on_sham_boxplots.png",
        )

    manifest = {
        "by_gvs_dir": str(by_gvs_dir),
        "selected_voxels_path": str(selected_voxels_path),
        "roi_img": str(roi_img_path),
        "roi_summary": str(roi_summary_path),
        "out_dir": str(out_dir),
        "n_rois": int(len(roi_members)),
        "target_trials": int(args.target_trials),
        "primary_metric": PRIMARY_METRIC,
        "secondary_metrics": list(SECONDARY_METRICS),
    }
    (out_dir / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(
        out_dir=out_dir,
        roi_df=roi_df,
        inventory_df=inventory_df,
        within_df=within_df,
        off_to_on_df=off_to_on_df,
        trial_distance_df=trial_distance_df,
        target_trials=int(args.target_trials),
    )


if __name__ == "__main__":
    main()
