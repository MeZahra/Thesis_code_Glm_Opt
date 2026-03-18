from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
CONNECTIVITY_ROOT = REPO_ROOT / "results" / "connectivity"
GPT_RESULTS_ROOT = CONNECTIVITY_ROOT / "GPT"
GPT_CODE_ROOT = REPO_ROOT / "group_analysis" / "GPT"
TMP_ROI_ROOT = CONNECTIVITY_ROOT / "tmp" / "tmp-roi_edge_network"
TMP_DCM_ROOT = CONNECTIVITY_ROOT / "tmp" / "dynamic_causal_modeling"
ROI_EDGE_RESULTS_ROOT = CONNECTIVITY_ROOT / "roi_edge_network"
BEHAVIOR_ROOT = REPO_ROOT / "results" / "behave_vs_bold"
RAW_TASK_BEHAVIOR_ROOT = Path("/Data/zahra/behaviour")

PRIMARY_METRIC = "mutual_information_ksg"
BENCHMARK_METRICS = (
    "mutual_information_ksg",
    "mutual_information",
    "linear_correlation_network",
    "partial_correlation",
)
SESSION_LABELS = {1: "OFF", 2: "ON"}
CIRCUIT_BASE_ROIS = [
    "Precentral",
    "Dorsolateral Prefrontal Cortex",
    "Basal Ganglia (relative)",
    "Thalamus",
    "Cerebellum",
]
DEFAULT_DROP_LABEL_PATTERNS = ("brain stem", "brain-stem", "cerebral white matter")
EXCLUDED_BASE_ROIS = (
    "Brain-Stem (relative)",
    "Cerebral White Matter (relative)",
)
ANATOMICAL_SYSTEM_ORDER = [
    "cognitive_control",
    "motor_sensorimotor",
    "subcortical_relay",
    "limbic_memory",
    "posterior_perceptual",
    "other_relative",
]
TASK_BEHAVIOR_COLUMN_SPECS = (
    {"index": 0, "label": "1/PT", "key": "task_1_pt"},
    {"index": 1, "label": "1/RT", "key": "task_1_rt"},
    {"index": 2, "label": "1/MT", "key": "task_1_mt"},
    {"index": 3, "label": "1/(RT+MT)", "key": "task_1_rt_mt"},
    {"index": 4, "label": "Vmax", "key": "task_vmax"},
    {"index": 5, "label": "Pmax", "key": "task_pmax"},
)
DEPENDENT_CONSECUTIVE_BEHAVIOR_COLS = (
    "behavior_vigor_delta",
    "behavior_lag1_corr_delta",
    "behavior_consistency_improvement_delta",
)
RAW_TASK_BEHAVIOR_FILE_RE = re.compile(
    r"^PSPD(?P<subject>\d+)_ses_(?P<session>\d+)_run_(?P<run>\d+)\.npy$"
)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def count_dependent_consecutive_behavior_metrics(behavior_cols: Iterable[str]) -> int:
    dependent = set(DEPENDENT_CONSECUTIVE_BEHAVIOR_COLS)
    return sum(col in dependent for col in behavior_cols)


def behavior_subset_passes_dependency_rule(
    behavior_cols: Iterable[str],
    max_dependent_consecutive_metrics: int = 1,
) -> bool:
    return (
        count_dependent_consecutive_behavior_metrics(behavior_cols)
        <= int(max_dependent_consecutive_metrics)
    )


def state_from_session(session: int) -> str:
    return SESSION_LABELS[int(session)]


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")
    return slug.lower()


def base_roi_name(label: str) -> str:
    return re.sub(r"^[LR]\s+", "", label).strip()


def infer_anatomical_system(base_roi: str) -> str:
    if base_roi in {
        "Dorsolateral Prefrontal Cortex",
        "vmPFC / dmPFC (Control & monitoring)",
        "Inferior Frontal Gyrus",
        "Cingulate Cortex",
    }:
        return "cognitive_control"
    if base_roi in {"Precentral", "Parietal Cortex", "Cerebellum"}:
        return "motor_sensorimotor"
    if base_roi in {"Basal Ganglia (relative)", "Thalamus"}:
        return "subcortical_relay"
    if base_roi in {"Amygdala", "Hippocampus (relative)"}:
        return "limbic_memory"
    if base_roi in {"Occipital Cortex (relative)", "Temporal Cortex"}:
        return "posterior_perceptual"
    return "other_relative"


def load_subject_session_split_summary() -> pd.DataFrame:
    path = CONNECTIVITY_ROOT / "data" / "subject_session_split_summary.csv"
    df = pd.read_csv(path)
    df["session"] = df["session"].astype(int)
    df["state"] = df["session"].map(state_from_session)
    return df


def load_metric_matrix(
    subject: str,
    session: int,
    metric: str,
) -> tuple[np.ndarray, list[str]]:
    candidate_dirs = [
        TMP_ROI_ROOT / "advanced_metrics" / f"{subject}_ses-{int(session)}" / metric,
        TMP_ROI_ROOT / "tmp" / f"{subject}_ses-{int(session)}" / metric,
    ]
    for metric_dir in candidate_dirs:
        matrix_path = metric_dir / f"{metric}.npy"
        labels_path = metric_dir / f"{metric}_connectome.labels.txt"
        if matrix_path.exists() and labels_path.exists():
            matrix = np.load(matrix_path)
            labels = labels_path.read_text(encoding="utf-8").splitlines()
            return matrix, labels
    raise FileNotFoundError(
        f"Could not locate {metric!r} for {subject} session {session} in known metric roots."
    )


def list_paired_subjects_for_metric(metric: str = PRIMARY_METRIC) -> list[str]:
    roots = [TMP_ROI_ROOT / "advanced_metrics", TMP_ROI_ROOT / "tmp"]
    session_1 = set()
    session_2 = set()
    for root in roots:
        for session_dir in root.glob("sub-pd*_ses-*"):
            metric_dir = session_dir / metric
            if not metric_dir.exists():
                continue
            subject = session_dir.name.split("_ses-")[0]
            session = session_dir.name.split("_ses-")[1]
            if session == "1" and (metric_dir / f"{metric}.npy").exists():
                session_1.add(subject)
            if session == "2" and (metric_dir / f"{metric}.npy").exists():
                session_2.add(subject)
    return sorted(session_1 & session_2)


def drop_labels_and_matrix(
    labels: list[str],
    matrix: np.ndarray,
    drop_patterns: Iterable[str] = DEFAULT_DROP_LABEL_PATTERNS,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    patterns = [str(pattern).strip().lower() for pattern in drop_patterns if str(pattern).strip()]
    keep_mask = np.ones(len(labels), dtype=bool)
    for idx, label in enumerate(labels):
        label_lower = str(label).lower()
        if any(pattern in label_lower for pattern in patterns):
            keep_mask[idx] = False
    kept_labels = [label for label, keep in zip(labels, keep_mask) if keep]
    keep_idx = np.where(keep_mask)[0]
    trimmed = np.asarray(matrix)[np.ix_(keep_idx, keep_idx)]
    return trimmed, kept_labels, keep_idx


def sanitize_matrix(
    matrix: np.ndarray,
    make_symmetric: bool = True,
    zero_diagonal: bool = True,
) -> np.ndarray:
    out = np.asarray(matrix, dtype=float).copy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    if make_symmetric:
        out = 0.5 * (out + out.T)
    if zero_diagonal:
        np.fill_diagonal(out, 0.0)
    return out


def aggregate_matrix_by_base_roi(
    matrix: np.ndarray,
    labels: list[str],
    include_base_rois: Iterable[str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, list[int]]]:
    base_to_indices: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        base = base_roi_name(label)
        base_to_indices.setdefault(base, []).append(idx)
    if include_base_rois is None:
        base_labels = sorted(base_to_indices)
    else:
        base_labels = [base for base in include_base_rois if base in base_to_indices]
    aggregated = np.zeros((len(base_labels), len(base_labels)), dtype=float)
    for row_idx, row_base in enumerate(base_labels):
        row_indices = base_to_indices[row_base]
        for col_idx, col_base in enumerate(base_labels):
            col_indices = base_to_indices[col_base]
            block = matrix[np.ix_(row_indices, col_indices)]
            aggregated[row_idx, col_idx] = float(np.nanmean(block))
    return aggregated, base_labels, base_to_indices


def aggregate_vector_by_base_roi(
    values: np.ndarray,
    labels: list[str],
    include_base_rois: Iterable[str] | None = None,
) -> pd.Series:
    mapping: dict[str, list[float]] = {}
    for value, label in zip(values, labels):
        mapping.setdefault(base_roi_name(label), []).append(float(value))
    if include_base_rois is None:
        keys = sorted(mapping)
    else:
        keys = [base for base in include_base_rois if base in mapping]
    return pd.Series({base: float(np.nanmean(mapping[base])) for base in keys})


def upper_triangle_mean(matrix: np.ndarray) -> float:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] < 2:
        return float("nan")
    iu = np.triu_indices_from(matrix, k=1)
    values = matrix[iu]
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def mean_between_sets(
    matrix: np.ndarray,
    labels: list[str],
    set_a: Iterable[str],
    set_b: Iterable[str],
) -> float:
    index_a = [idx for idx, label in enumerate(labels) if label in set(set_a)]
    index_b = [idx for idx, label in enumerate(labels) if label in set(set_b)]
    if not index_a or not index_b:
        return float("nan")
    if set(index_a) == set(index_b):
        block = matrix[np.ix_(index_a, index_a)]
        return upper_triangle_mean(block)
    block = matrix[np.ix_(index_a, index_b)]
    return float(np.nanmean(block))


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    if not np.any(valid_mask):
        return q
    pv = p[valid_mask]
    order = np.argsort(pv)
    ranked = pv[order]
    m = ranked.size
    scaled = ranked * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    reordered = np.empty_like(scaled)
    reordered[order] = scaled
    q[valid_mask] = reordered
    return q


def cohen_dz(delta: Iterable[float]) -> float:
    values = np.asarray(list(delta), dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    sd = float(np.std(values, ddof=1))
    if sd == 0.0:
        return 0.0
    return float(np.mean(values) / sd)


def exact_sign_flip_pvalue(delta: Iterable[float]) -> float:
    values = np.asarray(list(delta), dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan")
    if np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(np.mean(values)))
    if n <= 16:
        combos = 1 << n
        flips = (
            ((np.arange(combos)[:, None] >> np.arange(n)) & 1).astype(float) * 2.0
            - 1.0
        )
        null_means = np.abs((flips * values[None, :]).mean(axis=1))
        return float(np.mean(null_means >= observed - 1e-12))
    rng = np.random.default_rng(0)
    n_samples = 50000
    flips = rng.choice([-1.0, 1.0], size=(n_samples, n))
    null_means = np.abs((flips * values[None, :]).mean(axis=1))
    return float((np.sum(null_means >= observed - 1e-12) + 1) / (n_samples + 1))


def paired_delta_stats(delta: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(delta), dtype=float)
    values = values[np.isfinite(values)]
    summary = {
        "n_subjects": int(values.size),
        "mean_delta": float(np.mean(values)) if values.size else float("nan"),
        "median_delta": float(np.median(values)) if values.size else float("nan"),
        "std_delta": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
        "cohen_dz": cohen_dz(values),
        "p_signflip": exact_sign_flip_pvalue(values),
        "p_wilcoxon": float("nan"),
    }
    if values.size and not np.allclose(values, 0.0):
        try:
            summary["p_wilcoxon"] = float(
                stats.wilcoxon(values, zero_method="wilcox").pvalue
            )
        except ValueError:
            summary["p_wilcoxon"] = float("nan")
    return summary


def load_dcm_roi_nodes() -> pd.DataFrame:
    path = TMP_DCM_ROOT / "group_level" / "roi_nodes_used.csv"
    return pd.read_csv(path)


def load_dcm_labels() -> list[str]:
    df = load_dcm_roi_nodes()
    return df["node_name"].tolist()


def load_subject_dcm_matrix(subject: str, matrix_name: str) -> np.ndarray:
    path = TMP_DCM_ROOT / "subject_level" / subject / matrix_name
    return np.load(path)


def load_subject_dcm_vector(subject: str, vector_name: str) -> np.ndarray:
    path = TMP_DCM_ROOT / "subject_level" / subject / vector_name
    return np.load(path)


def list_paired_dcm_subjects() -> list[str]:
    root = TMP_DCM_ROOT / "subject_level"
    subjects = []
    for subject_dir in root.glob("sub-pd*"):
        if (
            (subject_dir / "dcm_session1_off.npy").exists()
            and (subject_dir / "dcm_session2_on.npy").exists()
        ):
            subjects.append(subject_dir.name)
    return sorted(subjects)


def find_single_behavior_file(pattern: str) -> Path:
    matches = sorted(BEHAVIOR_ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No behavior file matched {pattern!r}")
    return matches[0]


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return float("nan")
    filtered_values = values[mask].astype(float)
    filtered_weights = weights[mask].astype(float)
    if float(filtered_weights.sum()) == 0.0:
        return float(filtered_values.mean())
    return float(np.average(filtered_values, weights=filtered_weights))


def load_behavior_session_summary() -> pd.DataFrame:
    run_path = find_single_behavior_file("*_sub_ses_run_projection_behavior_metrics.csv")
    run_df = pd.read_csv(run_path)
    run_df = run_df.rename(columns={"sub_tag": "subject", "ses": "session"})
    run_df["session"] = run_df["session"].astype(int)

    grouped_rows = []
    for (subject, session), group in run_df.groupby(["subject", "session"], sort=True):
        weights = group["n_trials_paired_finite"].fillna(0)
        grouped_rows.append(
            {
                "subject": subject,
                "session": int(session),
                "state": state_from_session(int(session)),
                "n_runs_behavior": int(group.shape[0]),
                "n_trials_behavior": int(weights.sum()),
                "projection_metric": weighted_mean(
                    group["adjacent_diff_ratio_sum_projection"], weights
                ),
                "behavior_vigor": weighted_mean(
                    group["adjacent_diff_ratio_sum_behavior_col2"], weights
                ),
            }
        )
    summary = pd.DataFrame(grouped_rows)

    consecutive_path = find_single_behavior_file(
        "*_sub_session12_consecutive_trial_metrics.csv"
    )
    consecutive_df = pd.read_csv(consecutive_path).rename(columns={"sub_tag": "subject"})
    consecutive_df = consecutive_df[
        [
            "subject",
            "session_a_consecutive_mad_error",
            "session_b_consecutive_mad_error",
            "consecutive_mad_error_diff_session_b_minus_a",
            "session_a_consecutive_lag1_corr",
            "session_b_consecutive_lag1_corr",
            "consecutive_lag1_corr_diff_session_b_minus_a",
        ]
    ]

    j_path = find_single_behavior_file("*_sub_session12_pooled_j_stats.csv")
    j_df = pd.read_csv(j_path).rename(columns={"sub_tag": "subject"})
    j_df = j_df[
        [
            "subject",
            "session_a_j_projection",
            "session_b_j_projection",
            "j_projection_diff_session_b_minus_a",
        ]
    ]

    merged = summary.merge(consecutive_df, on="subject", how="left").merge(
        j_df, on="subject", how="left"
    )
    return merged.sort_values(["subject", "session"]).reset_index(drop=True)


def load_behavior_deltas() -> pd.DataFrame:
    session_summary = load_behavior_session_summary()
    paired = []
    for subject, group in session_summary.groupby("subject", sort=True):
        sessions = sorted(group["session"].unique().tolist())
        if sessions != [1, 2]:
            continue
        off = group[group["session"] == 1].iloc[0]
        on = group[group["session"] == 2].iloc[0]
        paired.append(
            {
                "subject": subject,
                "behavior_vigor_off": off["behavior_vigor"],
                "behavior_vigor_on": on["behavior_vigor"],
                "behavior_vigor_delta": on["behavior_vigor"] - off["behavior_vigor"],
                "projection_metric_off": off["projection_metric"],
                "projection_metric_on": on["projection_metric"],
                "projection_metric_delta": on["projection_metric"] - off["projection_metric"],
                "n_trials_behavior_off": int(off["n_trials_behavior"]),
                "n_trials_behavior_on": int(on["n_trials_behavior"]),
                "consecutive_mad_error_off": float(off["session_a_consecutive_mad_error"]),
                "consecutive_mad_error_on": float(off["session_b_consecutive_mad_error"]),
                "consecutive_mad_error_delta": float(
                    off["consecutive_mad_error_diff_session_b_minus_a"]
                ),
                "behavior_consistency_improvement_delta": float(
                    -off["consecutive_mad_error_diff_session_b_minus_a"]
                ),
                "behavior_lag1_corr_off": float(off["session_a_consecutive_lag1_corr"]),
                "behavior_lag1_corr_on": float(off["session_b_consecutive_lag1_corr"]),
                "behavior_lag1_corr_delta": float(
                    off["consecutive_lag1_corr_diff_session_b_minus_a"]
                ),
                "j_projection_off": float(off["session_a_j_projection"]),
                "j_projection_on": float(off["session_b_j_projection"]),
                "j_projection_delta": float(off["j_projection_diff_session_b_minus_a"]),
            }
        )
    return pd.DataFrame(paired).sort_values("subject").reset_index(drop=True)


def _subject_from_behavior_digits(subject_digits: str) -> str:
    return f"sub-pd{int(subject_digits):03d}"


def load_task_behavior_session_summary(
    behavior_root: Path = RAW_TASK_BEHAVIOR_ROOT,
) -> pd.DataFrame:
    behavior_root = Path(behavior_root)
    run_rows = []
    for path in sorted(behavior_root.glob("PSPD*_ses_*_run_*.npy")):
        match = RAW_TASK_BEHAVIOR_FILE_RE.match(path.name)
        if match is None:
            continue
        subject = _subject_from_behavior_digits(match.group("subject"))
        session = int(match.group("session"))
        run = int(match.group("run"))
        values = np.asarray(np.load(path), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < len(TASK_BEHAVIOR_COLUMN_SPECS):
            raise ValueError(f"Unexpected raw behavior shape in {path}: {values.shape}")

        row = {
            "subject": subject,
            "session": session,
            "run": run,
            "state": state_from_session(session),
            "n_trials_raw_behavior": int(values.shape[0]),
        }
        for spec in TASK_BEHAVIOR_COLUMN_SPECS:
            column = values[:, int(spec["index"])]
            finite = column[np.isfinite(column)]
            row[f"{spec['key']}_mean"] = float(np.mean(finite)) if finite.size else float("nan")
            row[f"{spec['key']}_n_finite"] = int(finite.size)
        run_rows.append(row)

    run_df = pd.DataFrame(run_rows)
    session_rows = []
    for (subject, session), group in run_df.groupby(["subject", "session"], sort=True):
        row = {
            "subject": subject,
            "session": int(session),
            "state": state_from_session(int(session)),
            "n_runs_raw_behavior": int(group.shape[0]),
            "n_trials_raw_behavior": int(group["n_trials_raw_behavior"].sum()),
        }
        for spec in TASK_BEHAVIOR_COLUMN_SPECS:
            mean_col = f"{spec['key']}_mean"
            count_col = f"{spec['key']}_n_finite"
            row[mean_col] = weighted_mean(group[mean_col], group[count_col])
            row[count_col] = int(group[count_col].sum())
        session_rows.append(row)

    return pd.DataFrame(session_rows).sort_values(["subject", "session"]).reset_index(drop=True)


def load_task_behavior_deltas(
    behavior_root: Path = RAW_TASK_BEHAVIOR_ROOT,
) -> pd.DataFrame:
    session_summary = load_task_behavior_session_summary(behavior_root=behavior_root)
    paired = []
    for subject, group in session_summary.groupby("subject", sort=True):
        sessions = sorted(group["session"].unique().tolist())
        if sessions != [1, 2]:
            continue
        off = group[group["session"] == 1].iloc[0]
        on = group[group["session"] == 2].iloc[0]
        row = {
            "subject": subject,
            "n_trials_raw_behavior_off": int(off["n_trials_raw_behavior"]),
            "n_trials_raw_behavior_on": int(on["n_trials_raw_behavior"]),
        }
        for spec in TASK_BEHAVIOR_COLUMN_SPECS:
            key = str(spec["key"])
            mean_col = f"{key}_mean"
            row[f"{key}_off"] = float(off[mean_col])
            row[f"{key}_on"] = float(on[mean_col])
            row[f"{key}_delta"] = float(on[mean_col] - off[mean_col])
        paired.append(row)

    return pd.DataFrame(paired).sort_values("subject").reset_index(drop=True)


def recursive_file_inventory(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def to_serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        if math.isnan(float(value)):
            return None
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value
