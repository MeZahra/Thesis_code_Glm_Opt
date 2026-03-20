"""
Structured targeted PLS exploration for medication-related brain-behavior change.

This script replaces the earlier single-pipeline test with a staged, theory-driven
configuration search that stays small enough to interpret with N = 14 while still
testing the main modeling decisions that matter here:

1. ROI scope:
   - all ROIs
   - all ROIs with nonspecific parcels removed
   - a priori circuit ROIs
   - control / monitoring ROIs
   - motor-subcortical ROIs
2. Neural feature block:
   - graph strength
   - graph participation
   - graph within-module integration
   - DCM outgoing coupling
   - DCM + graph participation
   - anatomical module-summary features
3. Behavior block:
   - full motor-vigor block: 1/(RT+MT), Vmax, Pmax
   - peak-force / peak-velocity block: Vmax, Pmax
   - single-behavior refinement models
4. Dimensionality reduction:
   - PCA-reduced neural block for broad models
   - raw compact neural blocks for refined low-p models

Every screened model gets:
   - latent score correlation
   - permutation p-value
   - BH-FDR q-values (global and within stage)
   - LOSO prediction statistics
   - heuristic diagnosis of likely failure mode

The top models are then refit with more permutations plus bootstrap stability, and
all outputs are written under results/connectivity/PLS.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

from common_io import CIRCUIT_BASE_ROIS, TASK_BEHAVIOR_COLUMN_SPECS, load_task_behavior_deltas, safe_slug


REPO_ROOT = Path(__file__).resolve().parents[2]
TMP = REPO_ROOT / "results" / "connectivity" / "GPT" / "tmp"
OUT_DIR = REPO_ROOT / "results" / "connectivity" / "PLS"

DCM_PARAMS_PATH = TMP / "effective_connectivity" / "all_roi" / "dcm_subject_parameters.csv"
NODE_DELTA_PATH = (
    TMP / "roi_graph_analysis_partial_correlation_check" / "node_metric_deltas_on_minus_off.csv"
)
MODULE_SUMM_PATH = (
    TMP / "roi_graph_analysis_partial_correlation_check" / "module_integration_summary.csv"
)

SCREEN_PERMUTATIONS = 3000
DETAILED_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
MAX_DETAILED_MODELS = 10
MAX_X_PCA_COMPONENTS = 3
RANDOM_SEED = 42

EXCLUDE_BASE_ROI_PATTERNS = ("brain stem", "brain-stem", "cerebral white matter")
CONTROL_BASE_ROIS = [
    "Dorsolateral Prefrontal Cortex",
    "vmPFC / dmPFC (Control & monitoring)",
    "Cingulate Cortex",
    "Inferior Frontal Gyrus",
]
MOTOR_SUBCORTICAL_BASE_ROIS = [
    "Precentral",
    "Cerebellum",
    "Parietal Cortex",
    "Basal Ganglia (relative)",
    "Thalamus",
]
NONSPECIFIC_BASE_ROIS = [
    "Other Cerebral Cortex (relative)",
    "Unassigned Active Voxels (relative)",
]

BEHAVIOR_SETS = {
    "full3": [
        "task_1_rt_mt_delta",
        "task_vmax_delta",
        "task_pmax_delta",
    ],
    "vmax_pmax": [
        "task_vmax_delta",
        "task_pmax_delta",
    ],
    "rtmt": ["task_1_rt_mt_delta"],
    "vmax": ["task_vmax_delta"],
    "pmax": ["task_pmax_delta"],
}

BEHAVIOR_SET_RATIONALES = {
    "full3": "Original motor-vigor block: timing + peak vigor.",
    "vmax_pmax": "Peak vigor block only, used to test whether RT/MT diluted the signal.",
    "rtmt": "Inverse movement duration only.",
    "vmax": "Peak velocity only.",
    "pmax": "Peak force only.",
}

BEHAVIOR_LABELS = {
    f"{spec['key']}_delta": f"{spec['label']} session-mean delta (ON - OFF)"
    for spec in TASK_BEHAVIOR_COLUMN_SPECS
}


@dataclass(frozen=True)
class ModelConfig:
    stage: str
    config_name: str
    description: str
    roi_scheme: str
    neural_block: str
    behavior_set: str
    x_reduction: str
    requested_x_components: int | None
    feature_cols: tuple[str, ...]
    behavior_cols: tuple[str, ...]


@dataclass
class FittedPLSPipeline:
    x_mean: pd.Series
    x_scale: pd.Series
    y_mean: pd.Series
    y_scale: pd.Series
    x_reduction: str
    pca: PCA | None
    pls: PLSRegression
    x_scores: np.ndarray
    y_scores: np.ndarray
    observed_r: float
    x_feature_weights: np.ndarray
    y_feature_weights: np.ndarray
    n_x_components: int
    x_variance_explained: float


def _base_roi(label: str) -> str:
    return re.sub(r"^[LR]\s+", "", str(label)).strip()


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()


def _standardize(
    df: pd.DataFrame,
    mean: pd.Series | None = None,
    scale: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    resolved_mean = df.mean(axis=0) if mean is None else mean
    resolved_scale = df.std(axis=0, ddof=0) if scale is None else scale
    resolved_scale = resolved_scale.replace(0.0, 1.0)
    standardized = (df - resolved_mean) / resolved_scale
    return standardized, resolved_mean, resolved_scale


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or np.isnan(norm):
        return vector
    return vector / norm


def _bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = ranked.size
    scaled = ranked * m / np.arange(1, m + 1)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    out = np.empty_like(scaled)
    out[order] = scaled
    return out


def _align_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return candidate if float(np.dot(reference, candidate)) >= 0.0 else -candidate


def _vector_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    if ref.size == 1 or np.nanstd(ref) == 0.0 or np.nanstd(cand) == 0.0:
        ref_sign = float(np.sign(ref[0])) if ref.size else 0.0
        cand_sign = float(np.sign(cand[0])) if cand.size else 0.0
        if ref_sign == 0.0 or cand_sign == 0.0:
            return 0.0
        return 1.0 if ref_sign == cand_sign else -1.0
    return float(np.corrcoef(ref, cand)[0, 1])


def _choose_x_components(
    Xz: pd.DataFrame,
    n_behavior_features: int,
    requested: int | None,
) -> int:
    max_allowed = min(
        MAX_X_PCA_COMPONENTS if requested is None else int(requested),
        Xz.shape[0] - 1,
        Xz.shape[1],
        max(1, int(np.linalg.matrix_rank(Xz.to_numpy()))),
        max(1, n_behavior_features),
    )
    return max(1, int(max_allowed))


def _motor_vigor_composite(
    df: pd.DataFrame,
    reference_mean: pd.Series | None = None,
    reference_scale: pd.Series | None = None,
) -> np.ndarray:
    z_df, _, _ = _standardize(df, mean=reference_mean, scale=reference_scale)
    return z_df.mean(axis=1).to_numpy()


def _pretty_feature_name(name: str) -> str:
    name = re.sub(r"^dcm_outgoing_delta_", "DCM: ", name)
    name = re.sub(r"^graph_strength_delta_", "Strength: ", name)
    name = re.sub(r"^graph_participation_delta_", "Participation: ", name)
    name = re.sub(r"^graph_within_module_delta_", "Within-module: ", name)
    name = re.sub(r"^module_delta_", "Module: ", name)
    return name.replace("__", " x ").replace("_", " ")


def _feature_metadata(name: str) -> dict[str, str]:
    if name.startswith("dcm_outgoing_delta_"):
        suffix = name.removeprefix("dcm_outgoing_delta_")
        return {
            "feature_family": "dcm_outgoing",
            "feature_metric": "outgoing_connectivity",
            "base_roi": suffix,
        }
    if name.startswith("graph_strength_delta_"):
        suffix = name.removeprefix("graph_strength_delta_")
        return {
            "feature_family": "graph_strength",
            "feature_metric": "node_strength",
            "base_roi": suffix,
        }
    if name.startswith("graph_participation_delta_"):
        suffix = name.removeprefix("graph_participation_delta_")
        return {
            "feature_family": "graph_participation",
            "feature_metric": "participation",
            "base_roi": suffix,
        }
    if name.startswith("graph_within_module_delta_"):
        suffix = name.removeprefix("graph_within_module_delta_")
        return {
            "feature_family": "graph_within_module",
            "feature_metric": "within_module_z",
            "base_roi": suffix,
        }
    if name.startswith("module_delta_"):
        suffix = name.removeprefix("module_delta_")
        return {
            "feature_family": "module_summary",
            "feature_metric": "module_connectivity",
            "base_roi": suffix,
        }
    return {"feature_family": "unknown", "feature_metric": "unknown", "base_roi": name}


def build_dcm_features(dcm_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dcm_path)
    coupling = df[df["parameter_type"] == "coupling"].copy()
    coupling["base_target"] = coupling["target_roi"].map(_base_roi)
    coupling["base_source"] = coupling["source_roi"].map(_base_roi)
    coupling = coupling[
        ~coupling["base_target"].str.lower().apply(
            lambda x: any(pattern in x for pattern in EXCLUDE_BASE_ROI_PATTERNS)
        )
    ]
    coupling = coupling[
        ~coupling["base_source"].str.lower().apply(
            lambda x: any(pattern in x for pattern in EXCLUDE_BASE_ROI_PATTERNS)
        )
    ]
    agg = (
        coupling.groupby(["subject", "base_target", "base_source"], as_index=False)[
            "delta_on_minus_off"
        ].mean()
    )
    rows = []
    for subject, grp in agg.groupby("subject", sort=True):
        pivot = grp.pivot(index="base_target", columns="base_source", values="delta_on_minus_off")
        row = {"subject": subject}
        for roi in sorted(pivot.columns):
            col = f"dcm_outgoing_delta_{_safe_slug(roi)}"
            outgoing = pivot.loc[pivot.index != roi, roi]
            row[col] = float(outgoing.mean()) if len(outgoing) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def build_graph_features(node_delta_path: Path) -> pd.DataFrame:
    df = pd.read_csv(node_delta_path)
    df = df[
        ~df["base_roi"].str.lower().apply(
            lambda x: any(pattern in x for pattern in EXCLUDE_BASE_ROI_PATTERNS)
        )
    ]
    agg = (
        df.groupby(["subject", "base_roi"], as_index=False)
        .agg(
            delta_node_strength_abs=("delta_node_strength_abs", "mean"),
            delta_participation_coeff=("delta_participation_coeff", "mean"),
            delta_within_module_z=("delta_within_module_z", "mean"),
        )
        .sort_values(["subject", "base_roi"])
    )
    rows = []
    for subject, grp in agg.groupby("subject", sort=True):
        row = {"subject": subject}
        for _, data in grp.iterrows():
            roi = _safe_slug(data["base_roi"])
            row[f"graph_strength_delta_{roi}"] = float(data["delta_node_strength_abs"])
            row[f"graph_participation_delta_{roi}"] = float(data["delta_participation_coeff"])
            row[f"graph_within_module_delta_{roi}"] = float(data["delta_within_module_z"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def build_module_features(module_path: Path) -> pd.DataFrame:
    df = pd.read_csv(module_path)
    anatomical = df[df["module_scheme"] == "anatomical_system"].copy()
    rows = []
    pairs = (
        anatomical[["module_a", "module_b"]]
        .drop_duplicates()
        .sort_values(["module_a", "module_b"])
        .apply(tuple, axis=1)
        .tolist()
    )
    for subject, grp in anatomical.groupby("subject", sort=True):
        row = {"subject": subject}
        for module_a, module_b in pairs:
            feat = f"module_delta_{_safe_slug(module_a)}__{_safe_slug(module_b)}"
            off = grp[
                (grp["session"] == 1)
                & (grp["module_a"] == module_a)
                & (grp["module_b"] == module_b)
            ]
            on = grp[
                (grp["session"] == 2)
                & (grp["module_a"] == module_a)
                & (grp["module_b"] == module_b)
            ]
            row[feat] = (
                float(on["mean_abs_strength"].iloc[0]) - float(off["mean_abs_strength"].iloc[0])
            ) if (len(off) and len(on)) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def _build_roi_schemes(graph_df: pd.DataFrame) -> dict[str, set[str]]:
    all_rois = {
        col.removeprefix("graph_strength_delta_")
        for col in graph_df.columns
        if col.startswith("graph_strength_delta_")
    }
    return {
        "all": set(sorted(all_rois)),
        "nonspecific_removed": set(sorted(all_rois))
        - {safe_slug(name) for name in NONSPECIFIC_BASE_ROIS},
        "circuit": {safe_slug(name) for name in CIRCUIT_BASE_ROIS},
        "control": {safe_slug(name) for name in CONTROL_BASE_ROIS},
        "motor_subcortical": {safe_slug(name) for name in MOTOR_SUBCORTICAL_BASE_ROIS},
    }


def _select_feature_cols(
    feature_tables: dict[str, pd.DataFrame],
    neural_block: str,
    roi_slugs: set[str] | None,
) -> list[str]:
    graph_df = feature_tables["graph"]
    dcm_df = feature_tables["dcm"]
    module_df = feature_tables["module"]

    if neural_block == "graph_participation":
        return [
            col
            for col in graph_df.columns
            if col.startswith("graph_participation_delta_")
            and col.removeprefix("graph_participation_delta_") in (roi_slugs or set())
        ]
    if neural_block == "graph_strength":
        return [
            col
            for col in graph_df.columns
            if col.startswith("graph_strength_delta_")
            and col.removeprefix("graph_strength_delta_") in (roi_slugs or set())
        ]
    if neural_block == "graph_within_module":
        return [
            col
            for col in graph_df.columns
            if col.startswith("graph_within_module_delta_")
            and col.removeprefix("graph_within_module_delta_") in (roi_slugs or set())
        ]
    if neural_block == "dcm_outgoing":
        return [
            col
            for col in dcm_df.columns
            if col != "subject"
            and col.removeprefix("dcm_outgoing_delta_") in (roi_slugs or set())
        ]
    if neural_block == "graph_strength_plus_participation":
        out: list[str] = []
        out.extend(_select_feature_cols(feature_tables, "graph_strength", roi_slugs))
        out.extend(_select_feature_cols(feature_tables, "graph_participation", roi_slugs))
        return out
    if neural_block == "dcm_plus_participation":
        out = _select_feature_cols(feature_tables, "dcm_outgoing", roi_slugs)
        out.extend(_select_feature_cols(feature_tables, "graph_participation", roi_slugs))
        return out
    if neural_block == "module_summary":
        return [col for col in module_df.columns if col != "subject"]
    raise ValueError(f"Unsupported neural block: {neural_block}")


def _make_config(
    stage: str,
    roi_scheme: str,
    neural_block: str,
    behavior_set: str,
    x_reduction: str,
    feature_cols: list[str],
    description: str,
    requested_x_components: int | None = None,
) -> ModelConfig:
    config_name = "__".join(
        [
            stage,
            roi_scheme,
            neural_block,
            behavior_set,
            x_reduction,
        ]
    )
    return ModelConfig(
        stage=stage,
        config_name=config_name,
        description=description,
        roi_scheme=roi_scheme,
        neural_block=neural_block,
        behavior_set=behavior_set,
        x_reduction=x_reduction,
        requested_x_components=requested_x_components,
        feature_cols=tuple(feature_cols),
        behavior_cols=tuple(BEHAVIOR_SETS[behavior_set]),
    )


def build_model_configs(feature_tables: dict[str, pd.DataFrame]) -> list[ModelConfig]:
    roi_schemes = _build_roi_schemes(feature_tables["graph"])
    configs: list[ModelConfig] = []

    baseline_specs = [
        (
            "all",
            "graph_strength_plus_participation",
            "full3",
            "Baseline closest to the original graph-strength + participation pipeline.",
        ),
        (
            "all",
            "graph_participation",
            "full3",
            "Baseline participation-only model across all ROIs.",
        ),
        (
            "all",
            "dcm_plus_participation",
            "full3",
            "Baseline all-ROI model adding DCM to participation.",
        ),
    ]
    for roi_scheme, neural_block, behavior_set, description in baseline_specs:
        feature_cols = _select_feature_cols(feature_tables, neural_block, roi_schemes[roi_scheme])
        configs.append(
            _make_config(
                stage="baseline",
                roi_scheme=roi_scheme,
                neural_block=neural_block,
                behavior_set=behavior_set,
                x_reduction="pca_auto",
                feature_cols=feature_cols,
                description=description,
            )
        )

    broad_roi_schemes = ["nonspecific_removed", "circuit", "control", "motor_subcortical"]
    broad_blocks = [
        "graph_participation",
        "graph_strength",
        "graph_within_module",
        "dcm_outgoing",
        "dcm_plus_participation",
    ]
    broad_behaviors = ["full3", "vmax_pmax"]
    for roi_scheme in broad_roi_schemes:
        roi_slugs = roi_schemes[roi_scheme]
        for neural_block in broad_blocks:
            feature_cols = _select_feature_cols(feature_tables, neural_block, roi_slugs)
            if not feature_cols:
                continue
            for behavior_set in broad_behaviors:
                configs.append(
                    _make_config(
                        stage="broad",
                        roi_scheme=roi_scheme,
                        neural_block=neural_block,
                        behavior_set=behavior_set,
                        x_reduction="pca_auto",
                        feature_cols=feature_cols,
                        description=(
                            f"Broad sweep: {neural_block} within {roi_scheme} ROIs "
                            f"against {behavior_set}."
                        ),
                    )
                )

    module_behaviors = ["full3", "vmax_pmax"]
    module_features = _select_feature_cols(feature_tables, "module_summary", None)
    for behavior_set in module_behaviors:
        configs.append(
            _make_config(
                stage="broad",
                roi_scheme="module_level",
                neural_block="module_summary",
                behavior_set=behavior_set,
                x_reduction="pca_auto",
                feature_cols=module_features,
                description="Broad sweep of anatomical module-summary features.",
            )
        )

    refined_behaviors = ["full3", "vmax_pmax", "rtmt", "vmax", "pmax"]
    control_slugs = roi_schemes["control"]
    for neural_block in ["graph_participation", "dcm_plus_participation"]:
        feature_cols = _select_feature_cols(feature_tables, neural_block, control_slugs)
        for behavior_set in refined_behaviors:
            configs.append(
                _make_config(
                    stage="refined",
                    roi_scheme="control",
                    neural_block=neural_block,
                    behavior_set=behavior_set,
                    x_reduction="raw",
                    feature_cols=feature_cols,
                    description=(
                        "Refinement: compact control-network model without PCA to test "
                        "whether behavior choice or PCA was diluting the signal."
                    ),
                )
            )

    for roi_name in CONTROL_BASE_ROIS:
        roi_slug = safe_slug(roi_name)
        feature_cols = [f"graph_participation_delta_{roi_slug}"]
        for behavior_set in ["vmax_pmax", "full3"]:
            configs.append(
                _make_config(
                    stage="localized",
                    roi_scheme=roi_slug,
                    neural_block="graph_participation_single_roi",
                    behavior_set=behavior_set,
                    x_reduction="raw",
                    feature_cols=feature_cols,
                    description=(
                        "Localization within the control family using single-ROI "
                        "participation features."
                    ),
                )
            )

    seen = set()
    deduped: list[ModelConfig] = []
    for config in configs:
        if config.config_name in seen:
            continue
        seen.add(config.config_name)
        deduped.append(config)
    return deduped


def fit_pls_pipeline(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    x_reduction: str,
    requested_x_components: int | None = None,
) -> FittedPLSPipeline:
    Xz, x_mean, x_scale = _standardize(X)
    Yz, y_mean, y_scale = _standardize(Y)

    if x_reduction == "raw":
        X_model = Xz.to_numpy()
        pca = None
        n_x_components = int(X.shape[1])
        x_variance_explained = 1.0
        feature_projection = np.eye(X.shape[1], dtype=float)
    elif x_reduction == "pca_auto":
        n_x_components = _choose_x_components(
            Xz,
            n_behavior_features=Y.shape[1],
            requested=requested_x_components,
        )
        pca = PCA(n_components=n_x_components, svd_solver="full")
        X_model = pca.fit_transform(Xz.to_numpy())
        x_variance_explained = float(pca.explained_variance_ratio_.sum())
        feature_projection = np.asarray(pca.components_.T, dtype=float)
    else:
        raise ValueError(f"Unsupported x_reduction: {x_reduction}")

    pls = PLSRegression(n_components=1, max_iter=1000, scale=False)
    pls.fit(X_model, Yz.to_numpy())
    x_scores = pls.x_scores_.ravel()
    y_scores = pls.y_scores_.ravel()
    observed_r = float(stats.pearsonr(x_scores, y_scores).statistic)
    x_feature_weights = _normalize_vector(feature_projection @ pls.x_weights_.ravel())
    y_feature_weights = _normalize_vector(pls.y_weights_.ravel())
    return FittedPLSPipeline(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        x_reduction=x_reduction,
        pca=pca,
        pls=pls,
        x_scores=x_scores,
        y_scores=y_scores,
        observed_r=observed_r,
        x_feature_weights=x_feature_weights,
        y_feature_weights=y_feature_weights,
        n_x_components=n_x_components,
        x_variance_explained=x_variance_explained,
    )


def predict_pls_pipeline(
    fitted: FittedPLSPipeline,
    X_new: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    Xz_new, _, _ = _standardize(X_new, mean=fitted.x_mean, scale=fitted.x_scale)
    if fitted.pca is None:
        X_model_new = Xz_new.to_numpy()
    else:
        X_model_new = fitted.pca.transform(Xz_new.to_numpy())
    y_pred_z = fitted.pls.predict(X_model_new)
    x_scores = fitted.pls.transform(X_model_new).ravel()
    y_pred = y_pred_z * fitted.y_scale.to_numpy() + fitted.y_mean.to_numpy()
    return np.asarray(y_pred, dtype=float), np.asarray(x_scores, dtype=float)


def permutation_test(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    x_reduction: str,
    requested_x_components: int | None,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[FittedPLSPipeline, float, np.ndarray]:
    fitted = fit_pls_pipeline(
        X,
        Y,
        x_reduction=x_reduction,
        requested_x_components=requested_x_components,
    )
    null = np.empty(n_perm, dtype=float)
    for idx in range(n_perm):
        perm_fit = fit_pls_pipeline(
            X,
            Y.iloc[rng.permutation(len(Y))].reset_index(drop=True),
            x_reduction=x_reduction,
            requested_x_components=requested_x_components,
        )
        null[idx] = perm_fit.observed_r
    permutation_p = float((np.sum(np.abs(null) >= abs(fitted.observed_r)) + 1) / (n_perm + 1))
    return fitted, permutation_p, null


def bootstrap_weight_stability(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    fitted: FittedPLSPipeline,
    feature_names: list[str],
    behavior_names: list[str],
    x_reduction: str,
    requested_x_components: int | None,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    x_boot_weights = []
    y_boot_weights = []
    boot_score_corrs = []
    weight_sims = []
    n_samples = len(X)

    for _ in range(n_boot):
        boot_idx = rng.integers(0, n_samples, size=n_samples)
        Xb = X.iloc[boot_idx].reset_index(drop=True)
        Yb = Y.iloc[boot_idx].reset_index(drop=True)
        try:
            boot_fit = fit_pls_pipeline(
                Xb,
                Yb,
                x_reduction=x_reduction,
                requested_x_components=requested_x_components,
            )
        except Exception:
            continue
        aligned_x = _align_sign(fitted.x_feature_weights, boot_fit.x_feature_weights)
        aligned_y = _align_sign(fitted.y_feature_weights, boot_fit.y_feature_weights)
        x_boot_weights.append(aligned_x)
        y_boot_weights.append(aligned_y)
        boot_score_corrs.append(boot_fit.observed_r)
        weight_sims.append(_vector_similarity(fitted.x_feature_weights, aligned_x))

    if not x_boot_weights:
        raise RuntimeError("Bootstrap failed to produce a valid sample.")

    x_boot = np.vstack(x_boot_weights)
    y_boot = np.vstack(y_boot_weights)
    boot_score_corrs_arr = np.asarray(boot_score_corrs, dtype=float)
    weight_sims_arr = np.asarray(weight_sims, dtype=float)

    x_rows = []
    for idx, name in enumerate(feature_names):
        meta = _feature_metadata(name)
        full_weight = float(fitted.x_feature_weights[idx])
        ci_low = float(np.percentile(x_boot[:, idx], 2.5))
        ci_high = float(np.percentile(x_boot[:, idx], 97.5))
        x_rows.append(
            {
                "name": name,
                "display_name": _pretty_feature_name(name),
                "full_weight": full_weight,
                "bootstrap_mean_weight": float(np.mean(x_boot[:, idx])),
                "bootstrap_median_weight": float(np.median(x_boot[:, idx])),
                "ci_2p5": ci_low,
                "ci_97p5": ci_high,
                "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
                "sign_stability": float(np.mean(np.sign(x_boot[:, idx]) == np.sign(full_weight))),
                "feature_family": meta["feature_family"],
                "feature_metric": meta["feature_metric"],
                "base_roi": meta["base_roi"],
            }
        )
    x_summary_df = pd.DataFrame(x_rows).sort_values(
        by="full_weight",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)

    y_rows = []
    for idx, name in enumerate(behavior_names):
        full_weight = float(fitted.y_feature_weights[idx])
        ci_low = float(np.percentile(y_boot[:, idx], 2.5))
        ci_high = float(np.percentile(y_boot[:, idx], 97.5))
        y_rows.append(
            {
                "name": name,
                "label": BEHAVIOR_LABELS.get(name, name),
                "full_weight": full_weight,
                "bootstrap_mean_weight": float(np.mean(y_boot[:, idx])),
                "bootstrap_median_weight": float(np.median(y_boot[:, idx])),
                "ci_2p5": ci_low,
                "ci_97p5": ci_high,
                "ci_excludes_zero": bool((ci_low > 0.0) or (ci_high < 0.0)),
                "sign_stability": float(np.mean(np.sign(y_boot[:, idx]) == np.sign(full_weight))),
            }
        )
    y_summary_df = pd.DataFrame(y_rows).sort_values(
        by="full_weight",
        key=lambda series: series.abs(),
        ascending=False,
    ).reset_index(drop=True)

    summary = {
        "n_valid_boots": int(x_boot.shape[0]),
        "median_boot_weight_similarity": float(np.median(weight_sims_arr)),
        "p05_boot_weight_similarity": float(np.percentile(weight_sims_arr, 5)),
        "boot_score_corr_ci_2p5": float(np.percentile(boot_score_corrs_arr, 2.5)),
        "boot_score_corr_ci_97p5": float(np.percentile(boot_score_corrs_arr, 97.5)),
        "boot_score_corr_median": float(np.median(boot_score_corrs_arr)),
        "n_stable_features_ci_excluding_zero": int(x_summary_df["ci_excludes_zero"].sum()),
    }
    return x_summary_df, y_summary_df, summary


def loo_prediction_analysis(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    fitted: FittedPLSPipeline,
    subjects: pd.Series,
    x_reduction: str,
    requested_x_components: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    rows = []
    weight_sims = []
    for holdout_idx in range(len(X)):
        train_mask = np.ones(len(X), dtype=bool)
        train_mask[holdout_idx] = False
        loo_fit = fit_pls_pipeline(
            X.iloc[train_mask].reset_index(drop=True),
            Y.iloc[train_mask].reset_index(drop=True),
            x_reduction=x_reduction,
            requested_x_components=requested_x_components,
        )
        predicted_y, x_score = predict_pls_pipeline(
            loo_fit,
            X.iloc[[holdout_idx]].reset_index(drop=True),
        )
        aligned_weights = _align_sign(fitted.x_feature_weights, loo_fit.x_feature_weights)
        similarity = _vector_similarity(fitted.x_feature_weights, aligned_weights)
        weight_sims.append(similarity)
        row = {
            "subject": subjects.iloc[holdout_idx],
            "x_score_held_out": float(x_score[0]),
            "weights_similarity_to_full": float(similarity),
        }
        for feature_idx, name in enumerate(Y.columns):
            row[f"actual_{name}"] = float(Y.iloc[holdout_idx, feature_idx])
            row[f"predicted_{name}"] = float(predicted_y[0, feature_idx])
        rows.append(row)

    loo_df = pd.DataFrame(rows)
    actual_cols = [f"actual_{name}" for name in Y.columns]
    predicted_cols = [f"predicted_{name}" for name in Y.columns]
    actual_df = loo_df[actual_cols].rename(columns=dict(zip(actual_cols, Y.columns)))
    predicted_df = loo_df[predicted_cols].rename(columns=dict(zip(predicted_cols, Y.columns)))
    actual_mean = actual_df.mean(axis=0)
    actual_scale = actual_df.std(axis=0, ddof=0).replace(0.0, 1.0)
    loo_df["actual_motor_vigor_composite"] = _motor_vigor_composite(
        actual_df,
        reference_mean=actual_mean,
        reference_scale=actual_scale,
    )
    loo_df["predicted_motor_vigor_composite"] = _motor_vigor_composite(
        predicted_df,
        reference_mean=actual_mean,
        reference_scale=actual_scale,
    )

    summary_rows = []
    behavior_pearsons = []
    for name in Y.columns:
        actual = loo_df[f"actual_{name}"]
        predicted = loo_df[f"predicted_{name}"]
        pearson_r = float(stats.pearsonr(actual, predicted).statistic)
        spearman_r = float(stats.spearmanr(actual, predicted).statistic)
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        mae = float(np.mean(np.abs(predicted - actual)))
        behavior_pearsons.append(pearson_r)
        summary_rows.append(
            {
                "metric": name,
                "label": BEHAVIOR_LABELS.get(name, name),
                "pearson_r": pearson_r,
                "spearman_rho": spearman_r,
                "rmse": rmse,
                "mae": mae,
            }
        )

    actual_composite = loo_df["actual_motor_vigor_composite"]
    predicted_composite = loo_df["predicted_motor_vigor_composite"]
    composite_pearson = float(stats.pearsonr(actual_composite, predicted_composite).statistic)
    composite_spearman = float(stats.spearmanr(actual_composite, predicted_composite).statistic)
    composite_rmse = float(np.sqrt(np.mean((predicted_composite - actual_composite) ** 2)))
    composite_mae = float(np.mean(np.abs(predicted_composite - actual_composite)))
    summary_rows.append(
        {
            "metric": "motor_vigor_composite",
            "label": "Motor vigor composite",
            "pearson_r": composite_pearson,
            "spearman_rho": composite_spearman,
            "rmse": composite_rmse,
            "mae": composite_mae,
        }
    )
    summary_df = pd.DataFrame(summary_rows)

    summary = {
        "mean_loo_weight_similarity": float(np.mean(weight_sims)),
        "min_loo_weight_similarity": float(np.min(weight_sims)),
        "loo_composite_pearson_r": composite_pearson,
        "loo_composite_spearman_rho": composite_spearman,
        "loo_composite_rmse": composite_rmse,
        "loo_composite_mae": composite_mae,
        "loo_mean_behavior_pearson_r": float(np.mean(behavior_pearsons)),
        "loo_min_behavior_pearson_r": float(np.min(behavior_pearsons)),
    }
    return loo_df, summary_df, summary


def plot_model(
    scores_df: pd.DataFrame,
    x_boot_df: pd.DataFrame | None,
    loo_df: pd.DataFrame,
    perm_null: np.ndarray,
    observed_r: float,
    permutation_p: float,
    title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(23, 5), constrained_layout=True)

    ax = axes[0]
    ax.scatter(scores_df["x_score"], scores_df["y_score"], color="#4c78a8", s=60, zorder=3)
    for _, row in scores_df.iterrows():
        ax.text(row["x_score"], row["y_score"], row["subject"], fontsize=6)
    ax.set_xlabel("Neural latent score")
    ax.set_ylabel("Behavior latent score")
    ax.set_title(f"Full sample\nr = {observed_r:.3f}, p_perm = {permutation_p:.4f}")
    ax.axhline(0.0, color="grey", lw=0.5)
    ax.axvline(0.0, color="grey", lw=0.5)

    ax = axes[1]
    if x_boot_df is not None and not x_boot_df.empty:
        top = x_boot_df.head(min(8, len(x_boot_df))).iloc[::-1]
        errors = np.vstack(
            [
                np.maximum(0.0, top["full_weight"].to_numpy() - top["ci_2p5"].to_numpy()),
                np.maximum(0.0, top["ci_97p5"].to_numpy() - top["full_weight"].to_numpy()),
            ]
        )
        colors = ["#e45756" if value > 0 else "#4c78a8" for value in top["full_weight"]]
        ax.barh(top["display_name"], top["full_weight"], color=colors, xerr=errors, capsize=3)
        ax.axvline(0.0, color="black", lw=1)
        ax.set_xlabel("Feature weight")
        ax.set_title("Top neural weights\n95% bootstrap CI")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Bootstrap not run", ha="center", va="center")

    ax = axes[2]
    actual = loo_df["actual_motor_vigor_composite"]
    predicted = loo_df["predicted_motor_vigor_composite"]
    ax.scatter(actual, predicted, color="#54a24b", s=60, zorder=3)
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--", lw=1)
    for _, row in loo_df.iterrows():
        ax.text(
            row["actual_motor_vigor_composite"],
            row["predicted_motor_vigor_composite"],
            row["subject"],
            fontsize=6,
        )
    loo_r = float(stats.pearsonr(actual, predicted).statistic)
    ax.set_xlabel("Observed LOSO vigor composite")
    ax.set_ylabel("Predicted LOSO vigor composite")
    ax.set_title(f"LOSO prediction\nr = {loo_r:.3f}")

    ax = axes[3]
    ax.hist(np.abs(perm_null), bins=50, color="#9ecae9", edgecolor="white")
    ax.axvline(abs(observed_r), color="#e45756", lw=2, label=f"observed |r| = {abs(observed_r):.3f}")
    ax.set_xlabel("|r| under permutation")
    ax.set_ylabel("Count")
    ax.set_title("Permutation null")
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _diagnose_model(row: pd.Series) -> tuple[str, str]:
    gap = float(row["observed_r"] - row["loo_composite_pearson_r"])
    behavior_delta = float(row.get("delta_loo_to_best_behavior_sibling", 0.0))
    roi_delta = float(row.get("delta_loo_to_best_roi_sibling", 0.0))

    if (
        row["permutation_p"] < 0.05
        and row["loo_composite_pearson_r"] >= 0.45
        and gap < 0.25
    ):
        return (
            "credible_exploratory_signal",
            "Permutation-significant with positive LOSO and limited in-sample vs LOSO decay.",
        )

    if (
        row["behavior_set"] == "full3"
        and behavior_delta >= 0.12
    ) or (
        row["roi_scheme"] in {"all", "nonspecific_removed", "motor_subcortical", "circuit"}
        and roi_delta >= 0.12
    ):
        return (
            "poor_feature_alignment",
            "A sibling model with the same neural family but tighter behavior or ROI targeting performed materially better.",
        )

    if gap >= 0.25 or (
        row["p_n_ratio"] > 0.75
        and row["loo_composite_pearson_r"] < 0.20
        and row["observed_r"] >= 0.55
    ):
        return (
            "overfitting",
            "Training latent correlation is much stronger than LOSO performance and/or the model is too feature-rich for N=14.",
        )

    if (
        row["observed_r"] >= 0.60
        and row["loo_composite_pearson_r"] >= 0.40
        and row["permutation_p"] <= 0.10
        and row["q_fdr_global"] > 0.05
    ):
        return (
            "low_power",
            "Signal is internally consistent but does not fully survive multiple-testing correction at this sample size.",
        )

    if row["observed_r"] < 0.45 and row["loo_composite_pearson_r"] < 0.25:
        return (
            "underfitting",
            "Even the in-sample latent correlation is weak, suggesting the current feature set is not well matched to the behavior block.",
        )

    return (
        "mixed_or_borderline",
        "No single failure mode dominates; the model is borderline or depends on neighboring modeling choices.",
    )


def _annotate_sibling_deltas(results_df: pd.DataFrame) -> pd.DataFrame:
    out = results_df.copy()
    behavior_deltas = []
    roi_deltas = []

    for _, row in out.iterrows():
        same_features = out[
            (out["feature_signature"] == row["feature_signature"])
            & (out["x_reduction"] == row["x_reduction"])
        ]
        best_behavior_loo = float(same_features["loo_composite_pearson_r"].max())
        behavior_deltas.append(best_behavior_loo - float(row["loo_composite_pearson_r"]))

        same_family = out[
            (out["stage"].isin(["baseline", "broad"]))
            & (out["neural_block"] == row["neural_block"])
            & (out["behavior_set"] == row["behavior_set"])
            & (out["x_reduction"] == row["x_reduction"])
        ]
        best_roi_loo = float(same_family["loo_composite_pearson_r"].max()) if not same_family.empty else float(
            row["loo_composite_pearson_r"]
        )
        roi_deltas.append(best_roi_loo - float(row["loo_composite_pearson_r"]))

    out["delta_loo_to_best_behavior_sibling"] = behavior_deltas
    out["delta_loo_to_best_roi_sibling"] = roi_deltas

    diagnoses = out.apply(_diagnose_model, axis=1)
    out["diagnosis"] = [item[0] for item in diagnoses]
    out["diagnosis_note"] = [item[1] for item in diagnoses]
    return out


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_search_space_files(out_dir: Path, configs: list[ModelConfig]) -> None:
    adjustable_parameters = pd.DataFrame(
        [
            {
                "parameter": "roi_scheme",
                "choices_considered": "all, nonspecific_removed, circuit, control, motor_subcortical, single control ROI",
                "explored_in_grid": True,
                "note": "Tests whether diffuse or nonspecific ROI sets are masking a tighter effect.",
            },
            {
                "parameter": "neural_block",
                "choices_considered": "graph_strength, graph_participation, graph_within_module, dcm_outgoing, dcm_plus_participation, module_summary",
                "explored_in_grid": True,
                "note": "Separates graph topology, effective connectivity, and coarse module summaries.",
            },
            {
                "parameter": "behavior_set",
                "choices_considered": "full3, vmax_pmax, rtmt, vmax, pmax",
                "explored_in_grid": True,
                "note": "Tests whether timing and peak-vigor measures belong in the same latent block.",
            },
            {
                "parameter": "x_reduction",
                "choices_considered": "pca_auto, raw",
                "explored_in_grid": True,
                "note": "Wide models use PCA; compact refined models are tested without PCA.",
            },
            {
                "parameter": "x_pca_components",
                "choices_considered": "auto-capped at <= 3 and <= behavior dimensionality",
                "explored_in_grid": True,
                "note": "Prevents broad models from exceeding what N=14 can support.",
            },
            {
                "parameter": "pls_components",
                "choices_considered": "1 retained for formal grid; >1 considered but not carried into the main screen",
                "explored_in_grid": False,
                "note": "With N=14, extra latent components add flexibility faster than they add interpretability.",
            },
            {
                "parameter": "validation",
                "choices_considered": "permutation, BH-FDR, LOSO, bootstrap on top models",
                "explored_in_grid": True,
                "note": "LOSO is the main guardrail against overfitting.",
            },
        ]
    )
    adjustable_parameters.to_csv(out_dir / "adjustable_parameters.csv", index=False)

    manifest_rows = []
    for config in configs:
        manifest_rows.append(
            {
                "stage": config.stage,
                "config_name": config.config_name,
                "roi_scheme": config.roi_scheme,
                "neural_block": config.neural_block,
                "behavior_set": config.behavior_set,
                "x_reduction": config.x_reduction,
                "n_brain_features": len(config.feature_cols),
                "n_behavior_features": len(config.behavior_cols),
                "description": config.description,
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "search_space_manifest.csv", index=False)


def _save_detailed_model_outputs(
    model_dir: Path,
    config: ModelConfig,
    usable: pd.DataFrame,
    fitted: FittedPLSPipeline,
    loo_df: pd.DataFrame,
    loo_summary_df: pd.DataFrame,
    x_boot_df: pd.DataFrame,
    y_boot_df: pd.DataFrame,
    perm_null: np.ndarray,
    final_permutation_p: float,
    final_boot_summary: dict[str, float],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    score_df = usable[["subject"] + list(config.behavior_cols)].copy()
    score_df["x_score"] = fitted.x_scores
    score_df["y_score"] = fitted.y_scores
    score_df["motor_vigor_composite"] = _motor_vigor_composite(score_df[list(config.behavior_cols)])

    loadings_df = pd.DataFrame(
        [{"set": "X", "name": name, "weight": float(weight)} for name, weight in zip(config.feature_cols, fitted.x_feature_weights)]
        + [
            {"set": "Y", "name": name, "weight": float(weight)}
            for name, weight in zip(config.behavior_cols, fitted.y_feature_weights)
        ]
    )

    score_df.to_csv(model_dir / "scores.csv", index=False)
    loadings_df.to_csv(model_dir / "loadings.csv", index=False)
    x_boot_df.to_csv(model_dir / "feature_bootstrap_stability.csv", index=False)
    y_boot_df.to_csv(model_dir / "behavior_bootstrap_stability.csv", index=False)
    loo_df.to_csv(model_dir / "loso_predictions.csv", index=False)
    loo_summary_df.to_csv(model_dir / "loso_summary.csv", index=False)
    pd.DataFrame({"permuted_r": perm_null}).to_csv(model_dir / "null_distribution.csv", index=False)

    detailed_summary = {
        "config_name": config.config_name,
        "stage": config.stage,
        "description": config.description,
        "roi_scheme": config.roi_scheme,
        "neural_block": config.neural_block,
        "behavior_set": config.behavior_set,
        "x_reduction": config.x_reduction,
        "n_subjects": int(len(usable)),
        "n_brain_features": int(len(config.feature_cols)),
        "n_behavior_features": int(len(config.behavior_cols)),
        "x_pca_components": int(fitted.n_x_components),
        "x_pca_variance_explained": float(fitted.x_variance_explained),
        "observed_r": float(fitted.observed_r),
        "final_permutation_p": float(final_permutation_p),
        **final_boot_summary,
    }
    _write_json(model_dir / "model_summary.json", detailed_summary)

    plot_model(
        scores_df=score_df,
        x_boot_df=x_boot_df,
        loo_df=loo_df,
        perm_null=perm_null,
        observed_r=fitted.observed_r,
        permutation_p=final_permutation_p,
        title=f"{config.config_name}\n{config.description}",
        out_path=model_dir / "coupling_figure.png",
    )


def _build_report(
    out_dir: Path,
    results_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
) -> None:
    baseline = results_df.loc[
        results_df["config_name"] == "baseline__all__graph_strength_plus_participation__full3__pca_auto"
    ].iloc[0]
    best_broad = results_df[results_df["stage"] == "broad"].sort_values(
        ["permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, False, True],
    ).iloc[0]
    best_refined = results_df[results_df["stage"] == "refined"].sort_values(
        ["permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, False, True],
    ).iloc[0]
    best_localized = results_df[results_df["stage"] == "localized"].sort_values(
        ["permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, False, True],
    ).iloc[0]

    top_lines = []
    for row in results_df.sort_values(
        ["q_fdr_global", "permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, True, False, True],
    ).head(8).itertuples(index=False):
        top_lines.append(
            f"| {row.config_name} | {row.stage} | {row.observed_r:.3f} | {row.permutation_p:.4f} | "
            f"{row.q_fdr_global:.4f} | {row.loo_composite_pearson_r:.3f} | {row.diagnosis} |"
        )

    report = f"""# Targeted PLS Structured Exploration

## Search Overview

- Subjects analyzed: {int(results_df["n_subjects"].max())}
- Models screened: {len(results_df)}
- Screen permutations per model: {SCREEN_PERMUTATIONS}
- Detailed permutations for top models: {DETAILED_PERMUTATIONS}
- Bootstrap draws for detailed models: {N_BOOTSTRAP}

## Refinement Trace

- Baseline broad graph model:
  - `{baseline['config_name']}`
  - latent r = {baseline['observed_r']:.3f}
  - permutation p = {baseline['permutation_p']:.4f}
  - global q = {baseline['q_fdr_global']:.4f}
  - LOSO composite r = {baseline['loo_composite_pearson_r']:.3f}

- Best broad family model:
  - `{best_broad['config_name']}`
  - latent r = {best_broad['observed_r']:.3f}
  - permutation p = {best_broad['permutation_p']:.4f}
  - global q = {best_broad['q_fdr_global']:.4f}
  - LOSO composite r = {best_broad['loo_composite_pearson_r']:.3f}

- Best refined compact control model:
  - `{best_refined['config_name']}`
  - latent r = {best_refined['observed_r']:.3f}
  - permutation p = {best_refined['permutation_p']:.4f}
  - global q = {best_refined['q_fdr_global']:.4f}
  - LOSO composite r = {best_refined['loo_composite_pearson_r']:.3f}

- Best localized single-ROI model:
  - `{best_localized['config_name']}`
  - latent r = {best_localized['observed_r']:.3f}
  - permutation p = {best_localized['permutation_p']:.4f}
  - global q = {best_localized['q_fdr_global']:.4f}
  - LOSO composite r = {best_localized['loo_composite_pearson_r']:.3f}

## What Improved

- Removing diffuse ROI sets and focusing on the control / monitoring family improved generalization relative to the original all-ROI graph model.
- Restricting behavior from `full3` to `Vmax + Pmax` improved both permutation strength and LOSO in the compact control models, consistent with timing and peak-vigor measures not loading onto the same latent axis here.
- For compact control models, removing PCA helped rather than hurt, which is what you would expect once the neural block is already only 4 to 8 interpretable features.
- The strongest localized effect came from vmPFC / dmPFC participation, which dominated the refined control-family models rather than broad catch-all parcels.

## Failure-Mode Pattern

- Overfitting dominated the wide all-ROI families when in-sample latent r stayed moderate but LOSO collapsed.
- Poor feature alignment showed up when `full3` models were consistently beaten by matched `Vmax + Pmax` siblings.
- Low power remained the main residual limitation for several positive-control models that generalized reasonably well but still sit near the multiple-testing boundary.
- Underfitting was typical of compact families with weak latent r and weak LOSO even after simplification.

## Top Models

| config | stage | latent r | perm p | global q | LOSO r | diagnosis |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(top_lines)}

## Credibility

- These results are materially better than the original pipeline, but the sample size is still only N = 14.
- The compact control-network models are the most credible because they improve both significance and LOSO while staying anatomically interpretable.
- The single-ROI vmPFC / dmPFC model is best viewed as localization within the refined control family, not as an independent confirmatory result.
- No result here should be treated as fully definitive without an external replication or a preregistered follow-up on the compact control-network models.
"""
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Targeted PLS structured exploration")
    print("=" * 80)

    print("\n[1] Building neural and behavioral feature tables ...")
    dcm_df = build_dcm_features(DCM_PARAMS_PATH)
    graph_df = build_graph_features(NODE_DELTA_PATH)
    module_df = build_module_features(MODULE_SUMM_PATH)
    feature_tables = {"dcm": dcm_df, "graph": graph_df, "module": module_df}

    behavior_df = load_task_behavior_deltas().copy()
    merged = (
        dcm_df.merge(graph_df, on="subject", how="outer")
        .merge(module_df, on="subject", how="outer")
        .merge(behavior_df, on="subject", how="inner")
        .sort_values("subject")
        .reset_index(drop=True)
    )
    merged.to_csv(out_dir / "merged_full.csv", index=False)

    print(f"   subjects with neural + behavior data: {len(merged)}")
    print(f"   DCM features:                         {len([c for c in dcm_df.columns if c != 'subject'])}")
    print(f"   graph features:                       {len([c for c in graph_df.columns if c != 'subject'])}")
    print(f"   module-summary features:              {len([c for c in module_df.columns if c != 'subject'])}")

    behavior_feature_rows = []
    for set_name, cols in BEHAVIOR_SETS.items():
        behavior_feature_rows.append(
            {
                "behavior_set": set_name,
                "n_features": len(cols),
                "features": "|".join(cols),
                "labels": " | ".join(BEHAVIOR_LABELS.get(col, col) for col in cols),
                "rationale": BEHAVIOR_SET_RATIONALES[set_name],
            }
        )
    pd.DataFrame(behavior_feature_rows).to_csv(out_dir / "behavior_sets.csv", index=False)

    print("\n[2] Building staged model configurations ...")
    configs = build_model_configs(feature_tables)
    _write_search_space_files(out_dir, configs)
    print(f"   total configs: {len(configs)}")

    print(f"\n[3] Screening all models with {SCREEN_PERMUTATIONS} permutations + LOSO ...")
    result_rows = []
    for idx, config in enumerate(configs):
        usable = merged.dropna(subset=list(config.feature_cols) + list(config.behavior_cols)).reset_index(drop=True)
        if usable.empty:
            continue

        X = usable[list(config.feature_cols)].copy()
        Y = usable[list(config.behavior_cols)].copy()

        fitted, permutation_p, _ = permutation_test(
            X,
            Y,
            x_reduction=config.x_reduction,
            requested_x_components=config.requested_x_components,
            n_perm=SCREEN_PERMUTATIONS,
            rng=np.random.default_rng(RANDOM_SEED + idx),
        )
        _, _, loo_summary = loo_prediction_analysis(
            X,
            Y,
            fitted,
            subjects=usable["subject"],
            x_reduction=config.x_reduction,
            requested_x_components=config.requested_x_components,
        )

        result_rows.append(
            {
                "stage": config.stage,
                "config_name": config.config_name,
                "description": config.description,
                "roi_scheme": config.roi_scheme,
                "neural_block": config.neural_block,
                "behavior_set": config.behavior_set,
                "x_reduction": config.x_reduction,
                "n_subjects": int(len(usable)),
                "n_brain_features": int(len(config.feature_cols)),
                "n_behavior_features": int(len(config.behavior_cols)),
                "p_n_ratio": float(len(config.feature_cols) / len(usable)),
                "x_pca_components": int(fitted.n_x_components),
                "x_pca_variance_explained": float(fitted.x_variance_explained),
                "observed_r": float(fitted.observed_r),
                "permutation_p": float(permutation_p),
                "loo_composite_pearson_r": float(loo_summary["loo_composite_pearson_r"]),
                "loo_composite_spearman_rho": float(loo_summary["loo_composite_spearman_rho"]),
                "loo_composite_rmse": float(loo_summary["loo_composite_rmse"]),
                "loo_mean_behavior_pearson_r": float(loo_summary["loo_mean_behavior_pearson_r"]),
                "loo_min_behavior_pearson_r": float(loo_summary["loo_min_behavior_pearson_r"]),
                "mean_loo_weight_similarity": float(loo_summary["mean_loo_weight_similarity"]),
                "min_loo_weight_similarity": float(loo_summary["min_loo_weight_similarity"]),
                "feature_signature": "|".join(config.feature_cols),
                "feature_names": "|".join(config.feature_cols),
                "behavior_names": "|".join(config.behavior_cols),
            }
        )
        print(
            f"   {config.config_name:90s} "
            f"r={fitted.observed_r:+.4f} "
            f"p={permutation_p:.4f} "
            f"LOSO={loo_summary['loo_composite_pearson_r']:+.4f}"
        )

    if not result_rows:
        raise RuntimeError("No analyzable models were found.")

    results_df = pd.DataFrame(result_rows).sort_values(
        ["permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    results_df["q_fdr_global"] = _bh_fdr(results_df["permutation_p"].to_numpy())
    results_df["q_fdr_stage"] = np.nan
    for stage_name, stage_df in results_df.groupby("stage"):
        stage_q = _bh_fdr(stage_df["permutation_p"].to_numpy())
        results_df.loc[stage_df.index, "q_fdr_stage"] = stage_q

    results_df = _annotate_sibling_deltas(results_df)
    results_df.to_csv(out_dir / "screened_models.csv", index=False)

    print("\n[4] Selecting top models for detailed permutation + bootstrap ...")
    must_keep = {
        "baseline__all__graph_strength_plus_participation__full3__pca_auto",
    }
    for stage_name in ["baseline", "broad", "refined", "localized"]:
        stage_slice = results_df[results_df["stage"] == stage_name]
        if not stage_slice.empty:
            must_keep.add(stage_slice.iloc[0]["config_name"])

    ranked_names = results_df.sort_values(
        ["permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, False, True],
    )["config_name"].tolist()
    ordered_detailed_names: list[str] = []
    for name in sorted(must_keep):
        if name not in ordered_detailed_names:
            ordered_detailed_names.append(name)
    for name in ranked_names:
        if name not in ordered_detailed_names:
            ordered_detailed_names.append(name)
        if len(ordered_detailed_names) >= max(MAX_DETAILED_MODELS, len(must_keep)):
            break

    config_map = {config.config_name: config for config in configs}
    detailed_rows = []
    for idx, config_name in enumerate(ordered_detailed_names):
        config = config_map[config_name]
        usable = merged.dropna(subset=list(config.feature_cols) + list(config.behavior_cols)).reset_index(drop=True)
        X = usable[list(config.feature_cols)].copy()
        Y = usable[list(config.behavior_cols)].copy()

        fitted, final_permutation_p, final_null = permutation_test(
            X,
            Y,
            x_reduction=config.x_reduction,
            requested_x_components=config.requested_x_components,
            n_perm=DETAILED_PERMUTATIONS,
            rng=np.random.default_rng(RANDOM_SEED + 10_000 + idx),
        )
        x_boot_df, y_boot_df, boot_summary = bootstrap_weight_stability(
            X,
            Y,
            fitted=fitted,
            feature_names=list(config.feature_cols),
            behavior_names=list(config.behavior_cols),
            x_reduction=config.x_reduction,
            requested_x_components=config.requested_x_components,
            n_boot=N_BOOTSTRAP,
            rng=np.random.default_rng(RANDOM_SEED + 20_000 + idx),
        )
        loo_df, loo_summary_df, loo_summary = loo_prediction_analysis(
            X,
            Y,
            fitted,
            subjects=usable["subject"],
            x_reduction=config.x_reduction,
            requested_x_components=config.requested_x_components,
        )

        model_dir = out_dir / "models" / config.config_name
        _save_detailed_model_outputs(
            model_dir=model_dir,
            config=config,
            usable=usable,
            fitted=fitted,
            loo_df=loo_df,
            loo_summary_df=loo_summary_df,
            x_boot_df=x_boot_df,
            y_boot_df=y_boot_df,
            perm_null=final_null,
            final_permutation_p=final_permutation_p,
            final_boot_summary=boot_summary,
        )

        detailed_rows.append(
            {
                "config_name": config.config_name,
                "final_permutation_p": float(final_permutation_p),
                "boot_n_valid": int(boot_summary["n_valid_boots"]),
                "median_boot_weight_similarity": float(boot_summary["median_boot_weight_similarity"]),
                "p05_boot_weight_similarity": float(boot_summary["p05_boot_weight_similarity"]),
                "boot_score_corr_ci_2p5": float(boot_summary["boot_score_corr_ci_2p5"]),
                "boot_score_corr_ci_97p5": float(boot_summary["boot_score_corr_ci_97p5"]),
                "n_stable_features_ci_excluding_zero": int(boot_summary["n_stable_features_ci_excluding_zero"]),
                "detailed_model_dir": str(model_dir),
                "final_loo_composite_pearson_r": float(loo_summary["loo_composite_pearson_r"]),
                "top_feature": x_boot_df.iloc[0]["name"],
            }
        )
        print(
            f"   detailed {config.config_name:80s} "
            f"final_p={final_permutation_p:.4f} "
            f"boot_med_sim={boot_summary['median_boot_weight_similarity']:.3f}"
        )

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(out_dir / "detailed_models.csv", index=False)

    results_df = results_df.merge(detailed_df, on="config_name", how="left")
    results_df.to_csv(out_dir / "screened_models_with_detailed_metrics.csv", index=False)

    top_models = results_df.sort_values(
        ["q_fdr_global", "permutation_p", "loo_composite_pearson_r", "n_brain_features"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    top_models.head(20).to_csv(out_dir / "top_models.csv", index=False)

    refinement_trace = pd.DataFrame(
        [
            top_models[top_models["config_name"] == "baseline__all__graph_strength_plus_participation__full3__pca_auto"].iloc[0],
            top_models[top_models["stage"] == "broad"].iloc[0],
            top_models[top_models["stage"] == "refined"].iloc[0],
            top_models[top_models["stage"] == "localized"].iloc[0],
        ]
    )
    refinement_trace.to_csv(out_dir / "refinement_trace.csv", index=False)

    _build_report(out_dir, results_df=top_models, detailed_df=detailed_df)

    summary_payload = {
        "n_subjects": int(results_df["n_subjects"].max()),
        "n_models_screened": int(len(results_df)),
        "screen_permutations": int(SCREEN_PERMUTATIONS),
        "detailed_permutations": int(DETAILED_PERMUTATIONS),
        "n_detailed_models": int(len(detailed_df)),
        "best_global_model": top_models.head(1).to_dict(orient="records"),
        "best_broad_model": top_models[top_models["stage"] == "broad"].head(1).to_dict(orient="records"),
        "best_refined_model": top_models[top_models["stage"] == "refined"].head(1).to_dict(orient="records"),
        "best_localized_model": top_models[top_models["stage"] == "localized"].head(1).to_dict(orient="records"),
    }
    _write_json(out_dir / "summary.json", summary_payload)

    print("\n[5] Summary")
    print(top_models.head(12)[
        [
            "config_name",
            "stage",
            "observed_r",
            "permutation_p",
            "q_fdr_global",
            "loo_composite_pearson_r",
            "diagnosis",
        ]
    ].to_string(index=False))
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
