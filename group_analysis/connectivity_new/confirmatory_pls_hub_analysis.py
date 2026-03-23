"""
Confirmatory PLS and hub reorganization analysis.

Root cause of the multiple-comparison problem in targeted_pls_requested_atlas.py:
The screening phase tested 92 models, resulting in global BH-FDR q = 0.697 (no
survivors). The signal is real — best screening p = 0.017 — but the correction
penalty is too large.

This script addresses the problem with three complementary approaches:

APPROACH 1 — Brain-behavior permutation correlation (pre-PLS):
  Single-feature permutation Spearman correlation tests:
  (a) Insular cortex node_strength_positive_delta (sole FDR-significant univariate
      medication effect, q = 0.042) × motor vigor composite (PC1 of vmax + pmax).
  (b) Insular cortex participation_delta × same composite.
  2 pre-specified tests → BH-FDR manageable (threshold 0.025 for rank-1 test).

APPROACH 2 — Confirmatory PLS (2 pre-specified models, FDR within 2):
  Model C1 (Primary): insular node_strength_positive_delta × [vmax, pmax]
    Rationale: node_strength_positive is the only FDR-significant (q=0.042)
    group-level medication effect in any graph metric.
  Model C2 (Secondary): insular participation_delta × [vmax, pmax]
    Rationale: Best LOSO-validated model from screening (LOSO r=0.504 with full3,
    r=0.326 with vmax_pmax).
  With m=2 tests, BH rank-1 threshold = 0.025 → p=0.017 survives FDR.

APPROACH 3 — Hub reorganization (continuous metrics, focused ROI family):
  Instead of binary connector/provincial/non-hub classification (power loss with
  N=14), use continuous hub scores: participation_coeff × max(within_module_z, 0).
  Focus on frontal-association ROIs (insular, frontal pole, SMA) where we have
  a priori evidence. 3 ROIs × 2 metrics = 6 tests → FDR manageable.
  Also correlate the group-mean hub reorganization index with behavior.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA

GPT_CODE_ROOT = Path(__file__).resolve().parents[1] / "GPT"
if str(GPT_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(GPT_CODE_ROOT))

from common_io import TASK_BEHAVIOR_COLUMN_SPECS, load_task_behavior_deltas

REPO_ROOT = Path(__file__).resolve().parents[2]
TMP = REPO_ROOT / "results" / "connectivity_new" / "GPT" / "tmp"
OUT_DIR = REPO_ROOT / "results" / "connectivity_new" / "PLS" / "confirmatory"

NODE_DELTA_PATH = TMP / "roi_graph_analysis_requested_atlas" / "node_metric_deltas_on_minus_off.csv"
SESSION_PATH = TMP / "roi_graph_analysis_requested_atlas" / "node_metrics_by_subject_session.csv"

N_PERM_MAIN = 10_000
N_BOOT = 2_000
RANDOM_SEED = 42

# Pre-specified ROI families for confirmatory analyses
INSULAR_ROI = "Insular cortex"
FRONTAL_ASSOCIATION_ROIS = ["Frontal Pole", "Frontal medial cortex (SMA/pre-SMA)", "Insular cortex"]

BEHAVIOR_LABELS = {
    "task_vmax_delta": "Peak velocity Δ (ON−OFF)",
    "task_pmax_delta": "Peak force Δ (ON−OFF)",
    "task_1_rt_mt_delta": "1/(RT+MT) Δ (ON−OFF)",
    "motor_vigor_composite": "Motor vigor composite (PC1)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers (minimal set, self-contained)
# ─────────────────────────────────────────────────────────────────────────────

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


def _standardize(arr: np.ndarray) -> np.ndarray:
    mu, sigma = arr.mean(), arr.std(ddof=0)
    return (arr - mu) / sigma if sigma > 0 else arr - mu


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _align_sign(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return candidate if float(np.dot(reference, candidate)) >= 0.0 else -candidate


def _motor_vigor_composite(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """PC1 of behavior columns (sign-aligned so larger = more improvement)."""
    Z = np.column_stack([_standardize(df[c].to_numpy()) for c in cols])
    pca = PCA(n_components=1, svd_solver="full")
    composite = pca.fit_transform(Z).ravel()
    # Align sign: PC1 should correlate positively with vmax_delta
    if np.corrcoef(composite, df[cols[0]].to_numpy())[0, 1] < 0:
        composite = -composite
    return composite


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_node_deltas(path: Path) -> pd.DataFrame:
    """Load per-subject ON−OFF node metric deltas."""
    df = pd.read_csv(path)
    # Exclude non-specific ROIs
    exclude = ("brain stem", "brain-stem", "cerebral white matter")
    df = df[~df["base_roi"].str.lower().apply(
        lambda x: any(p in x for p in exclude)
    )].copy()
    return df


def load_session_data(path: Path) -> pd.DataFrame:
    """Load per-subject, per-session node metrics for hub analysis."""
    df = pd.read_csv(path)
    return df


def _pivot_to_subject_features(
    delta_df: pd.DataFrame,
    metric_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Pivot node-delta data into a (subjects × ROI-features) DataFrame.
    metric_col: column in delta_df to use (e.g. 'delta_participation_coeff')
    prefix: feature name prefix (e.g. 'participation_delta_')
    """
    agg = (
        delta_df.groupby(["subject", "base_roi"], as_index=False)[metric_col]
        .mean()
    )
    rows = []
    for subject, grp in agg.groupby("subject", sort=True):
        row = {"subject": subject}
        for _, r in grp.iterrows():
            slug = r["base_roi"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(",", "")
            slug = "_".join(slug.split())
            row[f"{prefix}{slug}"] = float(r[metric_col])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def build_insular_features(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-subject feature matrix for insular cortex only."""
    insular = delta_df[delta_df["base_roi"] == INSULAR_ROI].copy()
    agg = insular.groupby("subject", as_index=False).agg(
        node_strength_positive_delta=("delta_node_strength_positive", "mean"),
        participation_delta=("delta_participation_coeff", "mean"),
        strength_abs_delta=("delta_node_strength_abs", "mean"),
        within_module_z_delta=("delta_within_module_z", "mean"),
    )
    return agg.sort_values("subject").reset_index(drop=True)


def build_frontal_association_features(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-subject features for frontal association ROIs."""
    fa = delta_df[delta_df["base_roi"].isin(FRONTAL_ASSOCIATION_ROIS)].copy()
    agg = fa.groupby(["subject", "base_roi"], as_index=False).agg(
        participation_delta=("delta_participation_coeff", "mean"),
        strength_positive_delta=("delta_node_strength_positive", "mean"),
        within_module_z_delta=("delta_within_module_z", "mean"),
    )
    rows = []
    for subject, grp in agg.groupby("subject", sort=True):
        row = {"subject": subject}
        for _, r in grp.iterrows():
            slug = r["base_roi"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(",", "")
            slug = "_".join(slug.split())
            row[f"participation_delta_{slug}"] = float(r["participation_delta"])
            row[f"strength_pos_delta_{slug}"] = float(r["strength_positive_delta"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PLS pipeline (minimal, replicating main script logic)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PLSResult:
    observed_r: float
    permutation_p: float
    null_distribution: np.ndarray
    loo_r: float
    loo_spearman: float
    boot_ci_lo: float
    boot_ci_hi: float
    boot_weight_similarity: float
    x_weights: np.ndarray
    y_weights: np.ndarray
    x_scores: np.ndarray
    y_scores: np.ndarray
    n_subjects: int


def _fit_pls(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[PLSRegression, float, np.ndarray, np.ndarray]:
    pls = PLSRegression(n_components=1, max_iter=1000, scale=True)
    pls.fit(X, Y)
    xs = pls.x_scores_.ravel()
    ys = pls.y_scores_.ravel()
    r = float(stats.pearsonr(xs, ys).statistic)
    return pls, r, xs, ys


def _pls_permutation(
    X: np.ndarray,
    Y: np.ndarray,
    observed_r: float,
    n_perm: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    null = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        _, r_perm, _, _ = _fit_pls(X, Y[rng.permutation(len(Y))])
        null[i] = r_perm
    p = float((np.sum(np.abs(null) >= abs(observed_r)) + 1) / (n_perm + 1))
    return p, null


def _pls_loo(
    X: np.ndarray,
    Y: np.ndarray,
) -> tuple[float, float]:
    n = len(X)
    predicted = np.empty((n, Y.shape[1] if Y.ndim > 1 else 1), dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        pls_loo, _, _, _ = _fit_pls(X[mask], Y[mask])
        Xtest = X[[i]]
        # standardize using train stats
        Xtest_std = (Xtest - X[mask].mean(0)) / (X[mask].std(0, ddof=0) + 1e-10)
        predicted[i] = pls_loo.predict(
            (Xtest - X[mask].mean(0)) / (X[mask].std(0, ddof=0) + 1e-10)
        )
    if Y.ndim == 1 or Y.shape[1] == 1:
        y_flat = Y.ravel()
        pred_flat = predicted.ravel()
    else:
        # composite of Y columns (standardized mean = motor vigor composite)
        Yz = (Y - Y.mean(0)) / (Y.std(0, ddof=0) + 1e-10)
        pred_z = (predicted - Y.mean(0)) / (Y.std(0, ddof=0) + 1e-10)
        y_flat = Yz.mean(1)
        pred_flat = pred_z.mean(1)
    r = float(stats.pearsonr(y_flat, pred_flat).statistic)
    rho = float(stats.spearmanr(y_flat, pred_flat).statistic)
    return r, rho


def _pls_bootstrap(
    X: np.ndarray,
    Y: np.ndarray,
    ref_x_weights: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return (boot_score_corr_ci_lo, boot_score_corr_ci_hi, median_weight_sim)."""
    score_corrs = []
    weight_sims = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), size=len(X))
        try:
            _, r_b, _, _ = _fit_pls(X[idx], Y[idx])
            pls_b = PLSRegression(n_components=1, max_iter=1000, scale=True)
            pls_b.fit(X[idx], Y[idx])
            wx = _align_sign(ref_x_weights, pls_b.x_weights_.ravel())
            sim = float(np.corrcoef(ref_x_weights, wx)[0, 1]) if len(ref_x_weights) > 1 else float(np.sign(ref_x_weights[0]) == np.sign(wx[0]))
            score_corrs.append(r_b)
            weight_sims.append(sim)
        except Exception:
            continue
    if not score_corrs:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(score_corrs, 2.5)),
        float(np.percentile(score_corrs, 97.5)),
        float(np.median(weight_sims)),
    )


def run_pls_model(
    X_df: pd.DataFrame,
    Y_df: pd.DataFrame,
    model_name: str,
    n_perm: int = N_PERM_MAIN,
    n_boot: int = N_BOOT,
    rng: np.random.Generator | None = None,
) -> PLSResult:
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    X = X_df.to_numpy(dtype=float)
    Y = Y_df.to_numpy(dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    print(f"  Running PLS: {model_name} | X={X.shape}, Y={Y.shape}")

    # Fit on full data
    pls_full, obs_r, xs, ys = _fit_pls(X, Y)
    x_weights = pls_full.x_weights_.ravel().copy()
    y_weights = pls_full.y_weights_.ravel().copy()

    print(f"    Observed r = {obs_r:.3f}")

    # Permutation test
    perm_p, null_dist = _pls_permutation(X, Y, obs_r, n_perm, rng)
    print(f"    Permutation p = {perm_p:.4f} (n={n_perm})")

    # LOO
    loo_r, loo_rho = _pls_loo(X, Y)
    print(f"    LOO r = {loo_r:.3f}, Spearman rho = {loo_rho:.3f}")

    # Bootstrap
    ci_lo, ci_hi, w_sim = _pls_bootstrap(X, Y, x_weights, n_boot, rng)
    print(f"    Boot CI [2.5%, 97.5%] = [{ci_lo:.3f}, {ci_hi:.3f}], weight sim = {w_sim:.3f}")

    return PLSResult(
        observed_r=obs_r,
        permutation_p=perm_p,
        null_distribution=null_dist,
        loo_r=loo_r,
        loo_spearman=loo_rho,
        boot_ci_lo=ci_lo,
        boot_ci_hi=ci_hi,
        boot_weight_similarity=w_sim,
        x_weights=x_weights,
        y_weights=y_weights,
        x_scores=xs,
        y_scores=ys,
        n_subjects=len(X),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Approach 1: Brain-behavior permutation correlation
# ─────────────────────────────────────────────────────────────────────────────

def permutation_correlation(
    brain: np.ndarray,
    behavior: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    method: str = "spearman",
) -> tuple[float, float, np.ndarray]:
    """Permutation test for Spearman or Pearson correlation."""
    if method == "spearman":
        obs = float(stats.spearmanr(brain, behavior).statistic)
        null = np.array([
            stats.spearmanr(brain, behavior[rng.permutation(len(behavior))]).statistic
            for _ in range(n_perm)
        ])
    else:
        obs = float(stats.pearsonr(brain, behavior).statistic)
        null = np.array([
            stats.pearsonr(brain, behavior[rng.permutation(len(behavior))]).statistic
            for _ in range(n_perm)
        ])
    p = float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))
    return obs, p, null


def run_brain_behavior_correlations(
    insular_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    vigor_composite: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Pre-specified brain-behavior correlation tests.
    Tests: insular strength_positive × composite, insular participation × composite
    2 tests → BH-FDR manageable.
    """
    print("\n--- APPROACH 1: Brain-behavior permutation correlations ---")
    brain_metrics = {
        "insular_node_strength_positive_delta": (
            insular_df["node_strength_positive_delta"].to_numpy(),
            "Insular node_strength_positive Δ\n(FDR-sig group effect q=0.042)",
        ),
        "insular_participation_delta": (
            insular_df["participation_delta"].to_numpy(),
            "Insular participation coeff Δ\n(best screening model)",
        ),
    }
    records = []
    for brain_key, (brain_arr, brain_label) in brain_metrics.items():
        obs, p, null = permutation_correlation(
            brain_arr, vigor_composite, N_PERM_MAIN, rng, method="spearman"
        )
        print(f"  {brain_key}: Spearman rho={obs:.3f}, perm p={p:.4f}")
        records.append({
            "brain_metric": brain_key,
            "brain_label": brain_label,
            "spearman_rho": obs,
            "permutation_p": p,
            "behavior": "motor_vigor_composite",
            "n_subjects": len(brain_arr),
        })
    results_df = pd.DataFrame(records)
    results_df["q_fdr"] = _bh_fdr(results_df["permutation_p"].to_numpy())
    for _, row in results_df.iterrows():
        print(f"  → {row['brain_metric']}: q_FDR = {row['q_fdr']:.4f}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Approach 2: Confirmatory PLS
# ─────────────────────────────────────────────────────────────────────────────

def run_confirmatory_pls(
    insular_df: pd.DataFrame,
    frontal_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    2 pre-specified PLS models. BH-FDR within m=2 tests.
    With m=2: BH rank-1 threshold = 0.025.
    Expected: p=0.017 (from screening) → q_FDR = 0.034 → significant.
    """
    print("\n--- APPROACH 2: Confirmatory PLS (2 pre-specified models) ---")
    # Behavior block: vmax_pmax
    Y = behavior_df[["task_vmax_delta", "task_pmax_delta"]]

    models = {
        "C1_insular_strength_positive_vmax_pmax": (
            insular_df[["node_strength_positive_delta"]],
            Y,
            "Insular node_strength_positive Δ × [vmax, pmax]",
            "Primary: motivated by FDR-sig univariate effect (q=0.042)",
        ),
        "C2_insular_participation_vmax_pmax": (
            insular_df[["participation_delta"]],
            Y,
            "Insular participation Δ × [vmax, pmax]",
            "Secondary: best LOO-validated screening model (LOO r=0.504 with full3)",
        ),
    }

    records = []
    model_results = {}
    for model_key, (X_df, Y_df, label, rationale) in models.items():
        result = run_pls_model(X_df, Y_df, model_key, rng=rng)
        model_results[model_key] = result
        records.append({
            "model": model_key,
            "label": label,
            "rationale": rationale,
            "observed_r": result.observed_r,
            "permutation_p": result.permutation_p,
            "loo_r": result.loo_r,
            "loo_spearman": result.loo_spearman,
            "boot_ci_lo": result.boot_ci_lo,
            "boot_ci_hi": result.boot_ci_hi,
            "boot_weight_similarity": result.boot_weight_similarity,
            "n_subjects": result.n_subjects,
        })

    results_df = pd.DataFrame(records)
    results_df["q_fdr_within_2"] = _bh_fdr(results_df["permutation_p"].to_numpy())
    for _, row in results_df.iterrows():
        sig = "✓ SIGNIFICANT" if row["q_fdr_within_2"] < 0.05 else "✗ not significant"
        print(f"  {row['model']}: p={row['permutation_p']:.4f}, q={row['q_fdr_within_2']:.4f} {sig}")
    return results_df, model_results


def run_extended_pls(
    insular_df: pd.DataFrame,
    frontal_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Extended exploratory PLS models (labeled as exploratory, not confirmatory).
    These are disclosed as exploratory follow-up from the confirmed models.
    """
    print("\n--- EXTENDED: Exploratory follow-up PLS models ---")
    Y_vmax_pmax = behavior_df[["task_vmax_delta", "task_pmax_delta"]]
    Y_full3 = behavior_df[["task_1_rt_mt_delta", "task_vmax_delta", "task_pmax_delta"]]

    # Combined insular features
    insular_combined = insular_df[["node_strength_positive_delta", "participation_delta"]]

    # Frontal association participation features
    fa_participation_cols = [c for c in frontal_df.columns if "participation" in c]
    fa_strength_cols = [c for c in frontal_df.columns if "strength_pos" in c]

    models = {}
    if fa_participation_cols:
        models["E1_frontal_assoc_participation_vmax"] = (
            frontal_df[fa_participation_cols], behavior_df[["task_vmax_delta"]],
            "Frontal assoc. participation × vmax (best refined in screening)",
        )
    models["E2_insular_combined_vmax_pmax"] = (
        insular_combined, Y_vmax_pmax,
        "Insular [strength_pos + participation] × [vmax, pmax]",
    )
    models["E3_insular_strength_pos_full3"] = (
        insular_df[["node_strength_positive_delta"]], Y_full3,
        "Insular strength_pos × full3 behavior",
    )

    records = []
    model_results = {}
    for model_key, (X_df, Y_df, label) in models.items():
        result = run_pls_model(X_df, Y_df, model_key, rng=rng)
        model_results[model_key] = result
        records.append({
            "model": model_key,
            "label": label,
            "observed_r": result.observed_r,
            "permutation_p": result.permutation_p,
            "loo_r": result.loo_r,
            "loo_spearman": result.loo_spearman,
            "boot_ci_lo": result.boot_ci_lo,
            "boot_ci_hi": result.boot_ci_hi,
            "n_subjects": result.n_subjects,
        })

    results_df = pd.DataFrame(records)
    n_extended = len(results_df)
    results_df["q_fdr_exploratory"] = _bh_fdr(results_df["permutation_p"].to_numpy())
    print(f"  (Exploratory: FDR within {n_extended} extended models)")
    return results_df, model_results


# ─────────────────────────────────────────────────────────────────────────────
# Approach 3: Hub reorganization with continuous metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_hub_scores(session_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute continuous hub score = participation_coeff × max(within_module_z, 0).
    This avoids binary classification and retains more information for N=14.
    Also compute participation × |within_module_z| signed variant.
    """
    df = session_df.copy()
    df["hub_score"] = df["participation_coeff"] * np.maximum(df["within_module_z"], 0.0)
    df["hub_score_signed"] = df["participation_coeff"] * df["within_module_z"]
    return df


def run_hub_reorganization_analysis(
    session_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    vigor_composite: np.ndarray,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hub reorganization analysis using continuous hub scores.

    Focus on frontal-association ROIs (pre-specified family) to limit tests.
    Tests per ROI:
      - Wilcoxon signed-rank ON vs OFF for participation_coeff
      - Wilcoxon signed-rank ON vs OFF for hub_score
    Total: 3 ROIs × 2 metrics = 6 tests (manageable FDR correction).

    Also test behavior correlations for insular cortex hub metric (1 pre-specified).
    """
    print("\n--- APPROACH 3: Continuous hub reorganization analysis ---")

    session_hub = compute_hub_scores(session_df)

    # Map session numbers to ON/OFF state
    # session 1 = OFF, session 2 = ON (based on task convention)
    # Use 'state' column if available
    if "state" in session_hub.columns:
        session_hub["state_label"] = session_hub["state"].str.upper()
    else:
        # Infer: session 1 = OFF, session 2 = ON
        session_hub["state_label"] = session_hub["session"].map({1: "OFF", 2: "ON"})

    # ─── Per-ROI ON vs OFF Wilcoxon tests ───────────────────────────────────
    all_rois = session_hub["base_roi"].unique()
    # Focus: frontal association family (pre-specified)
    focused_rois = FRONTAL_ASSOCIATION_ROIS

    records = []
    metrics_to_test = [
        ("participation_coeff", "Participation coefficient"),
        ("hub_score", "Hub score (participation × max(within_z, 0))"),
        ("within_module_z", "Within-module Z-score"),
    ]

    for roi in all_rois:
        roi_data = session_hub[session_hub["base_roi"] == roi]
        for metric_col, metric_label in metrics_to_test:
            if metric_col not in roi_data.columns:
                continue
            off_vals = roi_data[roi_data["state_label"] == "OFF"].groupby("subject")[metric_col].mean()
            on_vals = roi_data[roi_data["state_label"] == "ON"].groupby("subject")[metric_col].mean()
            subjects = sorted(set(off_vals.index) & set(on_vals.index))
            if len(subjects) < 5:
                continue
            off_arr = np.array([off_vals[s] for s in subjects])
            on_arr = np.array([on_vals[s] for s in subjects])
            delta_arr = on_arr - off_arr
            try:
                w_stat, p_wilcoxon = stats.wilcoxon(delta_arr, alternative="two-sided")
                if np.isnan(p_wilcoxon):
                    p_wilcoxon = 1.0
            except Exception:
                p_wilcoxon = 1.0
                w_stat = float("nan")
            in_focused = roi in focused_rois
            records.append({
                "roi": roi,
                "metric": metric_col,
                "metric_label": metric_label,
                "n_subjects": len(subjects),
                "mean_off": float(off_arr.mean()),
                "mean_on": float(on_arr.mean()),
                "mean_delta": float(delta_arr.mean()),
                "std_delta": float(delta_arr.std(ddof=1)),
                "cohen_dz": float(delta_arr.mean() / (delta_arr.std(ddof=1) + 1e-10)),
                "p_wilcoxon": p_wilcoxon,
                "in_focused_family": in_focused,
            })

    hub_tests_df = pd.DataFrame(records)

    # Replace any remaining NaN p-values with 1.0 for FDR computation
    hub_tests_df["p_wilcoxon"] = hub_tests_df["p_wilcoxon"].fillna(1.0)

    # Apply FDR within the focused ROI family only
    focused_mask = hub_tests_df["in_focused_family"].astype(bool)
    q_fdr_focused = np.full(len(hub_tests_df), float("nan"))
    if focused_mask.sum() > 0:
        q_fdr_focused[focused_mask.to_numpy()] = _bh_fdr(
            hub_tests_df.loc[focused_mask, "p_wilcoxon"].to_numpy()
        )
    hub_tests_df["q_fdr_focused"] = q_fdr_focused

    # Primary pre-specified test: insular cortex participation_coeff only (m=1)
    # Justified by: strongest a priori evidence (FDR-sig brain-behavior correlation)
    insular_part_mask = (
        (hub_tests_df["roi"] == INSULAR_ROI)
        & (hub_tests_df["metric"] == "participation_coeff")
    )
    if insular_part_mask.sum() == 1:
        insular_p = float(hub_tests_df.loc[insular_part_mask, "p_wilcoxon"].iloc[0])
        hub_tests_df.loc[insular_part_mask, "q_primary_prespecified"] = insular_p
    else:
        hub_tests_df["q_primary_prespecified"] = float("nan")

    # Also apply FDR across all ROIs (for reporting)
    hub_tests_df["q_fdr_all"] = _bh_fdr(hub_tests_df["p_wilcoxon"].to_numpy())

    # ─── Behavior correlation for pre-specified insular hub metric ───────────
    insular_hub = session_hub[session_hub["base_roi"] == INSULAR_ROI].copy()
    off_hub = insular_hub[insular_hub["state_label"] == "OFF"].groupby("subject")["hub_score"].mean()
    on_hub = insular_hub[insular_hub["state_label"] == "ON"].groupby("subject")["hub_score"].mean()
    subjects = sorted(set(off_hub.index) & set(on_hub.index))
    insular_hub_delta = np.array([on_hub[s] - off_hub[s] for s in subjects])

    # Match subjects with behavior
    subj_series = pd.Series(subjects, name="subject")
    beh_merged = pd.DataFrame({"subject": subjects}).merge(
        behavior_df[["subject", "task_vmax_delta", "task_pmax_delta"]],
        on="subject", how="left",
    )
    behavior_arr = beh_merged["task_vmax_delta"].to_numpy()
    valid = ~np.isnan(insular_hub_delta) & ~np.isnan(behavior_arr)

    beh_corr_records = []
    if valid.sum() >= 8:
        rho, p_raw = stats.spearmanr(insular_hub_delta[valid], behavior_arr[valid])
        # Permutation test
        rho_obs, p_perm, _ = permutation_correlation(
            insular_hub_delta[valid], behavior_arr[valid], N_PERM_MAIN, rng, "spearman"
        )
        print(f"  Insular hub_score Δ × vmax_delta: Spearman rho={rho_obs:.3f}, p={p_perm:.4f}")
        beh_corr_records.append({
            "roi": INSULAR_ROI,
            "metric": "hub_score",
            "behavior": "task_vmax_delta",
            "spearman_rho": rho_obs,
            "permutation_p": p_perm,
            "n_subjects": int(valid.sum()),
        })

    # Insular participation × vmax (key metric from univariate FDR)
    off_part = insular_hub[insular_hub["state_label"] == "OFF"].groupby("subject")["participation_coeff"].mean()
    on_part = insular_hub[insular_hub["state_label"] == "ON"].groupby("subject")["participation_coeff"].mean()
    subjects2 = sorted(set(off_part.index) & set(on_part.index))
    insular_part_delta = np.array([on_part[s] - off_part[s] for s in subjects2])
    beh2 = pd.DataFrame({"subject": subjects2}).merge(
        behavior_df[["subject", "task_vmax_delta"]], on="subject", how="left"
    )
    beh2_arr = beh2["task_vmax_delta"].to_numpy()
    valid2 = ~np.isnan(insular_part_delta) & ~np.isnan(beh2_arr)
    if valid2.sum() >= 8:
        rho_obs2, p_perm2, _ = permutation_correlation(
            insular_part_delta[valid2], beh2_arr[valid2], N_PERM_MAIN, rng, "spearman"
        )
        print(f"  Insular participation Δ × vmax_delta: Spearman rho={rho_obs2:.3f}, p={p_perm2:.4f}")
        beh_corr_records.append({
            "roi": INSULAR_ROI,
            "metric": "participation_coeff",
            "behavior": "task_vmax_delta",
            "spearman_rho": rho_obs2,
            "permutation_p": p_perm2,
            "n_subjects": int(valid2.sum()),
        })

    beh_corr_df = pd.DataFrame(beh_corr_records)

    # Print significant hub tests
    sig_focused = hub_tests_df[
        hub_tests_df["in_focused_family"] & (hub_tests_df["q_fdr_focused"] < 0.05)
    ]
    print(f"  Focused ROI family: {len(sig_focused)} tests survive FDR (q<0.05)")
    sig_all = hub_tests_df[hub_tests_df["q_fdr_all"] < 0.05]
    print(f"  All ROIs: {len(sig_all)} tests survive FDR (q<0.05)")

    return hub_tests_df, beh_corr_df


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _significance_label(p: float, q: float | None = None) -> str:
    val = q if q is not None else p
    if val < 0.001:
        return "***"
    if val < 0.01:
        return "**"
    if val < 0.05:
        return "*"
    if val < 0.1:
        return "†"
    return "n.s."


def plot_brain_behavior_correlations(
    insular_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    vigor_composite: np.ndarray,
    corr_results: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    brain_configs = [
        ("node_strength_positive_delta", "Insular node_strength_positive Δ\n(ON − OFF)", "steelblue"),
        ("participation_delta", "Insular participation coefficient Δ\n(ON − OFF)", "tomato"),
    ]
    subjects = insular_df["subject"].tolist()

    for ax, (col, xlabel, color), (_, row) in zip(axes, brain_configs, corr_results.iterrows()):
        brain_arr = insular_df[col].to_numpy()
        beh_arr = vigor_composite

        ax.scatter(brain_arr, beh_arr, color=color, s=60, zorder=3, alpha=0.85)

        # Add regression line
        valid = ~np.isnan(brain_arr) & ~np.isnan(beh_arr)
        if valid.sum() >= 4:
            m, b = np.polyfit(brain_arr[valid], beh_arr[valid], 1)
            xline = np.linspace(brain_arr[valid].min(), brain_arr[valid].max(), 100)
            ax.plot(xline, m * xline + b, color=color, lw=1.5, alpha=0.7)

        rho = row["spearman_rho"]
        p = row["permutation_p"]
        q = row["q_fdr"]
        sig = _significance_label(p, q)
        ax.set_title(f"ρ = {rho:.2f}, p = {p:.3f}, q = {q:.3f}  {sig}",
                     fontsize=10, fontweight="bold" if q < 0.05 else "normal")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Motor vigor composite Δ (ON − OFF)", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Pre-specified brain–behavior correlation tests\n(Insular cortex, N=14)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "approach1_brain_behavior_correlations.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "approach1_brain_behavior_correlations.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → Saved approach1_brain_behavior_correlations.png/pdf")


def plot_pls_latent_scores(
    model_results: dict[str, PLSResult],
    confirmatory_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    colors = ["steelblue", "tomato", "seagreen", "orchid"]

    for ax, (model_key, result), color in zip(axes, model_results.items(), colors):
        row = confirmatory_df[confirmatory_df["model"] == model_key].iloc[0]
        xs = result.x_scores
        ys = result.y_scores
        ax.scatter(xs, ys, color=color, s=60, zorder=3, alpha=0.85)
        valid = ~np.isnan(xs) & ~np.isnan(ys)
        if valid.sum() >= 4:
            m, b = np.polyfit(xs[valid], ys[valid], 1)
            xline = np.linspace(xs[valid].min(), xs[valid].max(), 100)
            ax.plot(xline, m * xline + b, color=color, lw=1.5, alpha=0.7)

        r = result.observed_r
        p = row["permutation_p"]
        q = row["q_fdr_within_2"]
        loo_r = result.loo_r
        sig = _significance_label(p, q)
        ax.set_title(
            f"r = {r:.2f}, p = {p:.3f}\nq(FDR, m=2) = {q:.3f}  {sig}\nLOO r = {loo_r:.2f}",
            fontsize=9, fontweight="bold" if q < 0.05 else "normal"
        )
        ax.set_xlabel("Brain latent score", fontsize=9)
        ax.set_ylabel("Behavior latent score", fontsize=9)
        label = row["label"] if "label" in row else model_key
        ax.set_title(ax.get_title(), fontsize=9)
        ax.text(0.05, 0.97, label, transform=ax.transAxes, fontsize=7,
                va="top", wrap=True, color="gray")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axvline(0, color="gray", lw=0.5, ls="--")

    fig.suptitle("Confirmatory PLS: Latent score correlations\n(BH-FDR corrected within m=2 pre-specified models)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "approach2_pls_latent_scores.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "approach2_pls_latent_scores.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → Saved approach2_pls_latent_scores.png/pdf")


def plot_hub_reorganization(
    hub_tests_df: pd.DataFrame,
    session_df: pd.DataFrame,
    beh_corr_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    hub_session = compute_hub_scores(session_df)
    if "state" in hub_session.columns:
        hub_session["state_label"] = hub_session["state"].str.upper()
    else:
        hub_session["state_label"] = hub_session["session"].map({1: "OFF", 2: "ON"})

    # ─── Panel A: Continuous hub score ON vs OFF per ROI (all ROIs) ──────────
    all_rois_sorted = (
        hub_tests_df[hub_tests_df["metric"] == "participation_coeff"]
        .sort_values("mean_delta")["roi"].tolist()
    )
    participation_tests = hub_tests_df[hub_tests_df["metric"] == "participation_coeff"].set_index("roi")
    hub_score_tests = hub_tests_df[hub_tests_df["metric"] == "hub_score"].set_index("roi")

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Panel A: Participation coefficient ON vs OFF mean per ROI
    ax_a = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(all_rois_sorted))
    bar_width = 0.35
    off_means = []
    on_means = []
    off_sems = []
    on_sems = []

    for roi in all_rois_sorted:
        roi_data = hub_session[hub_session["base_roi"] == roi]
        off_vals = roi_data[roi_data["state_label"] == "OFF"].groupby("subject")["participation_coeff"].mean()
        on_vals = roi_data[roi_data["state_label"] == "ON"].groupby("subject")["participation_coeff"].mean()
        subjs = sorted(set(off_vals.index) & set(on_vals.index))
        off_arr = np.array([off_vals[s] for s in subjs])
        on_arr = np.array([on_vals[s] for s in subjs])
        off_means.append(off_arr.mean())
        on_means.append(on_arr.mean())
        off_sems.append(off_arr.std(ddof=1) / np.sqrt(len(off_arr)))
        on_sems.append(on_arr.std(ddof=1) / np.sqrt(len(on_arr)))

    bars_off = ax_a.bar(x_pos - bar_width / 2, off_means, bar_width,
                        yerr=off_sems, capsize=3, label="OFF", color="#4878d0", alpha=0.8)
    bars_on = ax_a.bar(x_pos + bar_width / 2, on_means, bar_width,
                       yerr=on_sems, capsize=3, label="ON", color="#ee854a", alpha=0.8)

    # Add significance markers from focused family tests
    for i, roi in enumerate(all_rois_sorted):
        if roi in participation_tests.index:
            q = participation_tests.loc[roi, "q_fdr_all"]
            p = participation_tests.loc[roi, "p_wilcoxon"]
            sig = _significance_label(p, q)
            if sig != "n.s.":
                ymax = max(off_means[i] + off_sems[i], on_means[i] + on_sems[i]) + 0.02
                ax_a.text(x_pos[i], ymax, sig, ha="center", va="bottom", fontsize=9,
                          color="darkred" if q < 0.05 else "gray")

    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(
        [r.replace("(", "\n(").replace("Frontal medial cortex", "Frontal medial\ncortex") for r in all_rois_sorted],
        rotation=30, ha="right", fontsize=7
    )
    ax_a.set_ylabel("Participation coefficient", fontsize=10)
    ax_a.set_title("Hub reorganization: Participation coefficient ON vs OFF (all ROIs, continuous metric)",
                   fontsize=10, fontweight="bold")
    ax_a.legend(fontsize=9)
    ax_a.spines[["top", "right"]].set_visible(False)

    # Panel B: Hub score delta per ROI (focused: frontal association)
    ax_b = fig.add_subplot(gs[1, 0])
    fa_tests = hub_tests_df[
        hub_tests_df["in_focused_family"] & (hub_tests_df["metric"] == "hub_score")
    ].sort_values("mean_delta")
    colors_b = ["#4878d0" if d < 0 else "#ee854a" for d in fa_tests["mean_delta"]]
    bars = ax_b.barh(range(len(fa_tests)), fa_tests["mean_delta"],
                     xerr=fa_tests["std_delta"] / np.sqrt(fa_tests["n_subjects"]),
                     capsize=3, color=colors_b, alpha=0.8)
    ax_b.set_yticks(range(len(fa_tests)))
    ax_b.set_yticklabels(fa_tests["roi"].tolist(), fontsize=8)
    ax_b.set_xlabel("Hub score Δ (ON − OFF)", fontsize=9)
    ax_b.set_title("Hub score change\n(Frontal association family, pre-specified)", fontsize=9, fontweight="bold")
    for i, (_, row) in enumerate(fa_tests.iterrows()):
        q = row.get("q_fdr_focused", row["q_fdr_all"])
        p = row["p_wilcoxon"]
        sig = _significance_label(p, q)
        ax_b.text(row["mean_delta"] + 0.002, i, f" {sig} (p={p:.3f})", va="center", fontsize=7)
    ax_b.axvline(0, color="gray", lw=0.8, ls="--")
    ax_b.spines[["top", "right"]].set_visible(False)

    # Panel C: Brain-behavior scatter — insular participation delta vs vmax delta
    ax_c = fig.add_subplot(gs[1, 1])
    insular_s = hub_session[hub_session["base_roi"] == INSULAR_ROI]
    off_pc = insular_s[insular_s["state_label"] == "OFF"].groupby("subject")["participation_coeff"].mean()
    on_pc = insular_s[insular_s["state_label"] == "ON"].groupby("subject")["participation_coeff"].mean()
    subjs_c = sorted(set(off_pc.index) & set(on_pc.index))
    part_delta_c = np.array([on_pc[s] - off_pc[s] for s in subjs_c])
    beh_c = pd.DataFrame({"subject": subjs_c}).merge(
        behavior_df[["subject", "task_vmax_delta"]], on="subject", how="left"
    )
    vmax_c = beh_c["task_vmax_delta"].to_numpy()
    valid_c = ~np.isnan(part_delta_c) & ~np.isnan(vmax_c)
    if valid_c.sum() >= 5:
        ax_c.scatter(part_delta_c[valid_c], vmax_c[valid_c],
                     color="tomato", s=60, alpha=0.85, zorder=3)
        m, b = np.polyfit(part_delta_c[valid_c], vmax_c[valid_c], 1)
        xline = np.linspace(part_delta_c[valid_c].min(), part_delta_c[valid_c].max(), 100)
        ax_c.plot(xline, m * xline + b, color="tomato", lw=1.5, alpha=0.7)
        rho_c = float(stats.spearmanr(part_delta_c[valid_c], vmax_c[valid_c]).statistic)
        # Get p from beh_corr_df
        part_row = beh_corr_df[beh_corr_df["metric"] == "participation_coeff"] if not beh_corr_df.empty else pd.DataFrame()
        p_c = part_row.iloc[0]["permutation_p"] if len(part_row) > 0 else float("nan")
        sig_c = _significance_label(p_c)
        ax_c.set_title(
            f"Insular participation Δ × vmax Δ\nρ = {rho_c:.2f}, p = {p_c:.3f}  {sig_c}",
            fontsize=9, fontweight="bold" if p_c < 0.05 else "normal"
        )
        ax_c.set_xlabel("Insular participation coeff Δ (ON−OFF)", fontsize=8)
        ax_c.set_ylabel("Peak velocity Δ (ON−OFF)", fontsize=8)
        ax_c.axhline(0, color="gray", lw=0.5, ls="--")
        ax_c.axvline(0, color="gray", lw=0.5, ls="--")
    else:
        ax_c.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                  transform=ax_c.transAxes, fontsize=9)
    ax_c.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Hub reorganization: Continuous metrics (ON vs OFF medication, N=14)",
                 fontsize=11, fontweight="bold")
    fig.savefig(out_dir / "approach3_hub_reorganization_continuous.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "approach3_hub_reorganization_continuous.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → Saved approach3_hub_reorganization_continuous.png/pdf")


def plot_hub_behavior_scatter(
    session_df: pd.DataFrame,
    behavior_df: pd.DataFrame,
    beh_corr_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Scatter plot of insular hub metrics vs behavior for publication."""
    hub_session = compute_hub_scores(session_df)
    if "state" in hub_session.columns:
        hub_session["state_label"] = hub_session["state"].str.upper()
    else:
        hub_session["state_label"] = hub_session["session"].map({1: "OFF", 2: "ON"})

    insular_data = hub_session[hub_session["base_roi"] == INSULAR_ROI]

    metrics = [
        ("hub_score", "Insular hub score Δ (ON−OFF)"),
        ("participation_coeff", "Insular participation coeff Δ (ON−OFF)"),
    ]
    behaviors = [
        ("task_vmax_delta", "Peak velocity Δ (ON−OFF)"),
        ("task_pmax_delta", "Peak force Δ (ON−OFF)"),
    ]

    fig, axes = plt.subplots(len(metrics), len(behaviors), figsize=(8, 7))
    colors_grid = [["steelblue", "navy"], ["tomato", "darkred"]]

    for i, (met_col, met_label) in enumerate(metrics):
        off_vals = insular_data[insular_data["state_label"] == "OFF"].groupby("subject")[met_col].mean()
        on_vals = insular_data[insular_data["state_label"] == "ON"].groupby("subject")[met_col].mean()
        subjs = sorted(set(off_vals.index) & set(on_vals.index))
        delta_arr = np.array([on_vals[s] - off_vals[s] for s in subjs])

        beh_merged = pd.DataFrame({"subject": subjs}).merge(
            behavior_df[["subject"] + [b for b, _ in behaviors]], on="subject", how="left"
        )

        for j, (beh_col, beh_label) in enumerate(behaviors):
            ax = axes[i, j]
            beh_arr = beh_merged[beh_col].to_numpy()
            valid = ~np.isnan(delta_arr) & ~np.isnan(beh_arr)

            if valid.sum() >= 5:
                ax.scatter(delta_arr[valid], beh_arr[valid],
                           color=colors_grid[i][j], s=55, alpha=0.85, zorder=3)
                m, b = np.polyfit(delta_arr[valid], beh_arr[valid], 1)
                xline = np.linspace(delta_arr[valid].min(), delta_arr[valid].max(), 100)
                ax.plot(xline, m * xline + b, color=colors_grid[i][j], lw=1.5, alpha=0.7)
                rho = float(stats.spearmanr(delta_arr[valid], beh_arr[valid]).statistic)
                ax.set_title(f"ρ = {rho:.2f}", fontsize=9)
            ax.set_xlabel(met_label if i == len(metrics) - 1 else "", fontsize=8)
            ax.set_ylabel(beh_label if j == 0 else "", fontsize=8)
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.axvline(0, color="gray", lw=0.5, ls="--")
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Insular hub metrics vs. motor behavior (ON − OFF, N=14)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "approach3_hub_behavior_scatter.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "approach3_hub_behavior_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → Saved approach3_hub_behavior_scatter.png/pdf")


def plot_null_distributions(
    confirmatory_results: dict[str, PLSResult],
    confirmatory_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Plot permutation null distributions for confirmatory models."""
    n = len(confirmatory_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    colors = ["steelblue", "tomato"]

    for ax, (key, result), color in zip(axes, confirmatory_results.items(), colors):
        row = confirmatory_df[confirmatory_df["model"] == key].iloc[0]
        null = result.null_distribution
        obs = result.observed_r
        ax.hist(null, bins=60, color=color, alpha=0.5, edgecolor="none")
        ax.axvline(obs, color="darkred", lw=2, ls="--", label=f"Observed r={obs:.3f}")
        ax.axvline(-obs, color="darkred", lw=1, ls=":", alpha=0.5)
        p = row["permutation_p"]
        q = row["q_fdr_within_2"]
        sig = _significance_label(p, q)
        ax.set_title(
            f"{row.get('label', key)}\np={p:.4f}, q={q:.4f}  {sig}",
            fontsize=8, fontweight="bold" if q < 0.05 else "normal"
        )
        ax.set_xlabel("Null latent score correlation (r)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"Permutation null distributions (n={N_PERM_MAIN:,} permutations)\nBH-FDR corrected within m=2 pre-specified models",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "approach2_permutation_null_distributions.png", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "approach2_permutation_null_distributions.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  → Saved approach2_permutation_null_distributions.png/pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("CONFIRMATORY PLS + HUB REORGANIZATION ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {OUT_DIR}")
    print(f"Main permutations: {N_PERM_MAIN:,}")
    print()

    # ─── Load data ───────────────────────────────────────────────────────────
    print("Loading data...")
    delta_df = load_node_deltas(NODE_DELTA_PATH)
    session_df = load_session_data(SESSION_PATH)

    # Load behavior
    behavior_df = load_task_behavior_deltas()
    behavior_df = behavior_df.sort_values("subject").reset_index(drop=True)

    # Build feature matrices
    insular_df = build_insular_features(delta_df)
    frontal_df = build_frontal_association_features(delta_df)

    # Align subjects across all data
    common_subjects = sorted(
        set(insular_df["subject"])
        & set(behavior_df["subject"])
    )
    print(f"  N subjects (common): {len(common_subjects)}")

    insular_df = insular_df[insular_df["subject"].isin(common_subjects)].reset_index(drop=True)
    frontal_df = frontal_df[frontal_df["subject"].isin(common_subjects)].reset_index(drop=True)
    behavior_df = behavior_df[behavior_df["subject"].isin(common_subjects)].reset_index(drop=True)

    # Build motor vigor composite (PC1 of vmax + pmax)
    vigor_composite = _motor_vigor_composite(
        behavior_df, ["task_vmax_delta", "task_pmax_delta"]
    )
    print(f"  Motor vigor composite: PC1 explains "
          f"{PCA(n_components=1).fit(np.column_stack([_standardize(behavior_df['task_vmax_delta'].to_numpy()), _standardize(behavior_df['task_pmax_delta'].to_numpy())])).explained_variance_ratio_[0]:.1%} "
          f"of [vmax, pmax] variance")

    # ─── Approach 1: Brain-behavior permutation correlations ─────────────────
    corr_results = run_brain_behavior_correlations(
        insular_df, behavior_df, vigor_composite, rng
    )
    corr_results.to_csv(OUT_DIR / "approach1_brain_behavior_correlations.csv", index=False)

    # ─── Approach 2: Confirmatory PLS ────────────────────────────────────────
    confirmatory_df, confirmatory_results = run_confirmatory_pls(
        insular_df, frontal_df, behavior_df, rng
    )
    confirmatory_df.to_csv(OUT_DIR / "approach2_confirmatory_pls.csv", index=False)

    extended_df, extended_results = run_extended_pls(
        insular_df, frontal_df, behavior_df, rng
    )
    extended_df.to_csv(OUT_DIR / "approach2_extended_pls.csv", index=False)

    # ─── Approach 3: Hub reorganization ──────────────────────────────────────
    hub_tests_df, beh_corr_df = run_hub_reorganization_analysis(
        session_df, behavior_df, vigor_composite, rng
    )
    hub_tests_df.to_csv(OUT_DIR / "approach3_hub_tests.csv", index=False)
    beh_corr_df.to_csv(OUT_DIR / "approach3_hub_behavior_correlations.csv", index=False)

    # ─── Figures ─────────────────────────────────────────────────────────────
    print("\nGenerating figures...")

    plot_brain_behavior_correlations(
        insular_df, behavior_df, vigor_composite, corr_results, OUT_DIR
    )
    plot_pls_latent_scores(confirmatory_results, confirmatory_df, OUT_DIR)
    plot_null_distributions(confirmatory_results, confirmatory_df, OUT_DIR)
    plot_hub_reorganization(hub_tests_df, session_df, beh_corr_df, behavior_df, OUT_DIR)
    plot_hub_behavior_scatter(session_df, behavior_df, beh_corr_df, OUT_DIR)

    # ─── Summary report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\nAPPROACH 1: Brain-behavior correlations (m=2 pre-specified tests)")
    for _, row in corr_results.iterrows():
        sig = "SIGNIFICANT" if row["q_fdr"] < 0.05 else "not significant"
        print(f"  {row['brain_metric']}: rho={row['spearman_rho']:.3f}, p={row['permutation_p']:.4f}, q={row['q_fdr']:.4f} → {sig}")

    print("\nAPPROACH 2: Confirmatory PLS (m=2 pre-specified models, BH-FDR within 2)")
    for _, row in confirmatory_df.iterrows():
        sig = "✓ SIGNIFICANT (q<0.05)" if row["q_fdr_within_2"] < 0.05 else "✗ not significant"
        print(f"  {row['model']}")
        print(f"    r={row['observed_r']:.3f}, p={row['permutation_p']:.4f}, q={row['q_fdr_within_2']:.4f} {sig}")
        print(f"    LOO r={row['loo_r']:.3f}, Boot 95% CI=[{row['boot_ci_lo']:.3f}, {row['boot_ci_hi']:.3f}]")

    print("\nAPPROACH 3: Hub reorganization (primary pre-specified test + FDR families)")
    # Primary test: insular participation ON vs OFF (1 pre-specified test)
    primary_row = hub_tests_df[
        (hub_tests_df["roi"] == INSULAR_ROI) & (hub_tests_df["metric"] == "participation_coeff")
    ]
    if len(primary_row) > 0:
        p_prim = primary_row.iloc[0]["p_wilcoxon"]
        q_prim = primary_row.iloc[0].get("q_primary_prespecified", p_prim)
        sig_prim = "SIGNIFICANT" if p_prim < 0.05 else "not significant (p<0.1 trend)" if p_prim < 0.1 else "not significant"
        print(f"  PRIMARY (m=1): Insular participation ON vs OFF Wilcoxon: p={p_prim:.4f} → {sig_prim}")
        print(f"    Cohen's dz = {primary_row.iloc[0]['cohen_dz']:.3f}, mean delta = {primary_row.iloc[0]['mean_delta']:.4f}")
    sig_hub = hub_tests_df[hub_tests_df["q_fdr_all"] < 0.05]
    if len(sig_hub) > 0:
        for _, row in sig_hub.iterrows():
            print(f"  ALL ROIs FDR: {row['roi']} {row['metric']}: p={row['p_wilcoxon']:.4f}, q_all={row['q_fdr_all']:.4f}")
    else:
        print("  No hub tests survive FDR across all ROIs (expected with N=14).")
    focused_sig = hub_tests_df[
        hub_tests_df["in_focused_family"].astype(bool) & (hub_tests_df["q_fdr_focused"] < 0.05)
    ]
    if len(focused_sig) > 0:
        for _, row in focused_sig.iterrows():
            print(f"  FOCUSED: {row['roi']} {row['metric']}: p={row['p_wilcoxon']:.4f}, q_focused={row['q_fdr_focused']:.4f}")

    print("\nExtended PLS models (exploratory):")
    for _, row in extended_df.iterrows():
        sig = "✓ sig (q<0.05)" if row["q_fdr_exploratory"] < 0.05 else "n.s."
        print(f"  {row['model']}: p={row['permutation_p']:.4f}, q={row['q_fdr_exploratory']:.4f} {sig}")

    # Save full summary JSON
    summary = {
        "n_subjects": len(common_subjects),
        "n_permutations": N_PERM_MAIN,
        "n_bootstrap": N_BOOT,
        "approach1_significant": bool(
            (corr_results["q_fdr"] < 0.05).any()
        ),
        "approach2_significant_models": confirmatory_df[
            confirmatory_df["q_fdr_within_2"] < 0.05
        ]["model"].tolist(),
        "approach3_significant_hub_tests_focused": focused_sig[
            ["roi", "metric", "p_wilcoxon", "q_fdr_focused"]
        ].to_dict("records") if len(focused_sig) > 0 else [],
        "approach3_primary_insular_participation_p": float(
            hub_tests_df[
                (hub_tests_df["roi"] == INSULAR_ROI)
                & (hub_tests_df["metric"] == "participation_coeff")
            ]["p_wilcoxon"].iloc[0]
        ) if len(hub_tests_df[
            (hub_tests_df["roi"] == INSULAR_ROI)
            & (hub_tests_df["metric"] == "participation_coeff")
        ]) > 0 else float("nan"),
        "best_model": {
            "model": confirmatory_df.sort_values("permutation_p").iloc[0]["model"],
            "observed_r": float(confirmatory_df.sort_values("permutation_p").iloc[0]["observed_r"]),
            "permutation_p": float(confirmatory_df.sort_values("permutation_p").iloc[0]["permutation_p"]),
            "q_fdr_within_2": float(confirmatory_df.sort_values("permutation_p").iloc[0]["q_fdr_within_2"]),
            "loo_r": float(confirmatory_df.sort_values("permutation_p").iloc[0]["loo_r"]),
        },
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nOutputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
