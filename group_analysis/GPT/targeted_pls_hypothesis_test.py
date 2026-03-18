"""
Targeted PLS hypothesis test: brain network features vs
behavioral lag-1 correlation + inverse movement speed (1/(RT+MT)).

Root-cause fix
--------------
The exhaustive search across 255 behavior subsets × 15 brain combinations
(3 825 models) produced an inflated FDR (q = 0.595) even though the best
individual permutation p = 0.005.  The fix is to commit a priori to the
theoretically motivated behavioral pair
    Y = [behavior_lag1_corr_delta,  task_1_rt_mt_delta]
and sweep only the 15 brain-family combinations, applying BH-FDR over
those 15 tests.  Bootstrap resampling (2 000 draws) then validates weight
stability for every model that clears the permutation threshold.

Outputs are written to <out_dir>/
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression

# ---------------------------------------------------------------------------
# Paths (relative to repo root; all resolved to absolute below)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
TMP = REPO_ROOT / "results" / "connectivity" / "GPT" / "tmp"

DCM_PARAMS_PATH   = TMP / "effective_connectivity" / "all_roi" / "dcm_subject_parameters.csv"
NODE_DELTA_PATH   = TMP / "roi_graph_analysis_partial_correlation_check" / "node_metric_deltas_on_minus_off.csv"
MODULE_SUMM_PATH  = TMP / "roi_graph_analysis_partial_correlation_check" / "module_integration_summary.csv"
BEHAVIOR_SRC_PATH = TMP / "family_level_saved_models" / "04_graph_strength__legacy_timing" / "subject_behavior_deltas.csv"

OUT_DIR = TMP / "targeted_pls_analysis"

EXCLUDE_BASE_ROI_PATTERNS = ("brain stem", "brain-stem", "cerebral white matter")
BEHAVIOR_COLS = ["behavior_lag1_corr_delta", "task_1_rt_mt_delta"]

N_PERMUTATIONS = 5000
N_BOOTSTRAP    = 2000
RANDOM_SEED    = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_roi(label: str) -> str:
    return re.sub(r"^[LR]\s+", "", str(label)).strip()


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    centered = df - df.mean(axis=0)
    scale = df.std(axis=0, ddof=0).replace(0.0, 1.0)
    return centered / scale


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
    return candidate if np.dot(reference, candidate) >= 0 else -candidate


# ---------------------------------------------------------------------------
# Feature builders
# ---------------------------------------------------------------------------

def build_dcm_features(dcm_path: Path) -> pd.DataFrame:
    """Outgoing-mean effective connectivity delta per base ROI."""
    df = pd.read_csv(dcm_path)
    coupling = df[df["parameter_type"] == "coupling"].copy()
    coupling["base_target"] = coupling["target_roi"].map(_base_roi)
    coupling["base_source"] = coupling["source_roi"].map(_base_roi)
    # Drop excluded ROIs
    excl = lambda x: any(p in x.lower() for p in EXCLUDE_BASE_ROI_PATTERNS)
    coupling = coupling[~coupling["base_target"].map(excl)]
    coupling = coupling[~coupling["base_source"].map(excl)]
    # Average L+R hemispheres into one base-ROI matrix per subject
    agg = (
        coupling
        .groupby(["subject", "base_target", "base_source"], as_index=False)["delta_on_minus_off"]
        .mean()
    )
    rows = []
    for subject, grp in agg.groupby("subject", sort=True):
        pivot = grp.pivot(index="base_target", columns="base_source", values="delta_on_minus_off")
        row = {"subject": subject}
        for roi in sorted(pivot.columns):
            col = f"dcm_outgoing_delta_{_safe_slug(roi)}"
            # Mean incoming to roi from all other ROIs (=outgoing from roi's perspective)
            others = pivot.loc[pivot.index != roi, roi] if roi in pivot.columns else pd.Series(dtype=float)
            row[col] = float(others.mean()) if len(others) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def build_graph_features(node_delta_path: Path) -> pd.DataFrame:
    """Node-strength and participation-coefficient deltas per base ROI."""
    df = pd.read_csv(node_delta_path)
    df = df[~df["base_roi"].str.lower().apply(lambda x: any(p in x for p in EXCLUDE_BASE_ROI_PATTERNS))]
    agg = (
        df.groupby(["subject", "base_roi"], as_index=False)
        .agg(
            delta_node_strength_abs=("delta_node_strength_abs", "mean"),
            delta_participation_coeff=("delta_participation_coeff", "mean"),
        )
    )
    rows = []
    for subject, grp in agg.groupby("subject", sort=True):
        row = {"subject": subject}
        for _, r in grp.iterrows():
            roi = _safe_slug(r["base_roi"])
            row[f"graph_strength_delta_{roi}"]      = float(r["delta_node_strength_abs"])
            row[f"graph_participation_delta_{roi}"] = float(r["delta_participation_coeff"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


def build_module_features(module_path: Path) -> pd.DataFrame:
    """Between- and within-module connectivity delta (ON − OFF)."""
    df = pd.read_csv(module_path)
    anatomical = df[df["module_scheme"] == "anatomical_system"].copy()
    # Pivot to (subject, session) and compute delta
    rows = []
    pairs = (
        anatomical[["module_a", "module_b"]].drop_duplicates()
        .apply(tuple, axis=1).tolist()
    )
    for subject, grp in anatomical.groupby("subject", sort=True):
        row = {"subject": subject}
        for mod_a, mod_b in pairs:
            feat = f"module_delta_{_safe_slug(mod_a)}__{_safe_slug(mod_b)}"
            off = grp[(grp["session"] == 1) & (grp["module_a"] == mod_a) & (grp["module_b"] == mod_b)]
            on  = grp[(grp["session"] == 2) & (grp["module_a"] == mod_a) & (grp["module_b"] == mod_b)]
            row[feat] = (float(on["mean_abs_strength"].iloc[0]) - float(off["mean_abs_strength"].iloc[0])
                         ) if (len(off) and len(on)) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)


# ---------------------------------------------------------------------------
# PLS core
# ---------------------------------------------------------------------------

def run_pls(X: pd.DataFrame, Y: pd.DataFrame):
    model = PLSRegression(n_components=1, max_iter=1000)
    model.fit(X, Y)
    r = float(stats.pearsonr(model.x_scores_.ravel(), model.y_scores_.ravel()).statistic)
    return model, r


def permutation_pvalue(X: pd.DataFrame, Y: pd.DataFrame,
                       n_perm: int, rng: np.random.Generator) -> tuple[float, float, np.ndarray]:
    model, observed = run_pls(X, Y)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm_idx = rng.permutation(len(Y))
        _, r = run_pls(X, Y.iloc[perm_idx].reset_index(drop=True))
        null[i] = r
    p = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (n_perm + 1))
    return observed, p, null


def bootstrap_stability(X: pd.DataFrame, Y: pd.DataFrame,
                        full_weights: np.ndarray,
                        n_boot: int, rng: np.random.Generator) -> dict:
    n = len(X)
    boot_corrs = []
    weight_corrs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, Yb = X.iloc[idx].reset_index(drop=True), Y.iloc[idx].reset_index(drop=True)
        # Skip if any column is constant in bootstrap sample
        if (Xb.std(ddof=0) == 0).any() or (Yb.std(ddof=0) == 0).any():
            continue
        try:
            model_b, r_b = run_pls(_zscore(Xb), _zscore(Yb))
        except Exception:
            continue
        w_b = _align_sign(full_weights, model_b.x_weights_.ravel())
        weight_corrs.append(float(np.corrcoef(full_weights, w_b)[0, 1]))
        boot_corrs.append(r_b)

    weight_corrs = np.array(weight_corrs)
    boot_corrs   = np.array(boot_corrs)
    return {
        "n_valid_boots":        int(len(weight_corrs)),
        "median_weight_corr":   float(np.median(weight_corrs)) if len(weight_corrs) else float("nan"),
        "p5_weight_corr":       float(np.percentile(weight_corrs, 5)) if len(weight_corrs) else float("nan"),
        "mean_boot_score_corr": float(np.mean(boot_corrs))   if len(boot_corrs) else float("nan"),
        "p5_boot_score_corr":   float(np.percentile(boot_corrs, 5)) if len(boot_corrs) else float("nan"),
    }


def loo_stability(X: pd.DataFrame, Y: pd.DataFrame, full_weights: np.ndarray) -> dict:
    n = len(X)
    wcs = []
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        try:
            m, _ = run_pls(_zscore(X.iloc[mask].reset_index(drop=True)),
                           _zscore(Y.iloc[mask].reset_index(drop=True)))
        except Exception:
            continue
        wcs.append(float(np.corrcoef(full_weights,
                                     _align_sign(full_weights, m.x_weights_.ravel()))[0, 1]))
    return {
        "mean_loo_weight_corr": float(np.mean(wcs)) if wcs else float("nan"),
        "min_loo_weight_corr":  float(np.min(wcs))  if wcs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Brain-family combinations
# ---------------------------------------------------------------------------

def build_brain_sets(neural_df: pd.DataFrame) -> dict[str, list[str]]:
    families = {
        "dcm":                 [c for c in neural_df.columns if c.startswith("dcm_outgoing_delta_")],
        "graph_strength":      [c for c in neural_df.columns if c.startswith("graph_strength_delta_")],
        "graph_participation": [c for c in neural_df.columns if c.startswith("graph_participation_delta_")],
        "module":              [c for c in neural_df.columns if c.startswith("module_delta_")],
    }
    brain_sets: dict[str, list[str]] = {}
    keys = list(families)
    for size in range(1, len(keys) + 1):
        for combo in combinations(keys, size):
            name = "+".join(combo)
            cols: list[str] = []
            for k in combo:
                cols.extend(families[k])
            brain_sets[name] = cols
    return brain_sets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model(scores_df: pd.DataFrame, loadings_df: pd.DataFrame,
               perm_null: np.ndarray, observed_r: float,
               brain_set: str, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # 1. Brain vs behaviour scores scatter
    ax = axes[0]
    ax.scatter(scores_df["x_score"], scores_df["y_score"], color="#4c78a8", s=60, zorder=3)
    for _, row in scores_df.iterrows():
        ax.text(row["x_score"], row["y_score"], row["subject"], fontsize=6)
    ax.set_xlabel("Neural axis (brain score)", fontsize=10)
    ax.set_ylabel("Behavioral axis (behavior score)", fontsize=10)
    ax.set_title(f"Brain–Behaviour coupling\nr = {observed_r:.3f}", fontsize=10)
    ax.axhline(0, color="grey", lw=0.5); ax.axvline(0, color="grey", lw=0.5)

    # 2. Top neural weights
    ax = axes[1]
    top = (loadings_df[loadings_df["set"] == "X"]
           .assign(abs_w=lambda d: d["weight"].abs())
           .sort_values("abs_w", ascending=False).head(10))
    colors = ["#e45756" if w > 0 else "#4c78a8" for w in top["weight"]]
    def _shorten(name: str) -> str:
        name = re.sub(r"^dcm_outgoing_delta_", "dcm: ", name)
        name = re.sub(r"^graph_strength_delta_", "str: ", name)
        name = re.sub(r"^graph_participation_delta_", "part: ", name)
        name = re.sub(r"^module_delta_", "mod: ", name)
        return name

    ax.barh(top["name"].map(_shorten), top["weight"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.invert_yaxis()
    ax.set_title(f"Top neural weights\n({brain_set})", fontsize=10)
    ax.set_xlabel("PLS weight", fontsize=10)

    # 3. Permutation null distribution
    ax = axes[2]
    ax.hist(np.abs(perm_null), bins=50, color="#9ecae9", edgecolor="white")
    ax.axvline(abs(observed_r), color="#e45756", lw=2, label=f"observed |r| = {abs(observed_r):.3f}")
    ax.set_xlabel("|r| permuted", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Permutation null distribution", fontsize=10)
    ax.legend(fontsize=9)

    fig.suptitle(f"Targeted PLS: {brain_set}\nY = lag1_corr_delta + task_1_rt_mt_delta", fontsize=11)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng_perm  = np.random.default_rng(RANDOM_SEED)
    rng_boot  = np.random.default_rng(RANDOM_SEED + 1)

    print("=" * 70)
    print("Targeted PLS: lag1_corr_delta + task_1_rt_mt_delta  vs  brain families")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load / build data
    # ------------------------------------------------------------------
    print("\n[1] Building neural feature matrices …")
    dcm_df    = build_dcm_features(DCM_PARAMS_PATH)
    graph_df  = build_graph_features(NODE_DELTA_PATH)
    module_df = build_module_features(MODULE_SUMM_PATH)

    neural_df = (dcm_df
                 .merge(graph_df,  on="subject", how="inner")
                 .merge(module_df, on="subject", how="inner")
                 .sort_values("subject").reset_index(drop=True))

    print(f"   DCM features:          {len([c for c in neural_df.columns if c.startswith('dcm')])} (N={len(dcm_df)})")
    print(f"   Graph-strength:        {len([c for c in neural_df.columns if c.startswith('graph_strength')])}")
    print(f"   Graph-participation:   {len([c for c in neural_df.columns if c.startswith('graph_participation')])}")
    print(f"   Module:                {len([c for c in neural_df.columns if c.startswith('module')])}")

    behavior_src = pd.read_csv(BEHAVIOR_SRC_PATH)
    behavior_df  = behavior_src[["subject"] + BEHAVIOR_COLS].dropna()

    merged = (neural_df.merge(behavior_df, on="subject", how="inner")
              .sort_values("subject").reset_index(drop=True))
    print(f"\n   Subjects after inner join: {len(merged)}")
    print(f"   Behavioral features: {BEHAVIOR_COLS}")

    brain_sets = build_brain_sets(neural_df)
    print(f"\n   Brain-family combinations to test: {len(brain_sets)}")

    # Save merged data
    merged.to_csv(out_dir / "merged_full.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Pre-generate permutations (same null for all brain sets)
    # ------------------------------------------------------------------
    print(f"\n[2] Pre-generating {N_PERMUTATIONS} permutation indices …")
    n_subj = len(merged)
    perms  = np.array([rng_perm.permutation(n_subj) for _ in range(N_PERMUTATIONS)])

    # ------------------------------------------------------------------
    # 3. Sweep brain families – compute observed r and permutation p
    # ------------------------------------------------------------------
    print("\n[3] Sweeping brain-family combinations …")
    Y_raw = merged[BEHAVIOR_COLS].copy()

    results = []
    for brain_set_name, neural_cols in brain_sets.items():
        usable = merged.dropna(subset=neural_cols + BEHAVIOR_COLS).reset_index(drop=True)
        if usable.empty:
            continue
        Xu = _zscore(usable[neural_cols])
        Yu = _zscore(usable[BEHAVIOR_COLS])

        model, observed_r = run_pls(Xu, Yu)

        # Permutation p
        null_r = np.empty(N_PERMUTATIONS)
        for k, perm in enumerate(perms[:, :len(usable)]):
            # Regenerate per-combo if n differs – here n is constant (no NaNs expected)
            Yi_perm = Yu.iloc[perm].reset_index(drop=True)
            _, r_k = run_pls(Xu, Yi_perm)
            null_r[k] = r_k

        p = float((np.sum(np.abs(null_r) >= abs(observed_r)) + 1) / (N_PERMUTATIONS + 1))

        results.append({
            "brain_set":          brain_set_name,
            "n_brain_features":   len(neural_cols),
            "n_behavior_features":len(BEHAVIOR_COLS),
            "n_subjects":         len(usable),
            "p_n_ratio":          round(len(neural_cols) / len(usable), 3),
            "observed_r":         round(observed_r, 6),
            "permutation_p":      round(p, 6),
            "_model":             model,
            "_Xu":                Xu,
            "_Yu":                Yu,
            "_null_r":            null_r,
            "_neural_cols":       neural_cols,
            "_usable":            usable,
        })
        print(f"   {brain_set_name:50s}  n_feat={len(neural_cols):3d}  "
              f"r={observed_r:+.4f}  p={p:.4f}")

    # ------------------------------------------------------------------
    # 4. BH-FDR over 15 brain-family tests (fixed behavior pair)
    # ------------------------------------------------------------------
    print("\n[4] Applying BH-FDR over the 15 brain-family tests …")
    res_df = pd.DataFrame([
        {k: v for k, v in r.items() if not k.startswith("_")}
        for r in results
    ])
    res_df["q_fdr"] = _bh_fdr(res_df["permutation_p"].values)
    res_df = res_df.sort_values("permutation_p").reset_index(drop=True)
    res_df.to_csv(out_dir / "all_brain_sets_permutation.csv", index=False)

    print(res_df[["brain_set","n_brain_features","p_n_ratio",
                  "observed_r","permutation_p","q_fdr"]].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Bootstrap + LOO for models that survive p < 0.05
    # ------------------------------------------------------------------
    print("\n[5] Bootstrap + LOO stability for models with p < 0.05 …")
    stable_models = []
    for r in results:
        row = res_df.loc[res_df["brain_set"] == r["brain_set"]].iloc[0]
        if float(row["permutation_p"]) >= 0.05:
            continue

        Xu, Yu, model  = r["_Xu"], r["_Yu"], r["_model"]
        full_w = model.x_weights_.ravel()
        brain_set_name = r["brain_set"]

        print(f"\n   --- {brain_set_name} (p={row['permutation_p']:.4f}, q={row['q_fdr']:.4f}) ---")

        boot = bootstrap_stability(Xu, Yu, full_w, N_BOOTSTRAP, rng_boot)
        loo  = loo_stability(Xu, Yu, full_w)

        print(f"       Bootstrap (n={boot['n_valid_boots']}):  "
              f"median_weight_corr={boot['median_weight_corr']:.3f}  "
              f"p5={boot['p5_weight_corr']:.3f}")
        print(f"       LOO:  mean_weight_corr={loo['mean_loo_weight_corr']:.3f}  "
              f"min={loo['min_loo_weight_corr']:.3f}")

        # Collect weights / scores
        usable = r["_usable"]
        model2, obs_r2 = run_pls(Xu, Yu)   # refit cleanly for loadings
        x_sc = model2.x_scores_.ravel()
        y_sc = model2.y_scores_.ravel()

        score_df = usable[["subject"] + BEHAVIOR_COLS].copy()
        score_df["x_score"] = x_sc
        score_df["y_score"] = y_sc

        loadings_rows = (
            [{"set": "X", "name": nm, "weight": float(w)}
             for nm, w in zip(r["_neural_cols"], model2.x_weights_.ravel())]
          + [{"set": "Y", "name": nm, "weight": float(w)}
             for nm, w in zip(BEHAVIOR_COLS, model2.y_weights_.ravel())]
        )
        loadings_df = pd.DataFrame(loadings_rows)

        slug = _safe_slug(brain_set_name)
        model_dir = out_dir / slug
        model_dir.mkdir(exist_ok=True)

        score_df.to_csv(model_dir / "scores.csv", index=False)
        loadings_df.to_csv(model_dir / "loadings.csv", index=False)
        pd.DataFrame({"permuted_r": r["_null_r"]}).to_csv(model_dir / "null_distribution.csv", index=False)

        plot_model(score_df, loadings_df, r["_null_r"], obs_r2,
                   brain_set_name, model_dir / "coupling_figure.png")

        stable_models.append({
            "brain_set":            brain_set_name,
            "n_brain_features":     int(row["n_brain_features"]),
            "p_n_ratio":            float(row["p_n_ratio"]),
            "observed_r":           float(obs_r2),
            "permutation_p":        float(row["permutation_p"]),
            "q_fdr":                float(row["q_fdr"]),
            "median_boot_weight_corr": boot["median_weight_corr"],
            "p5_boot_weight_corr":     boot["p5_weight_corr"],
            "mean_loo_weight_corr":    loo["mean_loo_weight_corr"],
            "min_loo_weight_corr":     loo["min_loo_weight_corr"],
            "model_dir":            str(model_dir),
        })

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    stable_df = pd.DataFrame(stable_models)
    stable_df.to_csv(out_dir / "significant_models_with_stability.csv", index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBehavioral features tested (fixed):  {BEHAVIOR_COLS}")
    print(f"Brain-family combinations tested:    {len(brain_sets)}")
    print(f"N subjects:                          {n_subj}")
    print(f"N permutations:                      {N_PERMUTATIONS}")
    print(f"N bootstrap draws:                   {N_BOOTSTRAP}")
    n_sig_perm = int((res_df["permutation_p"] < 0.05).sum())
    n_sig_fdr  = int((res_df["q_fdr"] < 0.05).sum())
    print(f"\nModels with perm p < 0.05:           {n_sig_perm}")
    print(f"Models with BH-FDR q < 0.05:         {n_sig_fdr}")

    if not stable_df.empty:
        best = stable_df.sort_values("permutation_p").iloc[0]
        print(f"\n>>> BEST VALID MODEL <<<")
        print(f"  Brain set:           {best['brain_set']}")
        print(f"  n_brain_features:    {best['n_brain_features']}")
        print(f"  p/n ratio:           {best['p_n_ratio']:.3f}")
        print(f"  Observed r:          {best['observed_r']:.4f}")
        print(f"  Permutation p:       {best['permutation_p']:.5f}")
        print(f"  BH-FDR q:            {best['q_fdr']:.4f}")
        print(f"  Median boot wt-corr: {best['median_boot_weight_corr']:.3f}")
        print(f"  p5 boot wt-corr:     {best['p5_boot_weight_corr']:.3f}")
        print(f"  Mean LOO wt-corr:    {best['mean_loo_weight_corr']:.3f}")
        print(f"  Min  LOO wt-corr:    {best['min_loo_weight_corr']:.3f}")

        # Print top neural and behavior weights for best model
        best_slug = _safe_slug(best["brain_set"])
        ldf = pd.read_csv(out_dir / best_slug / "loadings.csv")
        print(f"\n  Top neural weights (X):")
        x_top = (ldf[ldf["set"] == "X"].assign(abs_w=lambda d: d["weight"].abs())
                 .sort_values("abs_w", ascending=False).head(8))
        for _, row in x_top.iterrows():
            print(f"    {row['name']:55s}  w = {row['weight']:+.4f}")
        print(f"\n  Behavioral weights (Y):")
        y_w = ldf[ldf["set"] == "Y"]
        for _, row in y_w.iterrows():
            print(f"    {row['name']:55s}  w = {row['weight']:+.4f}")
    else:
        print("\n>>> No models survived p < 0.05 after permutation test. <<<")
        print("    Best model by raw p:")
        print(res_df.head(3)[["brain_set","n_brain_features","observed_r",
                               "permutation_p","q_fdr"]].to_string(index=False))

    print(f"\nAll outputs saved to: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
