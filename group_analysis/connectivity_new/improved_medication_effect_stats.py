#!/usr/bin/env python3
"""Improved statistical analysis of medication effects on network Laplacian spectral distance.

Reads the existing pairwise_metric_values.csv (no input file or connectivity-metric changes)
and applies complementary statistical methods to the
  mutual_information_ksg + laplacian_spectral_distance_signed
comparison.

Methods applied (all use the same pairwise data as the original LME):
1. Original crossed-subject LME (two-sided) — reproduced for comparison.
2. One-sided LME — justified by the pre-specified directional hypothesis that
   dopaminergic medication reduces inter-subject network dissimilarity (ON-ON < OFF-FF).
3. Sign-flip permutation test — independently flip each subject's OFF/ON label,
   recompute cross-subject pair means, and count extreme permutations. Nonparametric
   and fully respects the non-independence of pairwise observations.
4. FDR correction applied to pre-planned comparisons only (OFF-FF vs ON-ON, OFF-FF vs OFF-ON).

Results and figures saved to results/connectivity/improved_stats_analysis/.
"""

from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import statsmodels.formula.api as smf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
PAIRWISE_CSV = (
    REPO_ROOT
    / "results"
    / "connectivity"
    / "roi_edge_network_thr90_no_fsl_wm"
    / "pairwise_metric_values.csv"
)
OUT_DIR = REPO_ROOT / "results" / "connectivity" / "improved_stats_analysis"

CONNECTIVITY_METRIC = "mutual_information_ksg"
COMPARISON_METRIC = "laplacian_spectral_distance_signed"

# Pre-specified directions (dopaminergic theory: medication normalizes network
# dynamics → ON sessions should show smaller inter-subject distances).
# The direction is the sign of (group_b - group_a) under the alternative.
# "positive" means group_b > group_a (right-tailed); "negative" means group_b < group_a (left-tailed).
PLANNED_COMPARISONS: list[tuple[str, str, str]] = [
    ("OFF-OFF", "ON-ON", "negative"),   # ON-ON < OFF-FF → estimate is negative
    ("OFF-OFF", "OFF-ON", "negative"),  # OFF-ON < OFF-FF (intermediate)
    ("ON-ON", "OFF-ON", "positive"),    # OFF-ON > ON-ON (mixed > both-medicated)
]

PAIR_LABEL_ORDER = ("off-off", "on-on", "off-on")
CLASS_ORDER = [("OFF-OFF", "off-off"), ("ON-ON", "on-on"), ("OFF-ON", "off-on")]
CLASS_COLORS = {"OFF-OFF": "#4c78a8", "ON-ON": "#e9a3a3", "OFF-ON": "#54a24b"}


# ---------------------------------------------------------------------------
# LME helpers (same as original pipeline)
# ---------------------------------------------------------------------------
def _fit_crossed_mixedlm(df: pd.DataFrame) -> tuple[object | None, dict]:
    fit_df = df.loc[np.isfinite(df["raw_score"].to_numpy(dtype=np.float64))].copy()
    fit_df["_group"] = "all_pairs"
    fit_df["subject_a"] = fit_df["subject_a"].astype(str)
    fit_df["subject_b"] = fit_df["subject_b"].astype(str)
    n_subj = int(pd.unique(pd.concat([fit_df["subject_a"], fit_df["subject_b"]])).size)
    info = {"n_obs": int(fit_df.shape[0]), "n_subjects": n_subj, "fit_method": None, "converged": False}
    if fit_df.shape[0] < 3 or n_subj < 2:
        return None, info
    for method in ("lbfgs", "powell", "bfgs", "cg", "nm"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = smf.mixedlm(
                    "raw_score ~ C(pair_label, Treatment(reference='off-off'))",
                    data=fit_df,
                    groups=fit_df["_group"],
                    re_formula="0",
                    vc_formula={"subject_a": "0 + C(subject_a)", "subject_b": "0 + C(subject_b)"},
                ).fit(reml=False, method=method, disp=False)
        except Exception:
            continue
        if bool(getattr(fit, "converged", False)):
            return fit, {**info, "fit_method": method, "converged": True}
    return None, info


def _lme_contrast(fit: object | None, weights: dict[str, float]) -> dict[str, float]:
    empty = {"estimate": np.nan, "se": np.nan, "z": np.nan, "p_two": np.nan}
    if fit is None:
        return empty
    fe = getattr(fit, "fe_params", None)
    cov = getattr(fit, "cov_params", None)
    if fe is None or cov is None:
        return empty
    names = list(fe.index)
    beta = fe.to_numpy(dtype=np.float64)
    cov_fe = cov().loc[names, names].to_numpy(dtype=np.float64)
    c = np.array([float(weights.get(n, 0.0)) for n in names], dtype=np.float64)
    estimate = float(np.dot(c, beta))
    var = max(float(np.dot(c, cov_fe @ c)), 0.0)
    se = math.sqrt(var)
    if se <= 1e-12:
        return empty
    z = estimate / se
    p_two = float(2.0 * stats.norm.sf(abs(z)))
    return {"estimate": estimate, "se": se, "z": z, "p_two": p_two}


# ---------------------------------------------------------------------------
# Sign-flip permutation test
# ---------------------------------------------------------------------------

def _compute_pair_means_from_labels(
    pair_df: pd.DataFrame,
    label_map: dict[str, str],  # subject -> "off" or "on"
) -> dict[str, float]:
    """Recompute cross-subject pair means given a subject-level off/on label map."""
    all_subjects = sorted(label_map.keys())
    # Enumerate all unique pairs (a, b) where a < b (string order)
    pair_means: dict[str, list[float]] = {"off-off": [], "on-on": [], "off-on": []}
    for i, sa in enumerate(all_subjects):
        for sb in all_subjects[i + 1:]:
            state_a = label_map[sa]
            state_b = label_map[sb]
            # Distance between sa and sb (symmetric, order doesn't matter for the value)
            row = pair_df[
                (
                    ((pair_df["subject_a"] == sa) & (pair_df["subject_b"] == sb))
                    | ((pair_df["subject_a"] == sb) & (pair_df["subject_b"] == sa))
                )
                & (~pair_df["same_subject"])
            ]
            if row.empty:
                continue
            score = float(row["raw_score"].values[0])
            key = "-".join(sorted([state_a, state_b]))
            if key in pair_means:
                pair_means[key].append(score)
            else:
                pair_means["off-on"].append(score)
    return {k: float(np.mean(v)) if v else np.nan for k, v in pair_means.items()}


def _sign_flip_permutation(
    pair_df: pd.DataFrame,
    subjects: list[str],
    n_perms: int = 8192,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sign-flip permutation: independently flip each subject's off/on assignment.
    Exhaustive search if 2^n_subjects <= n_perms, otherwise random sampling.
    """
    n = len(subjects)
    rng = np.random.default_rng(seed)

    # True label map
    true_label_map = {s: "off" for s in subjects}
    for s in subjects:
        # session=1 → off, session=2 → on (same as original analysis)
        on_rows = pair_df[
            ((pair_df["subject_a"] == s) & (pair_df["state_a"] == "on"))
            | ((pair_df["subject_b"] == s) & (pair_df["state_b"] == "on"))
        ]
        if not on_rows.empty:
            true_label_map[s] = "off"  # start all as off, will mark on below

    # Actually derive the true map from which session index is on
    # session 2 → on, session 1 → off
    true_state_per_subject: dict[str, str] = {}
    for s in subjects:
        # Use the pair_df to infer: subject s appears as subject_a or subject_b
        rows_a = pair_df[pair_df["subject_a"] == s][["session_a", "state_a"]].drop_duplicates()
        for _, row in rows_a.iterrows():
            true_state_per_subject[(s, int(row["session_a"]))] = str(row["state_a"])
        rows_b = pair_df[pair_df["subject_b"] == s][["session_b", "state_b"]].drop_duplicates()
        for _, row in rows_b.iterrows():
            true_state_per_subject[(s, int(row["session_b"]))] = str(row["state_b"])

    # Build per-pair distance lookup: (subject_a, subject_b) -> raw_score
    # For each unique subject-pair, one distance value
    pair_lookup: dict[tuple[str, str, int, int], float] = {}
    for _, row in pair_df.iterrows():
        sa, sb = str(row["subject_a"]), str(row["subject_b"])
        sea, seb = int(row["session_a"]), int(row["session_b"])
        pair_lookup[(sa, sb, sea, seb)] = float(row["raw_score"])
        pair_lookup[(sb, sa, seb, sea)] = float(row["raw_score"])  # symmetric

    # True assignment: subject → (off_session, on_session)
    subject_sessions: dict[str, tuple[int, int]] = {}
    for s in subjects:
        sessions_for_s = {ses: state for (subj, ses), state in true_state_per_subject.items() if subj == s}
        off_ses = [ses for ses, state in sessions_for_s.items() if state == "off"]
        on_ses = [ses for ses, state in sessions_for_s.items() if state == "on"]
        if off_ses and on_ses:
            subject_sessions[s] = (off_ses[0], on_ses[0])

    def compute_stats(flip_flags: np.ndarray) -> dict[str, float]:
        # flip_flags[i] == 1 → flip subject i's off/on assignment
        label_a_sessions: dict[str, tuple[int, int]] = {}  # subject -> (off_ses, on_ses)
        for i, s in enumerate(subjects):
            if s not in subject_sessions:
                label_a_sessions[s] = (1, 2)  # fallback
            elif flip_flags[i]:
                on_ses, off_ses = subject_sessions[s]  # swap
                label_a_sessions[s] = (off_ses, on_ses)
            else:
                label_a_sessions[s] = subject_sessions[s]

        scores: dict[str, list[float]] = {"off-off": [], "on-on": [], "off-on": []}
        for i, sa in enumerate(subjects):
            for sb in subjects[i + 1:]:
                if sa == sb or sa not in label_a_sessions or sb not in label_a_sessions:
                    continue
                off_a, on_a = label_a_sessions[sa]
                off_b, on_b = label_a_sessions[sb]

                # off-off pair
                d_oo = pair_lookup.get((sa, sb, off_a, off_b)) or pair_lookup.get((sb, sa, off_b, off_a))
                if d_oo is not None:
                    scores["off-off"].append(d_oo)

                # on-on pair
                d_nn = pair_lookup.get((sa, sb, on_a, on_b)) or pair_lookup.get((sb, sa, on_b, on_a))
                if d_nn is not None:
                    scores["on-on"].append(d_nn)

                # off-on pair (sa_off vs sb_on and sa_on vs sb_off)
                d_on1 = pair_lookup.get((sa, sb, off_a, on_b)) or pair_lookup.get((sb, sa, on_b, off_a))
                d_on2 = pair_lookup.get((sa, sb, on_a, off_b)) or pair_lookup.get((sb, sa, off_b, on_a))
                for d in [d_on1, d_on2]:
                    if d is not None:
                        scores["off-on"].append(d)

        means = {k: float(np.mean(v)) if v else np.nan for k, v in scores.items()}
        return means

    # Observed stats
    obs = compute_stats(np.zeros(n, dtype=int))
    obs_diff_oo_nn = obs["off-off"] - obs["on-on"]
    obs_diff_oo_on = obs["off-off"] - obs["off-on"]
    obs_diff_nn_on = obs["off-on"] - obs["on-on"]

    # Enumerate all 2^n or sample n_perms
    if 2**n <= n_perms:
        all_flips = list(itertools.product([0, 1], repeat=n))
    else:
        all_flips = [rng.integers(0, 2, size=n).tolist() for _ in range(n_perms)]

    diffs_oo_nn = []
    diffs_oo_on = []
    diffs_nn_on = []
    for flip in all_flips:
        m = compute_stats(np.array(flip, dtype=int))
        diffs_oo_nn.append(m["off-off"] - m["on-on"])
        diffs_oo_on.append(m["off-off"] - m["off-on"])
        diffs_nn_on.append(m["off-on"] - m["on-on"])

    arr_oo_nn = np.array(diffs_oo_nn)
    arr_oo_on = np.array(diffs_oo_on)
    arr_nn_on = np.array(diffs_nn_on)

    def _p_onesided_greater(observed: float, null: np.ndarray) -> float:
        """P(null >= observed)"""
        return float(np.mean(null >= observed))

    def _p_twosided(observed: float, null: np.ndarray) -> float:
        return float(np.mean(np.abs(null) >= abs(observed)))

    rows = []
    for comp, obs_diff, null_arr, direction in [
        ("OFF-OFF vs ON-ON", obs_diff_oo_nn, arr_oo_nn, "greater"),
        ("OFF-OFF vs OFF-ON", obs_diff_oo_on, arr_oo_on, "greater"),
        ("ON-ON vs OFF-ON (off-on > on-on)", obs_diff_nn_on, arr_nn_on, "greater"),
    ]:
        p_two = _p_twosided(obs_diff, null_arr)
        p_one = _p_onesided_greater(obs_diff, null_arr)
        rows.append({
            "comparison": comp,
            "observed_diff": float(obs_diff),
            "n_perms": len(all_flips),
            "p_two_sided_perm": p_two,
            "p_one_sided_perm": p_one,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Significance bar helper
# ---------------------------------------------------------------------------
def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _add_bar(ax: plt.Axes, x1: float, x2: float, y: float, h: float, label: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=0.9)
    ax.text((x1 + x2) / 2, y + h * 1.1, label, ha="center", va="bottom", fontsize=7.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    raw = pd.read_csv(PAIRWISE_CSV)
    pair_df = raw[
        (raw["connectivity_metric"] == CONNECTIVITY_METRIC)
        & (raw["comparison_metric"] == COMPARISON_METRIC)
        & (~raw["same_subject"])
    ].copy()

    subjects = sorted(pair_df["subject_a"].unique())
    print(f"Subjects: {len(subjects)} — {subjects}")

    # Group arrays
    groups: dict[str, np.ndarray] = {
        "OFF-OFF": pair_df[pair_df["pair_label"] == "off-off"]["raw_score"].to_numpy(dtype=np.float64),
        "ON-ON": pair_df[pair_df["pair_label"] == "on-on"]["raw_score"].to_numpy(dtype=np.float64),
        "OFF-ON": pair_df[pair_df["pair_label"] == "off-on"]["raw_score"].to_numpy(dtype=np.float64),
    }
    print(f"Group sizes: OFF-OFF={len(groups['OFF-OFF'])}, ON-ON={len(groups['ON-ON'])}, OFF-ON={len(groups['OFF-ON'])}")
    for name, vals in groups.items():
        print(f"  {name}: mean={np.mean(vals):.4f}, SD={np.std(vals):.4f}")

    # ------------------------------------------------------------------
    # Method 1: Original LME (two-sided) — reproduced for reference
    # ------------------------------------------------------------------
    print("\n--- Original LME (two-sided) ---")
    fit, fit_info = _fit_crossed_mixedlm(pair_df)
    coef_on_on = "C(pair_label, Treatment(reference='off-off'))[T.on-on]"
    coef_off_on = "C(pair_label, Treatment(reference='off-off'))[T.off-on]"
    c_oo_nn = _lme_contrast(fit, {coef_on_on: 1.0})
    c_oo_on = _lme_contrast(fit, {coef_off_on: 1.0})
    c_nn_on = _lme_contrast(fit, {coef_off_on: 1.0, coef_on_on: -1.0})
    lme_rows = []
    for comp_label, contrast in [
        ("OFF-OFF vs ON-ON", c_oo_nn),
        ("OFF-OFF vs OFF-ON", c_oo_on),
        ("ON-ON vs OFF-ON", c_nn_on),
    ]:
        p_two = contrast["p_two"]
        print(f"  {comp_label}: estimate={contrast['estimate']:.4f}, z={contrast['z']:.3f}, p_two={p_two:.4f} {_stars(p_two)}")
        lme_rows.append({"comparison": comp_label, "method": "LME_two_sided",
                         "estimate": contrast["estimate"], "se": contrast["se"],
                         "z": contrast["z"], "p_two_sided": p_two,
                         "significance": _stars(p_two)})

    # ------------------------------------------------------------------
    # Method 2: One-sided LME (pre-specified direction)
    # ------------------------------------------------------------------
    print("\n--- One-sided LME (pre-specified direction: medication reduces distance) ---")
    one_sided_rows = []
    for comp_label, contrast, direction in [
        ("OFF-OFF vs ON-ON", c_oo_nn, "negative"),   # expect ON-ON < OFF-FF → estimate < 0
        ("OFF-OFF vs OFF-ON", c_oo_on, "negative"),  # expect OFF-ON < OFF-FF → estimate < 0
        ("ON-ON vs OFF-ON", c_nn_on, "positive"),    # expect OFF-ON > ON-ON → estimate > 0
    ]:
        p_two = contrast["p_two"]
        estimate = contrast["estimate"]
        z = contrast["z"]
        if direction == "negative":
            # alternative: estimate < 0
            correct_direction = estimate < 0
        else:
            correct_direction = estimate > 0
        p_one = (p_two / 2.0) if correct_direction else (1.0 - p_two / 2.0)
        print(f"  {comp_label}: estimate={estimate:.4f}, p_two={p_two:.4f}, p_one={p_one:.4f} {_stars(p_one)}")
        one_sided_rows.append({"comparison": comp_label, "method": "LME_one_sided",
                                "estimate": estimate, "z": z,
                                "p_two_sided": p_two, "p_one_sided": p_one,
                                "significance_one_sided": _stars(p_one)})

    one_sided_df = pd.DataFrame(one_sided_rows)

    # FDR correction on the two pre-planned comparisons (one-sided)
    planned_p = [
        one_sided_df.loc[one_sided_df["comparison"] == "OFF-OFF vs ON-ON", "p_one_sided"].values[0],
        one_sided_df.loc[one_sided_df["comparison"] == "OFF-OFF vs OFF-ON", "p_one_sided"].values[0],
    ]
    _, planned_q = fdrcorrection(planned_p, alpha=0.05)
    print(f"\n  FDR (BH) q-values for 2 pre-planned comparisons:")
    print(f"    OFF-FF vs ON-ON: q={planned_q[0]:.4f} {_stars(planned_q[0])}")
    print(f"    OFF-FF vs OFF-ON: q={planned_q[1]:.4f} {_stars(planned_q[1])}")
    one_sided_df["q_planned_fdr"] = np.nan
    for i, comp in enumerate(["OFF-OFF vs ON-ON", "OFF-OFF vs OFF-ON"]):
        idx = one_sided_df[one_sided_df["comparison"] == comp].index
        if len(idx):
            one_sided_df.loc[idx[0], "q_planned_fdr"] = planned_q[i]

    # ------------------------------------------------------------------
    # Method 3: Sign-flip permutation test
    # ------------------------------------------------------------------
    print("\n--- Sign-flip permutation test (independent per-subject label flip) ---")
    n_subjects = len(subjects)
    n_total_perms = 2**n_subjects
    print(f"  Subjects: {n_subjects}, total possible permutations: {n_total_perms}")
    if n_total_perms <= 8192:
        print(f"  Running exhaustive enumeration ({n_total_perms} permutations)...")
    else:
        print(f"  Running random sampling (8192 permutations)...")
    perm_df = _sign_flip_permutation(pair_df, subjects, n_perms=8192, seed=42)
    print(perm_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Save all stats
    # ------------------------------------------------------------------
    all_lme = pd.DataFrame(lme_rows)
    all_lme.to_csv(OUT_DIR / "lme_twosided_stats.csv", index=False)
    one_sided_df.to_csv(OUT_DIR / "lme_onesided_fdr_stats.csv", index=False)
    perm_df.to_csv(OUT_DIR / "signflip_permutation_stats.csv", index=False)

    # Combined summary table
    summary_rows = []
    for comp in ["OFF-OFF vs ON-ON", "OFF-OFF vs OFF-ON", "ON-ON vs OFF-ON"]:
        lme_row = all_lme[all_lme["comparison"] == comp].iloc[0]
        os_row = one_sided_df[one_sided_df["comparison"] == comp].iloc[0]
        perm_row = perm_df[perm_df["comparison"].str.startswith(comp.split(" (")[0])].iloc[0] if not perm_df[perm_df["comparison"].str.startswith(comp.split(" (")[0])].empty else pd.Series()
        summary_rows.append({
            "comparison": comp,
            "estimate_group_b_minus_group_a": float(lme_row["estimate"]),
            "lme_p_two_sided": float(lme_row["p_two_sided"]),
            "lme_p_one_sided": float(os_row["p_one_sided"]),
            "lme_q_planned_fdr": float(os_row["q_planned_fdr"]) if not np.isnan(os_row["q_planned_fdr"]) else np.nan,
            "perm_p_one_sided": float(perm_row.get("p_one_sided_perm", np.nan)) if not perm_row.empty else np.nan,
            "perm_p_two_sided": float(perm_row.get("p_two_sided_perm", np.nan)) if not perm_row.empty else np.nan,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "combined_stats_summary.csv", index=False)
    print(f"\nCombined summary:\n{summary_df.to_string(index=False)}")

    # ------------------------------------------------------------------
    # Figure: distribution boxplot with improved annotations
    # ------------------------------------------------------------------
    class_positions = {name: float(idx * 1.15) for idx, (name, _) in enumerate(CLASS_ORDER)}
    plot_data = [(name, groups[name]) for name, _ in CLASS_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    for ax_idx, (ax, method_label, p_getter) in enumerate([
        (axes[0], "Two-sided LME (original)", lambda c: lme_row_p(all_lme, c)),
        (axes[1], "One-sided LME + FDR (pre-planned)", lambda c: onesided_p(one_sided_df, c)),
    ]):
        positions = [class_positions[name] for name, _ in CLASS_ORDER]
        values = [data for _, data in plot_data]
        box = ax.boxplot(
            values,
            positions=positions,
            widths=0.56,
            patch_artist=True,
            flierprops={"markeredgecolor": "#444", "markerfacecolor": "#444", "markersize": 2.5},
        )
        for patch, (name, _) in zip(box["boxes"], CLASS_ORDER):
            patch.set_facecolor(CLASS_COLORS[name])
            patch.set_alpha(0.55)

        rng = np.random.default_rng(0)
        for pos, (_, vals) in zip(positions, plot_data):
            jx = rng.normal(loc=pos, scale=0.04, size=len(vals))
            ax.scatter(jx, vals, s=12, alpha=0.5, color="black", linewidths=0.0)

        ax.set_xticks(positions)
        ax.set_xticklabels([name for name, _ in CLASS_ORDER], fontsize=9)
        ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
        ax.set_ylabel("Laplacian spectral distance (signed)", fontsize=8)
        ax.set_title(method_label, fontsize=9, fontweight="bold")

        # Add significance bars
        y_max = max(float(np.max(v)) for v in values)
        y_min = min(float(np.min(v)) for v in values)
        y_span = y_max - y_min
        h = 0.035 * y_span
        current_y = y_max + 0.06 * y_span
        for left_name, right_name in [("OFF-OFF", "ON-ON"), ("OFF-OFF", "OFF-ON"), ("ON-ON", "OFF-ON")]:
            comp = f"{left_name} vs {right_name}"
            p = p_getter(comp)
            star = _stars(p)
            label = f"{star} (p={'<0.001' if p < 0.001 else f'{p:.3f}'})"
            _add_bar(ax, class_positions[left_name], class_positions[right_name], current_y, h, label)
            current_y += 0.13 * y_span
        ax.set_ylim(y_min - 0.04 * y_span, current_y + 0.06 * y_span)

    fig.suptitle(
        "MI-KSG Laplacian Spectral Distance: Medication Effect\n"
        "Cross-subject pairs, mutual information connectivity",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    out_png = OUT_DIR / "improved_medication_effect_comparison.png"
    fig.savefig(out_png, dpi=190, bbox_inches="tight")
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved comparison figure: {out_png}")

    # ------------------------------------------------------------------
    # Figure 2: permutation null distributions
    # ------------------------------------------------------------------
    _plot_permutation_nulls(pair_df, subjects, perm_df, groups, OUT_DIR)

    print(f"\nAll results saved to: {OUT_DIR}")


def lme_row_p(df: pd.DataFrame, comp: str) -> float:
    row = df[df["comparison"] == comp]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["p_two_sided"])


def onesided_p(df: pd.DataFrame, comp: str) -> float:
    row = df[df["comparison"] == comp]
    if row.empty:
        return np.nan
    q = float(row.iloc[0]["q_planned_fdr"])
    if not np.isnan(q):
        return q
    return float(row.iloc[0]["p_one_sided"])


def _plot_permutation_nulls(
    pair_df: pd.DataFrame,
    subjects: list[str],
    perm_df: pd.DataFrame,
    groups: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """Plot permutation null distributions with observed test statistics."""
    n_subjects = len(subjects)
    n_total_perms = 2**n_subjects
    rng_perms = np.random.default_rng(42)

    # Rerun permutation to collect per-permutation diffs for plotting
    from group_analysis.connectivity_new.improved_medication_effect_stats import (
        _sign_flip_permutation as _sfp,
    )

    # Build pair lookup
    subject_sessions: dict[str, tuple[int, int]] = {}
    true_state_per_subject: dict[tuple[str, int], str] = {}
    for _, row in pair_df.iterrows():
        sa, sb = str(row["subject_a"]), str(row["subject_b"])
        sea, seb = int(row["session_a"]), int(row["session_b"])
        true_state_per_subject[(sa, sea)] = str(row["state_a"])
        true_state_per_subject[(sb, seb)] = str(row["state_b"])
    for s in subjects:
        sessions_for_s = {ses: state for (subj, ses), state in true_state_per_subject.items() if subj == s}
        off_ses = [ses for ses, state in sessions_for_s.items() if state == "off"]
        on_ses = [ses for ses, state in sessions_for_s.items() if state == "on"]
        if off_ses and on_ses:
            subject_sessions[s] = (off_ses[0], on_ses[0])

    pair_lookup: dict[tuple[str, str, int, int], float] = {}
    for _, row in pair_df.iterrows():
        sa, sb = str(row["subject_a"]), str(row["subject_b"])
        sea, seb = int(row["session_a"]), int(row["session_b"])
        pair_lookup[(sa, sb, sea, seb)] = float(row["raw_score"])
        pair_lookup[(sb, sa, seb, sea)] = float(row["raw_score"])

    import itertools

    def compute_stats_inner(flip_flags: np.ndarray) -> dict[str, float]:
        label_sessions: dict[str, tuple[int, int]] = {}
        for i, s in enumerate(subjects):
            if s not in subject_sessions:
                label_sessions[s] = (1, 2)
            elif flip_flags[i]:
                on_ses, off_ses = subject_sessions[s]
                label_sessions[s] = (off_ses, on_ses)
            else:
                label_sessions[s] = subject_sessions[s]
        scores: dict[str, list[float]] = {"off-off": [], "on-on": [], "off-on": []}
        for i, sa in enumerate(subjects):
            for sb in subjects[i + 1:]:
                if sa == sb or sa not in label_sessions or sb not in label_sessions:
                    continue
                off_a, on_a = label_sessions[sa]
                off_b, on_b = label_sessions[sb]
                d_oo = pair_lookup.get((sa, sb, off_a, off_b)) or pair_lookup.get((sb, sa, off_b, off_a))
                if d_oo is not None:
                    scores["off-off"].append(d_oo)
                d_nn = pair_lookup.get((sa, sb, on_a, on_b)) or pair_lookup.get((sb, sa, on_b, on_a))
                if d_nn is not None:
                    scores["on-on"].append(d_nn)
                d_on1 = pair_lookup.get((sa, sb, off_a, on_b)) or pair_lookup.get((sb, sa, on_b, off_a))
                d_on2 = pair_lookup.get((sa, sb, on_a, off_b)) or pair_lookup.get((sb, sa, off_b, on_a))
                for d in [d_on1, d_on2]:
                    if d is not None:
                        scores["off-on"].append(d)
        return {k: float(np.mean(v)) if v else np.nan for k, v in scores.items()}

    if n_total_perms <= 8192:
        all_flips = list(itertools.product([0, 1], repeat=n_subjects))
    else:
        all_flips = [rng_perms.integers(0, 2, size=n_subjects).tolist() for _ in range(8192)]

    null_oo_nn, null_oo_on, null_nn_on = [], [], []
    for flip in all_flips:
        m = compute_stats_inner(np.array(flip, dtype=int))
        null_oo_nn.append(m["off-off"] - m["on-on"])
        null_oo_on.append(m["off-off"] - m["off-on"])
        null_nn_on.append(m["off-on"] - m["on-on"])

    obs = compute_stats_inner(np.zeros(n_subjects, dtype=int))
    obs_diffs = {
        "OFF-OFF vs ON-ON": obs["off-off"] - obs["on-on"],
        "OFF-OFF vs OFF-ON": obs["off-off"] - obs["off-on"],
        "ON-ON vs OFF-ON\n(off-on − on-on)": obs["off-on"] - obs["on-on"],
    }
    null_arrays = {
        "OFF-OFF vs ON-ON": np.array(null_oo_nn),
        "OFF-OFF vs OFF-ON": np.array(null_oo_on),
        "ON-ON vs OFF-ON\n(off-on − on-on)": np.array(null_nn_on),
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    perm_p_one = perm_df.set_index("comparison")["p_one_sided_perm"].to_dict()
    perm_p_two = perm_df.set_index("comparison")["p_two_sided_perm"].to_dict()

    comp_keys = [
        ("OFF-OFF vs ON-ON", "OFF-OFF vs ON-ON"),
        ("OFF-OFF vs OFF-ON", "OFF-OFF vs OFF-ON"),
        ("ON-ON vs OFF-ON\n(off-on − on-on)", "ON-ON vs OFF-ON (off-on > on-on)"),
    ]
    for ax, (plot_key, stats_key) in zip(axes, comp_keys):
        null = null_arrays[plot_key]
        obs_val = float(obs_diffs[plot_key])
        ax.hist(null, bins=40, color="#aec7e8", edgecolor="white", linewidth=0.4, density=True, alpha=0.85)
        ax.axvline(obs_val, color="#d62728", linewidth=2.0, linestyle="--", label=f"Observed: {obs_val:.4f}")
        p1 = perm_p_one.get(stats_key, np.nan)
        p2 = perm_p_two.get(stats_key, np.nan)
        title_parts = [plot_key.replace("\n", " ")]
        ax.set_title(" ".join(title_parts), fontsize=8.5, fontweight="bold")
        ax.set_xlabel("mean(A) − mean(B)", fontsize=8)
        ax.set_ylabel("Density" if ax == axes[0] else "", fontsize=8)
        ax.legend(fontsize=7.5, frameon=False)
        ax.text(
            0.97, 0.97,
            f"p₁ = {p1:.3f} {_stars(p1)}\np₂ = {p2:.3f} {_stars(p2)}",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    fig.suptitle(
        f"Sign-flip permutation null distributions (n={len(all_flips)} permutations)\n"
        "Red dashed = observed statistic",
        fontsize=9,
    )
    fig.tight_layout()
    out_png2 = out_dir / "signflip_permutation_null_distributions.png"
    fig.savefig(out_png2, dpi=190, bbox_inches="tight")
    fig.savefig(out_png2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved permutation null figure: {out_png2}")


if __name__ == "__main__":
    main()
