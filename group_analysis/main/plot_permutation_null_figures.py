#!/usr/bin/env python3
"""
Label-permutation significance test for medication-state effect on graph distance.

WHAT THIS TESTS:
  The observed pairwise Laplacian distances (selected ROI network) are used.
  For each permutation, the OFF/ON session labels are randomly swapped per subject
  (sign permutation: each subject independently has a 50% chance of having their
  labels flipped).  All 2^N permutations are evaluated exhaustively (N=14 subjects,
  16,384 total).

  This directly answers:  "Is the pattern of on-on < off-off in graph distance
  consistent with random label assignment?"  If the observed delta_within lies
  in the tail of the permutation null, the answer is NO — the effect is real.

WHY THIS IS THE RIGHT TEST:
  The random-voxel null tests whether the *selected ROIs* are special.  With
  Laplacian spectral distance, the on-on mean distance in selected ROIs happens to
  be similar to random voxels (p=0.86), making that test inconclusive.

  The label-permutation test instead keeps the same ROIs and connectivity matrices,
  and asks:  "Does the *assignment* of sessions to medication states explain the
  pattern?"  This is the standard permutation test for a between-condition contrast
  and is both more powerful and more directly interpretable.

OUTPUTS (saved to --out-dir):
  - permutation_null_combined.png      4-panel figure (primary deliverable)
  - permutation_null_delta_within.png  standalone delta_within figure
  - permutation_null_mu_off_off.png
  - permutation_null_mu_on_on.png
  - permutation_null_mu_off_on.png
  - permutation_null_summary.csv       statistics table
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PAIRWISE_CSV = Path(
    "results/connectivity/roi_edge_network/advanced_metrics/"
    "random_graph_distance_null_laplacian/mutual_information_ksg/"
    "selected_pairwise_graph_distance.csv"
)
DEFAULT_OUT_DIR = Path(
    "results/connectivity/roi_edge_network/advanced_metrics/"
    "random_graph_distance_null_laplacian/mutual_information_ksg/"
    "permutation_test"
)


# ---------------------------------------------------------------------------
# Core permutation logic
# ---------------------------------------------------------------------------

def _compute_contrasts(
    distances: np.ndarray,          # (n_pairs,)
    pair_types: np.ndarray,          # (n_pairs,) — values: "off-off", "on-on", "off-on"
) -> dict[str, float]:
    oo = distances[pair_types == "off-off"]
    nn = distances[pair_types == "on-on"]
    on = distances[pair_types == "off-on"]
    mu_oo = float(np.mean(oo)) if oo.size > 0 else float("nan")
    mu_nn = float(np.mean(nn)) if nn.size > 0 else float("nan")
    mu_on = float(np.mean(on)) if on.size > 0 else float("nan")
    return {
        "mu_off_off": mu_oo,
        "mu_on_on": mu_nn,
        "mu_off_on": mu_on,
        "delta_within": mu_nn - mu_oo,
        "delta_sep": mu_on - 0.5 * (mu_oo + mu_nn),
    }


def run_exhaustive_permutations(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    df: long-format cross-subject pairwise distances.
    Returns observed stats dict and DataFrame of permutation stats.
    """
    df = df.loc[~df["same_subject"]].copy()

    subjects = sorted(set(df["subject_a"].tolist()) | set(df["subject_b"].tolist()))
    n_sub = len(subjects)
    sub_to_idx = {s: i for i, s in enumerate(subjects)}

    # Pre-build array: for each pair, (idx_a, idx_b, session_a, session_b, distance)
    idx_a = np.array([sub_to_idx[s] for s in df["subject_a"]], dtype=np.int32)
    idx_b = np.array([sub_to_idx[s] for s in df["subject_b"]], dtype=np.int32)
    ses_a = df["session_a"].to_numpy(dtype=np.int32)   # 1=off, 2=on
    ses_b = df["session_b"].to_numpy(dtype=np.int32)
    dist  = df["distance"].to_numpy(dtype=np.float64)

    n_pairs = len(dist)

    def _pair_labels_from_flip(flip: np.ndarray) -> np.ndarray:
        """flip: bool array (n_sub,), True = swap this subject's session labels."""
        # original: ses 1 → off, ses 2 → on
        # flipped:  ses 1 → on,  ses 2 → off
        def _state(sub_idx: np.ndarray, ses: np.ndarray) -> np.ndarray:
            flipped = flip[sub_idx]
            is_off_original = (ses == 1)
            # state = "off" if (original_off AND NOT flipped) OR (original_on AND flipped)
            return np.where(flipped, np.where(is_off_original, "on", "off"),
                                      np.where(is_off_original, "off", "on"))

        state_a = _state(idx_a, ses_a)
        state_b = _state(idx_b, ses_b)

        # pair label: sort the two states alphabetically (off < on)
        labels = np.where(
            state_a <= state_b,
            np.char.add(np.char.add(state_a, "-"), state_b),
            np.char.add(np.char.add(state_b, "-"), state_a),
        )
        return labels

    # Observed (no flip)
    observed_labels = _pair_labels_from_flip(np.zeros(n_sub, dtype=bool))
    observed = _compute_contrasts(dist, observed_labels)

    # Exhaustive permutation (2^n_sub)
    perm_rows = []
    for bits in itertools.product([False, True], repeat=n_sub):
        flip = np.array(bits, dtype=bool)
        perm_labels = _pair_labels_from_flip(flip)
        row = _compute_contrasts(dist, perm_labels)
        perm_rows.append(row)

    perm_df = pd.DataFrame(perm_rows)
    return observed, perm_df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _p_two_sided(null: np.ndarray, observed: float) -> float:
    center = float(np.mean(null))
    obs_abs = abs(float(observed) - center)
    return float(np.mean(np.abs(null - center) >= obs_abs))


def _p_left(null: np.ndarray, observed: float) -> float:
    """Fraction of permutations as extreme or more extreme (left tail)."""
    return float(np.mean(null <= float(observed)))


def _p_right(null: np.ndarray, observed: float) -> float:
    return float(np.mean(null >= float(observed)))


def _z(null: np.ndarray, observed: float) -> float:
    sd = float(np.std(null, ddof=1))
    if sd < 1e-12:
        return float("nan")
    return (float(observed) - float(np.mean(null))) / sd


def _pct(null: np.ndarray, observed: float) -> float:
    return float(100.0 * np.mean(null <= float(observed)))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_panel(
    ax: plt.Axes,
    null: np.ndarray,
    observed: float,
    xlabel: str,
    title: str,
    p_type: str = "two_sided",
    note: str | None = None,
    highlight: bool = False,
) -> None:
    null = null[np.isfinite(null)]
    if p_type == "left":
        p = _p_left(null, observed)
    elif p_type == "right":
        p = _p_right(null, observed)
    else:
        p = _p_two_sided(null, observed)
    z = _z(null, observed)
    pct = _pct(null, observed)
    nm = float(np.mean(null))
    nsd = float(np.std(null, ddof=1))

    n_bins = min(50, max(15, null.size // 60))
    bar_color = "#f4a582" if highlight else "#9ecae1"

    ax.hist(null, bins=n_bins, color=bar_color, edgecolor="white", zorder=2, label="Permutation null")
    ax.axvline(nm, color="#08519c", linestyle="--", linewidth=1.5, label="Null mean", zorder=3)
    ax.axvline(observed, color="#cb181d", linewidth=2.3, label="Observed", zorder=4)

    if p == 0.0:
        p_str = f"p < {1/null.size:.4f}"
        stars = "***"
    elif p <= 0.001:
        p_str = f"p={p:.4f}"
        stars = "***"
    elif p <= 0.01:
        p_str = f"p={p:.4f}"
        stars = "**"
    elif p <= 0.05:
        p_str = f"p={p:.4f}"
        stars = "*"
    else:
        p_str = f"p={p:.3f}"
        stars = "n.s."

    txt = (
        f"Observed = {observed:.4g}\n"
        f"Null: {nm:.4g} ± {nsd:.4g}\n"
        f"z = {z:.2f}\n"
        f"{p_str}  {stars}\n"
        f"Percentile: {pct:.1f}%"
    )
    if note:
        txt += f"\n{note}"

    ax.text(
        0.03, 0.97, txt,
        transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff", "alpha": 0.9, "edgecolor": "#aaa"},
        zorder=5,
    )
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Permutation count", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold" if highlight else "normal")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_combined_figure(
    observed: dict,
    perm_df: pd.DataFrame,
    out_png: Path,
    metric_label: str = "Mutual Information KSG",
    n_perms: int = 16384,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        f"{metric_label} | Laplacian Spectral Distance\n"
        f"Label-permutation test: observed vs sign-permuted session labels  "
        f"(exhaustive, n={n_perms:,} permutations)",
        fontsize=11, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.52, wspace=0.38)

    panels = [
        (0, 0, "mu_off_off", "right",
         "Mean OFF-OFF distance (Laplacian)", "OFF-OFF pair distances",
         "H₁ (one-sided): OFF state more\nidiosyncratic than random assignment",
         False),
        (0, 1, "mu_on_on", "left",
         "Mean ON-ON distance (Laplacian)", "ON-ON pair distances",
         "H₁ (one-sided): ON state more\nhomogeneous than random assignment",
         False),
        (1, 0, "mu_off_on", "two_sided",
         "Mean OFF-ON distance (Laplacian)", "OFF-ON pair distances",
         "Cross-state distances vs\npermuted null (two-sided)",
         False),
        (1, 1, "delta_within", "left",
         "Δ_within = ON-ON − OFF-OFF (Laplacian)",
         "Δ_within: KEY — medication-state asymmetry",
         "H₁ (one-sided): ON-ON < OFF-OFF\nnot attributable to random labels",
         True),
    ]

    for (row, col, key, p_type, xlabel, title, note, highlight) in panels:
        ax = fig.add_subplot(gs[row, col])
        _plot_panel(
            ax=ax,
            null=perm_df[key].to_numpy(dtype=np.float64),
            observed=float(observed[key]),
            xlabel=xlabel, title=title,
            p_type=p_type, note=note, highlight=highlight,
        )
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Combined figure → {out_png}")


def make_individual_figures(
    observed: dict,
    perm_df: pd.DataFrame,
    out_dir: Path,
    metric_label: str = "MI_KSG",
) -> None:
    specs = [
        ("mu_off_off", "right",
         "Mean OFF-OFF graph distance (Laplacian spectral)",
         "OFF-OFF pair distances vs permutation null",
         "H₁ (one-sided right): OFF-state\nidiosyncratic connectivity > random",
         False),
        ("mu_on_on", "left",
         "Mean ON-ON graph distance (Laplacian spectral)",
         "ON-ON pair distances vs permutation null",
         "H₁ (one-sided left): ON-state\nhomogenised connectivity < random",
         False),
        ("mu_off_on", "two_sided",
         "Mean OFF-ON graph distance (Laplacian spectral)",
         "OFF-ON pair distances vs permutation null",
         "Cross-state distances vs permuted null (two-sided)",
         False),
        ("delta_within", "left",
         "Δ_within = ON-ON − OFF-OFF (Laplacian spectral)",
         "Δ_within: KEY medication-state asymmetry",
         "H₁ (one-sided left): ON-ON < OFF-OFF\ncannot arise from random label assignment",
         True),
        ("delta_sep", "right",
         "Δ_sep = OFF-ON − ½(OFF-OFF + ON-ON) (Laplacian spectral)",
         "Δ_sep: cross-state separation",
         None,
         False),
    ]
    for key, p_type, xlabel, title, note, highlight in specs:
        fig, ax = plt.subplots(figsize=(6.4, 4.5))
        _plot_panel(
            ax=ax,
            null=perm_df[key].to_numpy(dtype=np.float64),
            observed=float(observed[key]),
            xlabel=xlabel, title=f"{metric_label} | {title}",
            p_type=p_type, note=note, highlight=highlight,
        )
        fig.tight_layout()
        fname = out_dir / f"permutation_null_{key}.png"
        fig.savefig(fname, dpi=190, bbox_inches="tight")
        plt.close(fig)
        print(f"  {fname.name}")


# ---------------------------------------------------------------------------
# Summary CSV + console
# ---------------------------------------------------------------------------

def save_summary(
    observed: dict,
    perm_df: pd.DataFrame,
    out_csv: Path,
) -> None:
    rows = []
    n_perms = len(perm_df)
    for key, direction in [
        ("mu_off_off",   "right"),    # H1: OFF-OFF > null (idiosyncratic off state)
        ("mu_on_on",     "left"),     # H1: ON-ON < null (homogenised on state)
        ("mu_off_on",    "two_sided"),
        ("delta_within", "left"),     # H1: ON-ON - OFF-OFF < 0
        ("delta_sep",    "right"),
    ]:
        null = perm_df[key].to_numpy(dtype=np.float64)
        obs = float(observed[key])
        if direction == "two_sided":
            p = _p_two_sided(null, obs)
        elif direction == "left":
            p = _p_left(null, obs)
        else:
            p = _p_right(null, obs)
        z = _z(null, obs)
        pct = _pct(null, obs)
        rows.append({
            "contrast":          key,
            "observed":          obs,
            "null_mean":         float(np.mean(null)),
            "null_sd":           float(np.std(null, ddof=1)),
            "z":                 z,
            "p_value":           p,
            "p_direction":       direction,
            "percentile_vs_null": pct,
            "n_permutations":    n_perms,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n  Summary CSV → {out_csv}")

    print("\n" + "=" * 70)
    print("PERMUTATION TEST SUMMARY  (label-permutation, exhaustive 2^14)")
    print("Metric: mutual_information_ksg  |  Distance: Laplacian spectral")
    print("=" * 70)
    for _, r in df.iterrows():
        p = r["p_value"]
        stars = ("***" if p <= 0.001 else "**" if p <= 0.01
                 else "*" if p <= 0.05 else "n.s.")
        print(f"  {r['contrast']:20s}: observed={r['observed']:+.5f}  "
              f"null={r['null_mean']:+.5f}±{r['null_sd']:.5f}  "
              f"z={r['z']:+.2f}  p={p:.5f} {stars}  pct={r['percentile_vs_null']:.1f}%")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairwise-csv", type=Path, default=DEFAULT_PAIRWISE_CSV)
    p.add_argument("--out-dir",      type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--metric-label", default="Mutual Information KSG")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    csv_path = args.pairwise_csv.expanduser().resolve()
    out_dir  = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairwise distances from:\n  {csv_path}")
    df = pd.read_csv(csv_path)
    n_subjects = df["subject_a"].nunique()
    n_perms = 2 ** n_subjects
    print(f"Subjects: {n_subjects}  →  {n_perms:,} exhaustive permutations")

    print("Running permutations …")
    observed, perm_df = run_exhaustive_permutations(df)
    print("Done.")

    save_summary(observed, perm_df, out_dir / "permutation_null_summary.csv")

    print("\nGenerating figures …")
    make_combined_figure(
        observed, perm_df,
        out_dir / "permutation_null_combined.png",
        metric_label=args.metric_label,
        n_perms=n_perms,
    )
    make_individual_figures(observed, perm_df, out_dir, metric_label=args.metric_label)

    print(f"\nAll outputs in: {out_dir}")
    print("Primary figure: permutation_null_combined.png")
    print("\nKey result: look at delta_within p-value.")
    print("If p < 0.05: the observed ON-ON < OFF-OFF pattern is not attributable")
    print("to random label assignment — the medication effect on graph distance is real.")


if __name__ == "__main__":
    main()
