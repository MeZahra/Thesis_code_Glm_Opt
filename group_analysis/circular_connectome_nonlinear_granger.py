"""
Circular connectome plots for nonlinear_granger across all condition pairs.
"Significant" edges = |Δ| > mean + 1.5*std of all off-diagonal |Δ| values.
Produces:
  - nonlinear_granger_circular_gvs_vs_sham_all_pairs.png  (2×4 grid)
  - nonlinear_granger_circular_gvs_vs_gvs_all_pairs.png   (4×7 grid)
  - nonlinear_granger_circular_all_pairs_individual/       (one file per pair)
"""

import os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import sys

BASE   = "/home/zkavian/Thesis_code_Glm_Opt/results/connectivity/roi_edge_network/advanced_metrics"
METRIC = sys.argv[1] if len(sys.argv) > 1 else "nonlinear_granger"
OUT    = os.path.join(BASE, "condition_edge_comparison_v2", METRIC)
os.makedirs(OUT, exist_ok=True)
IND    = os.path.join(OUT, "individual_pairs")
os.makedirs(IND, exist_ok=True)

CONDITIONS = ["sham", "GVS1", "GVS2", "GVS3", "GVS4", "GVS5", "GVS6", "GVS7", "GVS8"]
GVS_CONDS  = [c for c in CONDITIONS if c != "sham"]

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading matrices …")
DATA = {}
for cond in CONDITIONS:
    DATA[cond] = np.load(f"{BASE}/{cond}/{METRIC}/{METRIC}.npy")

LABELS = open(f"{BASE}/{CONDITIONS[0]}/{METRIC}/{METRIC}_connectome.labels.txt").read().strip().split("\n")
N = len(LABELS)

# Directed metric — use full off-diagonal
MASK = ~np.eye(N, dtype=bool)
ROWS, COLS = np.where(MASK)

# ── Threshold: mean + 1.5*std across ALL condition pairs ──────────────────────
print("Computing adaptive threshold …")
all_abs_diffs = []
for ca, cb in itertools.combinations(CONDITIONS, 2):
    d = np.abs(DATA[cb][MASK] - DATA[ca][MASK])
    all_abs_diffs.append(d)
all_abs_diffs = np.concatenate(all_abs_diffs)
THRESHOLD = all_abs_diffs.mean() + 1.5 * all_abs_diffs.std()
print(f"  Global threshold = {THRESHOLD:.5f}  "
      f"(mean={all_abs_diffs.mean():.5f}, std={all_abs_diffs.std():.5f})")

# ── Node layout: circle ───────────────────────────────────────────────────────
ANGLES = np.linspace(0, 2 * np.pi, N, endpoint=False) - np.pi / 2
XS     = np.cos(ANGLES)
YS     = np.sin(ANGLES)

# Short labels (remove "(relative)" clutter for display)
SHORT = [l.replace(" (relative)", "").replace(" (Control & monitoring)", "")
          for l in LABELS]

# Colour map for signed change
COL_INC = "#d62728"   # red  = connectivity increased in B vs A
COL_DEC = "#1f77b4"   # blue = decreased

# ── Core draw function ────────────────────────────────────────────────────────

def draw_circular(ax, ca, cb, threshold=THRESHOLD, top_n=None,
                  title=None, fontsize=5.5):
    """
    Draw circular connectome for pair (ca → cb).
    Edges shown: |Δ| > threshold  OR  top_n largest if threshold gives < 5.
    Arrowheads show direction (directed metric).
    """
    va   = DATA[ca][MASK]
    vb   = DATA[cb][MASK]
    diff = vb - va
    adif = np.abs(diff)

    sig  = adif >= threshold
    if sig.sum() < 5 and top_n is None:          # fallback: show at least 5
        top_n_fallback = 5
        idx_sorted = np.argsort(adif)[::-1][:top_n_fallback]
        sig = np.zeros(len(adif), dtype=bool)
        sig[idx_sorted] = True

    if top_n is not None:
        idx_sorted = np.argsort(adif)[::-1][:top_n]
        sig = np.zeros(len(adif), dtype=bool)
        sig[idx_sorted] = True

    n_sig = sig.sum()
    vmax  = adif[sig].max() if n_sig > 0 else 1.0

    # Draw edges
    for k in np.where(sig)[0]:
        i, j = ROWS[k], COLS[k]
        d    = diff[k]
        col  = COL_INC if d > 0 else COL_DEC
        alpha = 0.25 + 0.75 * (adif[k] / vmax)
        lw    = 0.4  + 2.5  * (adif[k] / vmax)

        # Arrow: from i → j (directed)
        dx = XS[j] - XS[i]
        dy = YS[j] - YS[i]
        # shorten arrow so head is visible
        shrink = 0.06
        ax.annotate(
            "", xy=(XS[j] - shrink * dx, YS[j] - shrink * dy),
            xytext=(XS[i] + shrink * dx, YS[i] + shrink * dy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=col, lw=lw, alpha=alpha,
                mutation_scale=6,
            ),
            zorder=2,
        )

    # Nodes
    ax.scatter(XS, YS, s=30, c="#2c3e50", zorder=4,
               edgecolors="white", linewidths=0.4)

    # Labels
    for i in range(N):
        angle_deg = np.degrees(ANGLES[i])
        ha        = "left" if XS[i] >= 0 else "right"
        rot       = angle_deg if -90 <= angle_deg <= 90 else angle_deg + 180
        ax.text(1.18 * XS[i], 1.18 * YS[i], SHORT[i],
                fontsize=fontsize, ha=ha, va="center",
                rotation=rot, rotation_mode="anchor",
                color="#1a1a2e")

    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_aspect("equal")
    ax.axis("off")
    t = title or f"{ca} → {cb}"
    ax.set_title(f"{t}\n({n_sig} sig. edges, thr={threshold:.4f})",
                 fontsize=7.5, fontweight="bold", pad=3)


def legend_handles():
    return [
        mpatches.Patch(color=COL_INC, label="↑ in B (stronger)"),
        mpatches.Patch(color=COL_DEC, label="↓ in B (weaker)"),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  Figure A — GVS vs Sham  (8 panels, 2 rows × 4 cols)
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting GVS vs Sham  (2×4 grid) …")
fig, axes = plt.subplots(2, 4, figsize=(32, 17))
axes = axes.flat

for ax, gvs in zip(axes, GVS_CONDS):
    draw_circular(ax, ca="sham", cb=gvs, title=f"sham → {gvs}")

fig.legend(handles=legend_handles(), loc="lower center",
           ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(
    f"Nonlinear Granger — Significantly Changed Edges: GVS vs Sham\n"
    f"(threshold = {THRESHOLD:.4f}; arrowhead = direction of causal influence)",
    fontsize=14, y=1.01,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"{METRIC}_circular_gvs_vs_sham_all_pairs.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("  saved gvs_vs_sham grid.")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure B — GVS vs GVS  (28 panels, 4 rows × 7 cols)
# ══════════════════════════════════════════════════════════════════════════════
print("Plotting GVS vs GVS  (4×7 grid) …")
gvs_pairs = list(itertools.combinations(GVS_CONDS, 2))  # 28 pairs

fig, axes = plt.subplots(4, 7, figsize=(56, 33))
axes = axes.flat

for ax, (ca, cb) in zip(axes, gvs_pairs):
    draw_circular(ax, ca=ca, cb=cb)

fig.legend(handles=legend_handles(), loc="lower center",
           ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(
    f"Nonlinear Granger — Significantly Changed Edges: GVS vs GVS\n"
    f"(threshold = {THRESHOLD:.4f}; arrowhead = direction of causal influence)",
    fontsize=18, y=1.005,
)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"{METRIC}_circular_gvs_vs_gvs_all_pairs.png"),
            dpi=120, bbox_inches="tight")
plt.close(fig)
print("  saved gvs_vs_gvs grid.")


# ══════════════════════════════════════════════════════════════════════════════
#  Individual files — one PNG per pair (all 36)
# ══════════════════════════════════════════════════════════════════════════════
print("Saving individual pair plots …")
all_pairs = [("sham", g) for g in GVS_CONDS] + list(itertools.combinations(GVS_CONDS, 2))

for ca, cb in all_pairs:
    fig, ax = plt.subplots(figsize=(9, 9))
    draw_circular(ax, ca=ca, cb=cb, fontsize=7)
    ax.legend(handles=legend_handles(), loc="lower left",
              fontsize=8, bbox_to_anchor=(-0.05, -0.08))
    fig.tight_layout()
    fname = f"{METRIC}_circular_{ca}_vs_{cb}.png"
    fig.savefig(os.path.join(IND, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"  saved {len(all_pairs)} individual plots → {IND}")
print("Done.")
