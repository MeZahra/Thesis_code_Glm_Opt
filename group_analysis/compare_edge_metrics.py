"""
Comprehensive multi-metric edge connectivity comparison across GVS conditions.

Loads all connectivity matrices from results/connectivity/roi_edge_network/advanced_metrics,
computes edge-level statistics, produces heatmaps, connectome plots, and a full summary table.
"""

import sys, os, json, itertools, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm, Normalize
import matplotlib.patches as mpatches
from scipy import stats

# ── paths ──────────────────────────────────────────────────────────────────────
BASE   = "/home/zkavian/Thesis_code_Glm_Opt/results/connectivity/roi_edge_network/advanced_metrics"
OUT    = os.path.join(BASE, "condition_edge_comparison_v2")
os.makedirs(OUT, exist_ok=True)

CONDITIONS = ["sham", "GVS1", "GVS2", "GVS3", "GVS4", "GVS5", "GVS6", "GVS7", "GVS8"]
GVS_CONDS  = [c for c in CONDITIONS if c != "sham"]

METRICS = [
    "graph_correlation_network",
    "instantaneous_phase_sync",
    "linear_granger",
    "mutual_information",
    "mutual_information_ksg",
    "nonlinear_granger",
    "partial_correlation",
    "wavelet_transform_coherence",
]

METRIC_LABELS = {
    "graph_correlation_network":   "Graph Corr",
    "instantaneous_phase_sync":    "Phase Sync",
    "linear_granger":              "Lin Granger",
    "mutual_information":          "MI (naive)",
    "mutual_information_ksg":      "MI-KSG",
    "nonlinear_granger":           "Nonlin Granger",
    "partial_correlation":         "Partial Corr",
    "wavelet_transform_coherence": "Wavelet Coh",
}

PALETTE = plt.cm.get_cmap("tab10")
ALWAYS_EXCLUDED_ROI_PATTERNS = ("ventricular csf", "ventrical csf", "lateral ventricle")

# ── helpers ────────────────────────────────────────────────────────────────────

def load_matrix(cond, metric):
    p = os.path.join(BASE, cond, metric, f"{metric}.npy")
    return np.load(p)


def load_labels(metric):
    # labels file from any condition (same across conditions)
    p = os.path.join(BASE, CONDITIONS[0], metric,
                     f"{metric}_connectome.labels.txt")
    return open(p).read().strip().split("\n")


def keep_node_mask(labels):
    labels_lower = [str(v).strip().lower() for v in labels]
    return np.asarray(
        [not any(pattern in label for pattern in ALWAYS_EXCLUDED_ROI_PATTERNS) for label in labels_lower],
        dtype=bool,
    )


def is_directed(metric):
    p = os.path.join(BASE, CONDITIONS[0], metric, f"{metric}_meta.json")
    with open(p) as f:
        meta = json.load(f)
    return bool(meta.get("directed", 0))


def edge_vectors(matrix, directed):
    """Return flat vector of upper-triangle (or all off-diagonal) edges."""
    n = matrix.shape[0]
    if directed:
        mask = ~np.eye(n, dtype=bool)
    else:
        mask = np.triu(np.ones(n, dtype=bool), k=1)
    return matrix[mask], mask


def edge_labels(labels, mask, directed, n):
    """Return list of (i,j,label_i,label_j,name) for each selected edge."""
    rows, cols = np.where(mask)
    out = []
    for r, c in zip(rows, cols):
        if directed:
            name = f"{labels[r]} -> {labels[c]}"
        else:
            name = f"{labels[r]} -- {labels[c]}"
        out.append((r, c, labels[r], labels[c], name))
    return out


def wrap(s, w=20):
    return "\n".join(textwrap.wrap(s, w))


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  Load all data
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading matrices …")
DATA   = {}   # DATA[metric][cond] = (n×n matrix)
LABELS = {}   # LABELS[metric] = list of node-name strings
DIRECTED = {} # DIRECTED[metric] = bool

for metric in METRICS:
    DIRECTED[metric] = is_directed(metric)
    labels_raw = load_labels(metric)
    keep_mask = keep_node_mask(labels_raw)
    LABELS[metric] = [label for label, keep in zip(labels_raw, keep_mask.tolist()) if keep]
    DATA[metric]     = {}
    for cond in CONDITIONS:
        try:
            mat = load_matrix(cond, metric)
            if mat.shape[0] == keep_mask.size and mat.shape[1] == keep_mask.size:
                mat = mat[np.ix_(keep_mask, keep_mask)]
            elif mat.shape[0] != len(LABELS[metric]) or mat.shape[1] != len(LABELS[metric]):
                raise ValueError(
                    f"Unexpected matrix shape for {cond}/{metric}: {mat.shape}, "
                    f"expected {(keep_mask.size, keep_mask.size)} or {(len(LABELS[metric]), len(LABELS[metric]))}"
                )
            DATA[metric][cond] = mat
        except FileNotFoundError:
            print(f"  MISSING: {cond}/{metric}")

print("  done.\n")

# Compute edge-level flat vectors for each metric × condition
EDGE_VEC  = {}   # EDGE_VEC[metric][cond] = 1-D array of edge values
EDGE_INFO = {}   # EDGE_INFO[metric] = list of (i,j,label_i,label_j,name)

for metric in METRICS:
    n = DATA[metric][CONDITIONS[0]].shape[0]
    directed = DIRECTED[metric]
    EDGE_INFO[metric] = None
    EDGE_VEC[metric]  = {}
    for cond in CONDITIONS:
        mat = DATA[metric][cond]
        vec, mask = edge_vectors(mat, directed)
        EDGE_VEC[metric][cond] = vec
        if EDGE_INFO[metric] is None:
            EDGE_INFO[metric] = edge_labels(LABELS[metric], mask, directed, n)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Pairwise condition differences (mean |Δ| per edge)
# ═══════════════════════════════════════════════════════════════════════════════
print("Computing pairwise condition differences …")

# Build a DataFrame: rows = all condition pairs, cols = metrics
# Value = mean |Δ| across edges
pair_records = []
ALL_PAIRS = list(itertools.combinations(CONDITIONS, 2))

for ca, cb in ALL_PAIRS:
    pair_type = "gvs_vs_sham" if "sham" in (ca, cb) else "gvs_vs_gvs"
    rec = {"cond_a": ca, "cond_b": cb, "pair": f"{ca} vs {cb}",
           "pair_type": pair_type}
    for metric in METRICS:
        va = EDGE_VEC[metric][ca]
        vb = EDGE_VEC[metric][cb]
        rec[metric] = float(np.mean(np.abs(va - vb)))
    pair_records.append(rec)

df_pairs = pd.DataFrame(pair_records)
df_pairs.to_csv(os.path.join(OUT, "all_pairwise_mean_abs_diff.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  3.  Figure 1 — 9×9 pairwise mean-|Δ| heatmap per metric (multi-panel)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 1 — per-metric 9×9 condition heatmaps …")

fig, axes = plt.subplots(2, 4, figsize=(28, 14))
axes = axes.flat

for ax, metric in zip(axes, METRICS):
    # Build 9×9 symmetric matrix
    mat = np.zeros((9, 9))
    for i, ca in enumerate(CONDITIONS):
        for j, cb in enumerate(CONDITIONS):
            if i == j:
                mat[i, j] = 0
            else:
                va = EDGE_VEC[metric][ca]
                vb = EDGE_VEC[metric][cb]
                mat[i, j] = np.mean(np.abs(va - vb))

    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(CONDITIONS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CONDITIONS, fontsize=8)
    ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="mean |Δ|", shrink=0.8)

    # Annotate cells
    vmax = mat.max()
    for i in range(9):
        for j in range(9):
            if i != j:
                v = mat[i, j]
                color = "white" if v > 0.6 * vmax else "black"
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=5.5, color=color)

fig.suptitle("Mean |Δ| per Edge — All Condition Pairs × All Metrics", fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_per_metric_pairwise_heatmaps.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  4.  Figure 2 — cross-metric sensitivity bar chart (GVS vs Sham / GVS vs GVS)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 2 — cross-metric sensitivity bar chart …")

df_sham = df_pairs[df_pairs.pair_type == "gvs_vs_sham"]
df_gvsg = df_pairs[df_pairs.pair_type == "gvs_vs_gvs"]

metric_order = METRICS  # keep original order

mean_sham = {m: df_sham[m].mean() for m in METRICS}
mean_gvsg = {m: df_gvsg[m].mean() for m in METRICS}
std_sham  = {m: df_sham[m].std()  for m in METRICS}
std_gvsg  = {m: df_gvsg[m].std()  for m in METRICS}

# Normalise to [0,1] separately so both series are comparable
max_sham = max(mean_sham.values()); max_gvsg = max(mean_gvsg.values())
norm_sham = {m: v / max_sham for m, v in mean_sham.items()}
norm_gvsg = {m: v / max_gvsg for m, v in mean_gvsg.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
labels_short = [METRIC_LABELS[m] for m in METRICS]
x = np.arange(len(METRICS))
w = 0.35

# Raw values
ax = axes[0]
bars1 = ax.bar(x - w/2, [mean_sham[m] for m in METRICS], w,
               label="GVS vs Sham", color="#4c72b0", alpha=0.85,
               yerr=[std_sham[m] for m in METRICS], capsize=4)
bars2 = ax.bar(x + w/2, [mean_gvsg[m] for m in METRICS], w,
               label="GVS vs GVS", color="#dd8452", alpha=0.85,
               yerr=[std_gvsg[m] for m in METRICS], capsize=4)
ax.set_xticks(x); ax.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean |Δ| across edges"); ax.set_title("Raw sensitivity")
ax.legend(); ax.grid(axis="y", alpha=0.3)

# Normalised
ax = axes[1]
ax.bar(x - w/2, [norm_sham[m] for m in METRICS], w,
       label="GVS vs Sham (norm)", color="#4c72b0", alpha=0.85)
ax.bar(x + w/2, [norm_gvsg[m] for m in METRICS], w,
       label="GVS vs GVS (norm)", color="#dd8452", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Normalised sensitivity (0–1)"); ax.set_title("Normalised sensitivity")
ax.legend(); ax.grid(axis="y", alpha=0.3)

fig.suptitle("Cross-Metric Sensitivity: GVS vs Sham  vs  GVS vs GVS", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_cross_metric_sensitivity_bars.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  5.  Figure 3 — per-metric edge-difference heatmaps for strongest pair
#       (both GVS-vs-sham and GVS-vs-GVS)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 3 — edge-difference heatmaps for strongest pairs …")

def strongest_pair(df, metric, pair_type):
    sub = df[df.pair_type == pair_type]
    idx = sub[metric].idxmax()
    row = sub.loc[idx]
    return row.cond_a, row.cond_b

def signed_diff_matrix(cond_a, cond_b, metric):
    return DATA[metric][cond_b] - DATA[metric][cond_a]


fig, axes = plt.subplots(len(METRICS), 2, figsize=(16, 4 * len(METRICS)))

for row_idx, metric in enumerate(METRICS):
    labels = LABELS[metric]
    short  = METRIC_LABELS[metric]

    for col_idx, pair_type in enumerate(["gvs_vs_sham", "gvs_vs_gvs"]):
        ax = axes[row_idx, col_idx]
        ca, cb = strongest_pair(df_pairs, metric, pair_type)
        diff   = signed_diff_matrix(ca, cb, metric)

        vabs = np.abs(diff).max() or 1.0
        norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)
        im = ax.imshow(diff, cmap="RdBu_r", norm=norm, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Δ (B–A)")

        n = len(labels)
        tick_step = max(1, n // 12)
        tick_idx  = list(range(0, n, tick_step))
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([wrap(labels[i], 14) for i in tick_idx],
                           rotation=60, ha="right", fontsize=6)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([wrap(labels[i], 14) for i in tick_idx], fontsize=6)

        tag = "GVS vs Sham" if pair_type == "gvs_vs_sham" else "GVS vs GVS"
        ax.set_title(f"{short} | {tag}\n{ca} → {cb}", fontsize=9, fontweight="bold")

fig.suptitle("Edge-Level Difference Heatmaps — Strongest Pairs per Metric", fontsize=13, y=1.001)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_strongest_pair_edge_diff_heatmaps.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  6.  Figure 4 — top-K edges per metric per contrast (scatter / ranked bar)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 4 — top-20 edges per metric per contrast …")

TOP_K = 20

fig, axes = plt.subplots(len(METRICS), 2, figsize=(22, 5 * len(METRICS)))
top_edge_records = []

for row_idx, metric in enumerate(METRICS):
    edge_info = EDGE_INFO[metric]
    n_edges   = len(edge_info)

    for col_idx, pair_type in enumerate(["gvs_vs_sham", "gvs_vs_gvs"]):
        ax = axes[row_idx, col_idx]
        ca, cb = strongest_pair(df_pairs, metric, pair_type)
        va  = EDGE_VEC[metric][ca]
        vb  = EDGE_VEC[metric][cb]
        dif = vb - va
        abs_dif = np.abs(dif)

        top_idx  = np.argsort(abs_dif)[::-1][:TOP_K]
        top_vals = abs_dif[top_idx]
        top_sgn  = np.sign(dif[top_idx])
        top_names = [edge_info[i][4] for i in top_idx]

        colors = ["#e05a5a" if s > 0 else "#5a7be0" for s in top_sgn]
        y_pos  = np.arange(TOP_K)

        ax.barh(y_pos, top_vals, color=colors, edgecolor="none", height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([wrap(n, 35) for n in top_names], fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel("|Δ|")
        tag = "GVS vs Sham" if pair_type == "gvs_vs_sham" else "GVS vs GVS"
        ax.set_title(f"{METRIC_LABELS[metric]} | {tag}\n{ca} → {cb}", fontsize=9)
        ax.axvline(0, color="k", lw=0.5)

        red_p  = mpatches.Patch(color="#e05a5a", label=f"{cb} > {ca}")
        blue_p = mpatches.Patch(color="#5a7be0", label=f"{ca} > {cb}")
        ax.legend(handles=[red_p, blue_p], fontsize=7, loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        for rank, (idx, val, sgn) in enumerate(
                zip(top_idx, top_vals, top_sgn), start=1):
            ei = edge_info[idx]
            top_edge_records.append({
                "rank": rank, "metric": metric, "contrast": tag,
                "pair": f"{ca} vs {cb}",
                "node_i": ei[0], "node_j": ei[1],
                "label_i": ei[2], "label_j": ei[3], "edge": ei[4],
                "delta_signed": float(dif[top_idx[rank-1]]),
                "delta_abs": float(val),
                "direction": f"{cb} > {ca}" if sgn > 0 else f"{ca} > {cb}",
            })

fig.suptitle(f"Top-{TOP_K} Most Changed Edges — Strongest Pair per Metric",
             fontsize=13, y=1.001)
fig.tight_layout()
fig.savefig(os.path.join(OUT, f"fig4_top{TOP_K}_edges_per_metric.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)

df_top_edges = pd.DataFrame(top_edge_records)
df_top_edges.to_csv(os.path.join(OUT, "top_edges_per_metric_per_contrast.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  7.  Figure 5 — metric × condition separability: effect-size (Cohen's d proxy)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 5 — effect-size matrix (metric × condition-pair) …")

# For each metric × pair, compute effect-size = mean|Δ| / pooled_std_of_edges
es_records = []
for ca, cb in ALL_PAIRS:
    pair_type = "gvs_vs_sham" if "sham" in (ca, cb) else "gvs_vs_gvs"
    for metric in METRICS:
        va = EDGE_VEC[metric][ca]
        vb = EDGE_VEC[metric][cb]
        pooled_std = (np.std(va) + np.std(vb)) / 2 + 1e-12
        es = np.mean(np.abs(va - vb)) / pooled_std
        es_records.append({"pair": f"{ca}\nvs\n{cb}",
                            "pair_type": pair_type,
                            "metric": METRIC_LABELS[metric],
                            "effect_size": es})

df_es = pd.DataFrame(es_records)

# Pivot to matrix form
es_pivot = df_es.pivot_table(index="pair", columns="metric",
                              values="effect_size", aggfunc="mean")

fig, ax = plt.subplots(figsize=(14, 22))
im = ax.imshow(es_pivot.values, cmap="viridis", aspect="auto")
plt.colorbar(im, ax=ax, label="Effect size (mean|Δ| / pooled σ)")
ax.set_xticks(range(len(es_pivot.columns)))
ax.set_xticklabels(es_pivot.columns, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(es_pivot.index)))
ax.set_yticklabels(es_pivot.index, fontsize=7)
ax.set_title("Effect Size per Condition Pair × Metric\n(mean|Δ| / pooled σ of edges)",
             fontsize=12)

# Annotate
vmax_es = es_pivot.values.max()
for i in range(len(es_pivot.index)):
    for j in range(len(es_pivot.columns)):
        v = es_pivot.values[i, j]
        col = "white" if v < 0.6 * vmax_es else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color=col)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_effect_size_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)
df_es.to_csv(os.path.join(OUT, "effect_size_per_pair_metric.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  8.  Figure 6 — per-metric full-condition connectivity heatmap grid
#       (one row = one condition, one col = one metric)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 6 — full connectivity heatmap grid …")

fig, axes = plt.subplots(len(CONDITIONS), len(METRICS),
                         figsize=(4 * len(METRICS), 4 * len(CONDITIONS)))

for ri, cond in enumerate(CONDITIONS):
    for ci, metric in enumerate(METRICS):
        ax = axes[ri, ci]
        mat = DATA[metric][cond]
        vmax = np.abs(mat).max() or 1
        cmap = "RdBu_r" if mat.min() < -0.05 else "hot_r"
        im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.axis("off")
        if ri == 0:
            ax.set_title(METRIC_LABELS[metric], fontsize=8, fontweight="bold")
        if ci == 0:
            ax.set_ylabel(cond, fontsize=8)
            ax.text(-0.05, 0.5, cond, transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=8, fontweight="bold")

fig.suptitle("Connectivity Matrices per Condition × Metric", fontsize=14, y=1.001)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_full_connectivity_grid.png"),
            dpi=100, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  9.  Figure 7 — connectome-style dot plot for top changed edges
#       (nodes arranged on circle, top edges drawn as arcs coloured by Δ sign)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 7 — circular connectome plots for strongest pairs …")

def circular_connectome(ax, edge_info_all, diffs, labels, title,
                        top_k=25, directed=False):
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = np.cos(angles)
    ys = np.sin(angles)

    abs_diff = np.abs(diffs)
    top_idx  = np.argsort(abs_diff)[::-1][:top_k]

    vmax = abs_diff[top_idx].max() or 1.0

    for rank, eidx in enumerate(top_idx):
        ei  = edge_info_all[eidx]
        i, j = ei[0], ei[1]
        d = diffs[eidx]
        alpha = 0.3 + 0.7 * (abs_diff[eidx] / vmax)
        col   = "#cc3333" if d > 0 else "#3355cc"
        lw    = 0.5 + 2.5 * (abs_diff[eidx] / vmax)
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color=col, alpha=alpha, linewidth=lw, zorder=2)
        if directed and d > 0:
            ax.annotate("", xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=lw * 0.6, mutation_scale=8))

    # Draw nodes
    ax.scatter(xs, ys, s=60, c="steelblue", zorder=3, edgecolors="white", linewidths=0.5)
    for i, (x, y, lab) in enumerate(zip(xs, ys, labels)):
        angle_deg = np.degrees(angles[i])
        ha = "left" if x >= 0 else "right"
        rotation = angle_deg if -90 <= angle_deg <= 90 else angle_deg + 180
        ax.text(1.12 * x, 1.12 * y, lab,
                fontsize=4.5, ha=ha, va="center",
                rotation=rotation, rotation_mode="anchor")

    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold")

    # Legend
    red_p  = mpatches.Patch(color="#cc3333", label="Increased in B")
    blue_p = mpatches.Patch(color="#3355cc", label="Decreased in B")
    ax.legend(handles=[red_p, blue_p], fontsize=6, loc="lower left",
              bbox_to_anchor=(-0.05, -0.08))


fig, axes = plt.subplots(len(METRICS), 2,
                         figsize=(18, 9 * len(METRICS)))

for row_idx, metric in enumerate(METRICS):
    edge_info = EDGE_INFO[metric]
    labels    = LABELS[metric]
    direc     = DIRECTED[metric]

    for col_idx, pair_type in enumerate(["gvs_vs_sham", "gvs_vs_gvs"]):
        ax = axes[row_idx, col_idx]
        ca, cb = strongest_pair(df_pairs, metric, pair_type)
        va = EDGE_VEC[metric][ca]
        vb = EDGE_VEC[metric][cb]
        dif = vb - va

        tag = "GVS vs Sham" if pair_type == "gvs_vs_sham" else "GVS vs GVS"
        title = f"{METRIC_LABELS[metric]}\n{tag}: {ca} → {cb}"
        circular_connectome(ax, edge_info, dif, labels, title,
                            top_k=25, directed=direc)

fig.suptitle("Circular Connectome — Top-25 Most Changed Edges per Metric",
             fontsize=14, y=1.001)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_circular_connectome_strongest_pairs.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 10.  Figure 8 — condition-trajectory plot per metric
#      (mean edge value across all GVS conditions, sham as reference line)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 8 — condition trajectory plots …")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flat

for ax, metric in zip(axes, METRICS):
    sham_mean = np.mean(EDGE_VEC[metric]["sham"])
    sham_std  = np.std(EDGE_VEC[metric]["sham"])

    gvs_means = [np.mean(EDGE_VEC[metric][c]) for c in GVS_CONDS]
    gvs_stds  = [np.std(EDGE_VEC[metric][c])  for c in GVS_CONDS]
    gvs_max   = [np.percentile(np.abs(EDGE_VEC[metric][c]), 95) for c in GVS_CONDS]

    x = np.arange(1, len(GVS_CONDS) + 1)
    ax.axhline(sham_mean, color="gray", ls="--", lw=1.5, label="Sham mean")
    ax.axhspan(sham_mean - sham_std, sham_mean + sham_std,
               alpha=0.12, color="gray", label="Sham ±σ")

    ax.errorbar(x, gvs_means, yerr=gvs_stds, fmt="o-", color="#4c72b0",
                capsize=4, label="GVS mean ±σ")
    ax.plot(x, gvs_max, "s--", color="#dd8452", alpha=0.7, label="GVS 95th pct")

    ax.set_xticks(x)
    ax.set_xticklabels(GVS_CONDS, fontsize=8)
    ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
    ax.set_xlabel("GVS condition"); ax.set_ylabel("Edge value")
    ax.legend(fontsize=6); ax.grid(alpha=0.3)

fig.suptitle("Mean Edge Value Across GVS Conditions (Sham as reference)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_condition_trajectory.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 11.  Figure 9 — per-metric: GVS vs Sham scatter (mean|Δ| per GVS level)
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 9 — GVS-level scatter per metric …")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flat

for ax, metric in zip(axes, METRICS):
    vals = []
    for cond in GVS_CONDS:
        va = EDGE_VEC[metric]["sham"]
        vb = EDGE_VEC[metric][cond]
        vals.append(np.mean(np.abs(va - vb)))

    x = np.arange(1, 9)
    ax.bar(x, vals, color=PALETTE(np.linspace(0, 1, 8)), edgecolor="white", width=0.7)
    # baseline: expected from GVS vs GVS variability
    baseline = np.mean([np.mean(np.abs(
        EDGE_VEC[metric][ca] - EDGE_VEC[metric][cb]))
        for ca, cb in itertools.combinations(GVS_CONDS, 2)])
    ax.axhline(baseline, color="red", ls="--", lw=1.2,
               label=f"Avg GVS–GVS diff = {baseline:.4f}")
    ax.set_xticks(x); ax.set_xticklabels(GVS_CONDS, fontsize=8)
    ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean |Δ| from Sham"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig.suptitle("Per-GVS-Level Separation from Sham\n(red dashed = avg GVS–GVS variability)",
             fontsize=13, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_gvs_sham_separation_per_level.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 12.  Summary table
# ═══════════════════════════════════════════════════════════════════════════════
print("Building summary table …")

rows = []
for metric in METRICS:
    directed = DIRECTED[metric]

    # Mean |Δ| GVS vs Sham / GVS vs GVS
    sham_pairs = df_pairs[df_pairs.pair_type == "gvs_vs_sham"][metric]
    gvsg_pairs = df_pairs[df_pairs.pair_type == "gvs_vs_gvs"][metric]
    mean_sham_diff = sham_pairs.mean()
    mean_gvsg_diff = gvsg_pairs.mean()
    cv_gvsg        = gvsg_pairs.std() / (gvsg_pairs.mean() + 1e-12)  # variability across GVS pairs
    snr            = mean_sham_diff / (mean_gvsg_diff + 1e-12)

    # Strongest pairs
    ca_s, cb_s = strongest_pair(df_pairs, metric, "gvs_vs_sham")
    ca_g, cb_g = strongest_pair(df_pairs, metric, "gvs_vs_gvs")
    val_s = df_pairs[df_pairs.pair_type=="gvs_vs_sham"][metric].max()
    val_g = df_pairs[df_pairs.pair_type=="gvs_vs_gvs"][metric].max()

    # KS test: are GVS edge distributions different from Sham?
    sham_vec = EDGE_VEC[metric]["sham"]
    ks_pvals = []
    for cond in GVS_CONDS:
        _, pv = stats.ks_2samp(sham_vec, EDGE_VEC[metric][cond])
        ks_pvals.append(pv)
    n_sig_sham = sum(p < 0.05 for p in ks_pvals)

    # KS test within GVS
    gvs_ks_pvals = []
    for ca, cb in itertools.combinations(GVS_CONDS, 2):
        _, pv = stats.ks_2samp(EDGE_VEC[metric][ca], EDGE_VEC[metric][cb])
        gvs_ks_pvals.append(pv)
    n_sig_gvs = sum(p < 0.05 for p in gvs_ks_pvals)
    n_total_gvs_pairs = len(gvs_ks_pvals)

    # Top key edges for strongest pairs
    def top3_edges(ca, cb, k=3):
        va, vb = EDGE_VEC[metric][ca], EDGE_VEC[metric][cb]
        dif = np.abs(vb - va)
        idx = np.argsort(dif)[::-1][:k]
        return "; ".join(EDGE_INFO[metric][i][4] + f" (|Δ|={dif[i]:.4f})"
                         for i in idx)

    rows.append({
        "Metric":                       METRIC_LABELS[metric],
        "Directed":                     "Yes" if directed else "No",
        "Mean|Δ| GVS vs Sham":         f"{mean_sham_diff:.5f}",
        "Mean|Δ| GVS vs GVS":          f"{mean_gvsg_diff:.5f}",
        "SNR (Sham/GVS ratio)":         f"{snr:.4f}",
        "CV GVS vs GVS":                f"{cv_gvsg:.4f}",
        "KS sig vs Sham (of 8)":        f"{n_sig_sham}/8",
        "KS sig within GVS":            f"{n_sig_gvs}/{n_total_gvs_pairs}",
        "Distinguish GVS vs Sham":      "Strong" if n_sig_sham >= 6 else
                                        "Partial" if n_sig_sham >= 3 else "Weak",
        "Distinguish between GVS":      "Strong" if n_sig_gvs >= 20 else
                                        "Partial" if n_sig_gvs >= 10 else "Weak",
        "Strongest GVS-Sham pair":      f"{ca_s} vs {cb_s} (|Δ|={val_s:.5f})",
        "Strongest GVS-GVS pair":       f"{ca_g} vs {cb_g} (|Δ|={val_g:.5f})",
        "Key edges (GVS vs Sham)":      top3_edges(ca_s, cb_s),
        "Key edges (GVS vs GVS)":       top3_edges(ca_g, cb_g),
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(OUT, "metric_comparison_summary_table.csv"), index=False)

# ─── Pretty-print summary ──────────────────────────────────────────────────────
print("\n" + "="*90)
print("METRIC COMPARISON SUMMARY")
print("="*90)
for _, r in df_summary.iterrows():
    print(f"\n{'─'*60}")
    print(f"  Metric:                  {r['Metric']}  (Directed: {r['Directed']})")
    print(f"  Mean|Δ| GVS vs Sham:     {r['Mean|Δ| GVS vs Sham']}")
    print(f"  Mean|Δ| GVS vs GVS:      {r['Mean|Δ| GVS vs GVS']}")
    print(f"  SNR (Sham / GVS ratio):  {r['SNR (Sham/GVS ratio)']}")
    print(f"  CV (GVS–GVS spread):     {r['CV GVS vs GVS']}")
    print(f"  KS sig vs Sham:          {r['KS sig vs Sham (of 8)']}")
    print(f"  KS sig within GVS:       {r['KS sig within GVS']}")
    print(f"  Distinguish GVS vs Sham: {r['Distinguish GVS vs Sham']}")
    print(f"  Distinguish between GVS: {r['Distinguish between GVS']}")
    print(f"  Strongest GVS–Sham pair: {r['Strongest GVS-Sham pair']}")
    print(f"  Strongest GVS–GVS pair:  {r['Strongest GVS-GVS pair']}")
    print(f"  Key edges (vs Sham):     {r['Key edges (GVS vs Sham)']}")
    print(f"  Key edges (vs GVS):      {r['Key edges (GVS vs GVS)']}")
print("="*90)


# ═══════════════════════════════════════════════════════════════════════════════
# 13.  Figure 10 — Summary radar / bubble chart
# ═══════════════════════════════════════════════════════════════════════════════
print("\nFigure 10 — summary bubble chart …")

ks_sham = [int(r["KS sig vs Sham (of 8)"].split("/")[0]) for _, r in df_summary.iterrows()]
ks_gvs  = [int(r["KS sig within GVS"].split("/")[0])     for _, r in df_summary.iterrows()]
mean_s  = [float(r["Mean|Δ| GVS vs Sham"]) for _, r in df_summary.iterrows()]
mean_g  = [float(r["Mean|Δ| GVS vs GVS"])  for _, r in df_summary.iterrows()]

fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(ks_sham, ks_gvs,
                     s=[v * 3000 / max(mean_s + [1e-9]) for v in mean_s],
                     c=mean_g, cmap="plasma", edgecolors="gray", linewidths=0.8,
                     alpha=0.85, zorder=3)
plt.colorbar(scatter, ax=ax, label="Mean|Δ| GVS vs GVS")

for i, row in df_summary.iterrows():
    ax.annotate(row["Metric"], (ks_sham[i], ks_gvs[i]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)

ax.set_xlabel("# GVS conditions significantly different from Sham (KS, p<0.05) / 8", fontsize=10)
ax.set_ylabel("# GVS–GVS pairs significantly different (KS, p<0.05) / 28",           fontsize=10)
ax.set_title("Metric Sensitivity Overview\n"
             "(bubble size ∝ mean|Δ| GVS-Sham, colour = mean|Δ| GVS-GVS)",
             fontsize=11)
ax.set_xlim(-0.5, 8.5); ax.set_ylim(-0.5, 28.5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig10_summary_bubble_chart.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 14.  Figure 11 — rank summary table image
# ═══════════════════════════════════════════════════════════════════════════════
print("Figure 11 — summary table image …")

# Compute a composite ranking score:
#   score = (norm KS_sham) * 0.4 + (norm KS_gvs) * 0.3 + (norm mean_sham) * 0.2 + (norm cv_gvs) * 0.1
def norm01(arr):
    mn, mx = min(arr), max(arr)
    if mx == mn: return [0.5] * len(arr)
    return [(v - mn) / (mx - mn) for v in arr]

cv_gvsg = [float(r["CV GVS vs GVS"]) for _, r in df_summary.iterrows()]

score = [0.4*a + 0.3*b + 0.2*c + 0.1*d
         for a, b, c, d in zip(norm01(ks_sham), norm01(ks_gvs),
                                norm01(mean_s), norm01(cv_gvsg))]
rank_order = np.argsort(score)[::-1]

fig, ax = plt.subplots(figsize=(18, 5))
ax.axis("off")

col_headers = ["Rank", "Metric", "Directed",
               "KS GVS-Sham\n(sig/8)",
               "KS GVS-GVS\n(sig/28)",
               "Mean|Δ|\nGVS-Sham",
               "Mean|Δ|\nGVS-GVS",
               "Dist.\nGVS-Sham",
               "Dist.\nGVS-GVS",
               "Composite\nScore"]

table_data = []
for rank_pos, i in enumerate(rank_order, start=1):
    r = df_summary.iloc[i]
    table_data.append([
        rank_pos,
        r["Metric"],
        r["Directed"],
        r["KS sig vs Sham (of 8)"],
        r["KS sig within GVS"],
        r["Mean|Δ| GVS vs Sham"],
        r["Mean|Δ| GVS vs GVS"],
        r["Distinguish GVS vs Sham"],
        r["Distinguish between GVS"],
        f"{score[i]:.3f}",
    ])

tbl = ax.table(cellText=table_data, colLabels=col_headers,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.8)

# Colour header row
for j in range(len(col_headers)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colours
for row_i in range(1, len(table_data) + 1):
    bg = "#eaf4fb" if row_i % 2 == 0 else "white"
    for col_i in range(len(col_headers)):
        tbl[row_i, col_i].set_facecolor(bg)

# Highlight top-2 rows
for col_i in range(len(col_headers)):
    tbl[1, col_i].set_facecolor("#fde68a")
    tbl[2, col_i].set_facecolor("#fef3c7")

ax.set_title("Metric Sensitivity Ranking (composite score)", fontsize=13,
             pad=20, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig11_summary_ranking_table.png"),
            dpi=150, bbox_inches="tight")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 15.  Save final ranked summary CSV
# ═══════════════════════════════════════════════════════════════════════════════
df_summary["composite_score"] = [score[i] for i in range(len(METRICS))]
df_ranked = df_summary.iloc[rank_order].reset_index(drop=True)
df_ranked.index += 1
df_ranked.index.name = "Rank"
df_ranked.to_csv(os.path.join(OUT, "metric_sensitivity_ranked.csv"))

print(f"\nAll outputs saved to: {OUT}")
print("Done.")
