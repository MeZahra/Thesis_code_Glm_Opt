from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_io import ROI_EDGE_RESULTS_ROOT, ensure_dir, write_json


def _plot_top_edges(edge_df: pd.DataFrame, out_path: Path) -> None:
    top = edge_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.barh(top["edge"], top["mean_signed_delta_session2_minus_session1"], color="#4c78a8")
    ax.axvline(0.0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Mean signed session2 - session1 delta")
    ax.set_title("Top KSG Edge Changes")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_null_validation(random_df: pd.DataFrame, observed_delta: float, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.hist(random_df["delta_sep"], bins=25, color="#bab0ac", edgecolor="white")
    ax.axvline(observed_delta, color="#e45756", linewidth=2)
    ax.set_xlabel("Random-draw delta_sep")
    ax.set_ylabel("Count")
    ax.set_title("KSG Random-Null Separation")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_supplementary_ksg_summary(out_dir: Path) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    ksg_root = ROI_EDGE_RESULTS_ROOT / "mutual_information_ksg"
    null_root = ROI_EDGE_RESULTS_ROOT / "advanced_metrics" / "random_graph_distance_null" / "mutual_information_ksg"
    observed_null_path = (
        ROI_EDGE_RESULTS_ROOT
        / "advanced_metrics"
        / "random_graph_distance_null"
        / "observed_vs_random_null_summary.csv"
    )

    edge_change_df = pd.read_csv(ksg_root / "top_edges_session2_minus_session1.csv")
    heatmap_df = pd.read_csv(ksg_root / "top_edges_subject_delta_heatmap.csv")
    selected_df = pd.read_csv(null_root / "selected_contrast_summary.csv")
    random_df = pd.read_csv(null_root / "random_draw_contrast_summary.csv")
    observed_df = pd.read_csv(observed_null_path)
    observed_ksg = observed_df[observed_df["metric"] == "mutual_information_ksg"].iloc[0]

    consistency_rows = []
    for edge in heatmap_df.columns[1:]:
        values = heatmap_df[edge].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean_delta = float(np.mean(values))
        sign_consistency = float(np.mean(np.sign(values) == np.sign(mean_delta))) if mean_delta != 0 else float(np.mean(values == 0))
        consistency_rows.append(
            {
                "edge": edge,
                "mean_delta": mean_delta,
                "std_delta": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                "n_subjects": int(values.size),
                "n_positive": int(np.sum(values > 0)),
                "n_negative": int(np.sum(values < 0)),
                "sign_consistency_fraction": sign_consistency,
            }
        )
    consistency_df = pd.DataFrame(consistency_rows).sort_values(
        "sign_consistency_fraction", ascending=False
    )

    null_summary_df = pd.DataFrame(
        [
            {
                "observed_delta_sep": float(selected_df["delta_sep"].iloc[0]),
                "observed_delta_within": float(selected_df["delta_within"].iloc[0]),
                "null_mean_delta_sep": float(random_df["delta_sep"].mean()),
                "null_sd_delta_sep": float(random_df["delta_sep"].std(ddof=1)),
                "null_mean_delta_within": float(random_df["delta_within"].mean()),
                "null_sd_delta_within": float(random_df["delta_within"].std(ddof=1)),
                "observed_null_delta_sep_z": float(observed_ksg["delta_sep_z_vs_null"]),
                "observed_null_delta_sep_p_empirical_right": float(
                    observed_ksg["delta_sep_p_empirical_right"]
                ),
                "observed_null_delta_sep_percentile": float(
                    observed_ksg["delta_sep_percentile_vs_null"]
                ),
                "observed_null_delta_within_z": float(observed_ksg["delta_within_z_vs_null"]),
                "observed_null_delta_within_p_empirical_two_sided": float(
                    observed_ksg["delta_within_p_empirical_two_sided"]
                ),
                "observed_null_delta_within_percentile": float(
                    observed_ksg["delta_within_percentile_vs_null"]
                ),
            }
        ]
    )

    edge_path = out_dir / "ksg_edge_change_summary.csv"
    consistency_path = out_dir / "ksg_edge_consistency_summary.csv"
    null_path = out_dir / "ksg_random_null_summary.csv"
    top_fig_path = out_dir / "ksg_top_edges_for_supplement.png"
    null_fig_path = out_dir / "ksg_null_validation.png"

    edge_change_df.to_csv(edge_path, index=False)
    consistency_df.to_csv(consistency_path, index=False)
    null_summary_df.to_csv(null_path, index=False)
    _plot_top_edges(edge_change_df, top_fig_path)
    _plot_null_validation(random_df, float(selected_df["delta_sep"].iloc[0]), null_fig_path)

    write_json(
        out_dir / "supplementary_ksg_summary.json",
        {
            "n_edges_ranked": int(edge_change_df.shape[0]),
            "best_ranked_edge": edge_change_df.iloc[0]["edge"],
            "observed_delta_sep": float(selected_df["delta_sep"].iloc[0]),
            "observed_delta_sep_p_empirical_right": float(
                observed_ksg["delta_sep_p_empirical_right"]
            ),
        },
    )

    return {
        "edge_change_path": edge_path,
        "consistency_path": consistency_path,
        "null_summary_path": null_path,
        "top_figure_path": top_fig_path,
        "null_figure_path": null_fig_path,
    }
