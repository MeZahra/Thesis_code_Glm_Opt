"""
Generate focused figures for the three headline targeted-PLS models.

Outputs are saved under results/connectivity/PLS/figures.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
PLS_ROOT = REPO_ROOT / "results" / "connectivity" / "PLS"
FIG_DIR = PLS_ROOT / "figures"
MODEL_ROOT = PLS_ROOT / "models"
METRICS_PATH = PLS_ROOT / "screened_models_with_detailed_metrics.csv"

KEY_MODELS = [
    {
        "config_name": "broad__control__dcm_plus_participation__vmax_pmax__pca_auto",
        "label": "Broad Control\nDCM + Participation",
        "short_label": "Broad",
        "color": "#c97a2b",
    },
    {
        "config_name": "refined__control__graph_participation__vmax_pmax__raw",
        "label": "Compact Control\nParticipation Only",
        "short_label": "Compact",
        "color": "#2f6c8f",
    },
    {
        "config_name": "localized__vmpfc_dmpfc_control_monitoring__graph_participation_single_roi__vmax_pmax__raw",
        "label": "Localized vmPFC/dmPFC\nParticipation",
        "short_label": "Localized",
        "color": "#3c8c63",
    },
]


def _load_model_bundle(config_name: str) -> dict:
    metrics_df = pd.read_csv(METRICS_PATH).set_index("config_name")
    if config_name not in metrics_df.index:
        raise KeyError(f"Model not found in metrics table: {config_name}")

    model_dir = MODEL_ROOT / config_name
    return {
        "metrics": metrics_df.loc[config_name].to_dict(),
        "scores": pd.read_csv(model_dir / "scores.csv"),
        "loo": pd.read_csv(model_dir / "loso_predictions.csv"),
        "loo_summary": pd.read_csv(model_dir / "loso_summary.csv"),
        "boot": pd.read_csv(model_dir / "feature_bootstrap_stability.csv"),
        "null": pd.read_csv(model_dir / "null_distribution.csv"),
        "summary": json.loads((model_dir / "model_summary.json").read_text(encoding="utf-8")),
        "model_dir": model_dir,
    }


def _model_title(item: dict, payload: dict) -> str:
    metrics = payload["metrics"]
    return (
        f"{item['label']}\n"
        f"latent r={metrics['observed_r']:.3f}, "
        f"p={metrics['final_permutation_p']:.4f}, "
        f"q={metrics['q_fdr_global']:.4f}, "
        f"LOSO r={metrics['loo_composite_pearson_r']:.3f}"
    )


def _plot_diagnostics_row(axes: np.ndarray, entry: dict) -> None:
    item = entry["item"]
    payload = entry["payload"]
    color = item["color"]
    scores_df = payload["scores"]
    loo_df = payload["loo"]
    boot_df = payload["boot"]
    null_df = payload["null"]
    metrics = payload["metrics"]

    ax = axes[0]
    ax.scatter(scores_df["x_score"], scores_df["y_score"], color=color, s=50, alpha=0.9)
    for _, row in scores_df.iterrows():
        ax.text(row["x_score"], row["y_score"], row["subject"], fontsize=5.5)
    ax.axhline(0.0, color="0.7", lw=0.8)
    ax.axvline(0.0, color="0.7", lw=0.8)
    ax.set_xlabel("Neural latent score")
    ax.set_ylabel("Behavior latent score")
    ax.set_title(_model_title(item, payload), fontsize=10)

    ax = axes[1]
    actual = loo_df["actual_motor_vigor_composite"]
    predicted = loo_df["predicted_motor_vigor_composite"]
    ax.scatter(actual, predicted, color=color, s=50, alpha=0.9)
    lo = float(min(actual.min(), predicted.min()))
    hi = float(max(actual.max(), predicted.max()))
    ax.plot([lo, hi], [lo, hi], color="0.6", linestyle="--", lw=1)
    for _, row in loo_df.iterrows():
        ax.text(
            row["actual_motor_vigor_composite"],
            row["predicted_motor_vigor_composite"],
            row["subject"],
            fontsize=5.5,
        )
    ax.set_xlabel("Observed LOSO vigor composite")
    ax.set_ylabel("Predicted LOSO vigor composite")
    ax.set_title(
        f"LOSO composite\nr={metrics['loo_composite_pearson_r']:.3f}, "
        f"mean behavior r={metrics['loo_mean_behavior_pearson_r']:.3f}",
        fontsize=10,
    )

    ax = axes[2]
    top = boot_df.reindex(boot_df["full_weight"].abs().sort_values(ascending=False).index).head(6)
    top = top.iloc[::-1]
    errors = np.vstack(
        [
            np.maximum(0.0, top["full_weight"].to_numpy() - top["ci_2p5"].to_numpy()),
            np.maximum(0.0, top["ci_97p5"].to_numpy() - top["full_weight"].to_numpy()),
        ]
    )
    colors = ["#d65f5f" if value > 0 else "#4c78a8" for value in top["full_weight"]]
    ax.barh(top["display_name"], top["full_weight"], color=colors, xerr=errors, capsize=3)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Feature weight")
    ax.set_title(
        f"Bootstrap weights\nmedian similarity={metrics['median_boot_weight_similarity']:.3f}",
        fontsize=10,
    )

    ax = axes[3]
    perm_values = np.abs(null_df["permuted_r"].to_numpy())
    ax.hist(perm_values, bins=40, color="#b9d8ea", edgecolor="white")
    ax.axvline(abs(metrics["observed_r"]), color="#d65f5f", lw=2)
    ax.axvline(abs(metrics["loo_composite_pearson_r"]), color="#3c8c63", lw=2, linestyle="--")
    ax.set_xlabel("|r| under permutation")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Permutation null\nobs={metrics['observed_r']:.3f}, LOSO={metrics['loo_composite_pearson_r']:.3f}",
        fontsize=10,
    )


def make_diagnostics_figure(model_payloads: list[dict]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(18, 13), constrained_layout=True)

    for row_idx, entry in enumerate(model_payloads):
        _plot_diagnostics_row(axes[row_idx], entry)

    fig.suptitle("Targeted PLS Key Models: Diagnostics", fontsize=14)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "targeted_pls_key_models_diagnostics.png", dpi=220)
    fig.savefig(FIG_DIR / "targeted_pls_key_models_diagnostics.pdf")
    plt.close(fig)


def make_separate_diagnostics_figures(model_payloads: list[dict]) -> list[Path]:
    saved_paths: list[Path] = []
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for entry in model_payloads:
        item = entry["item"]
        slug = item["short_label"].lower()
        title_label = item["label"].replace("\n", " ")
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
        _plot_diagnostics_row(axes, entry)
        fig.suptitle(f"Targeted PLS Diagnostics: {title_label}", fontsize=14)

        png_path = FIG_DIR / f"targeted_pls_key_models_diagnostics_{slug}.png"
        pdf_path = FIG_DIR / f"targeted_pls_key_models_diagnostics_{slug}.pdf"
        fig.savefig(png_path, dpi=220)
        fig.savefig(pdf_path)
        plt.close(fig)
        saved_paths.extend([png_path, pdf_path])

    return saved_paths


def make_summary_figure(model_payloads: list[dict]) -> None:
    labels = [entry["item"]["short_label"] for entry in model_payloads]
    colors = [entry["item"]["color"] for entry in model_payloads]
    metrics_rows = [entry["payload"]["metrics"] for entry in model_payloads]
    loo_tables = [entry["payload"]["loo_summary"] for entry in model_payloads]

    observed = [row["observed_r"] for row in metrics_rows]
    loo_comp = [row["loo_composite_pearson_r"] for row in metrics_rows]
    final_p = [row["final_permutation_p"] for row in metrics_rows]
    q_vals = [row["q_fdr_global"] for row in metrics_rows]
    boot_sim = [row["median_boot_weight_similarity"] for row in metrics_rows]
    n_features = [row["n_brain_features"] for row in metrics_rows]
    xpc = [row["x_pca_components"] for row in metrics_rows]
    stable_counts = [row["n_stable_features_ci_excluding_zero"] for row in metrics_rows]
    vmax_loo = [
        float(table.loc[table["metric"] == "task_vmax_delta", "pearson_r"].iloc[0]) for table in loo_tables
    ]
    pmax_loo = [
        float(table.loc[table["metric"] == "task_pmax_delta", "pearson_r"].iloc[0]) for table in loo_tables
    ]

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    ax = axes[0, 0]
    width = 0.35
    ax.bar(x - width / 2, observed, width=width, color=colors, alpha=0.9, label="Latent r")
    ax.bar(x + width / 2, loo_comp, width=width, color=colors, alpha=0.45, label="LOSO r")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Correlation")
    ax.set_title("Full-sample vs LOSO performance")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[0, 1]
    neglogp = [-np.log10(value) for value in final_p]
    ax.bar(x, neglogp, color=colors, alpha=0.9)
    ax.axhline(-np.log10(0.05), color="#d65f5f", lw=1.5, linestyle="--", label="p = 0.05")
    for idx, value in enumerate(q_vals):
        ax.text(x[idx], neglogp[idx] + 0.05, f"q={value:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("-log10(final permutation p)")
    ax.set_title("Permutation strength")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 0]
    width = 0.35
    ax.bar(x - width / 2, vmax_loo, width=width, color=colors, alpha=0.9, label="Vmax LOSO r")
    ax.bar(x + width / 2, pmax_loo, width=width, color=colors, alpha=0.45, label="Pmax LOSO r")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Pearson r")
    ax.set_title("Behavior-specific LOSO prediction")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    ax.bar(x, boot_sim, color=colors, alpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median bootstrap weight similarity")
    ax.set_title("Stability and model size")
    twin = ax.twinx()
    twin.plot(x, n_features, color="black", marker="o", lw=1.5, label="n features")
    twin.set_ylabel("Neural features")
    for idx, (stable_count, xpc_val) in enumerate(zip(stable_counts, xpc)):
        ax.text(
            x[idx],
            boot_sim[idx] + 0.03,
            f"stable={int(stable_count)}\nXpc={int(xpc_val)}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    fig.suptitle("Targeted PLS Key Models: Summary Comparison", fontsize=14)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "targeted_pls_key_models_summary.png", dpi=220)
    fig.savefig(FIG_DIR / "targeted_pls_key_models_summary.pdf")
    plt.close(fig)


def main() -> None:
    model_payloads = []
    for item in KEY_MODELS:
        payload = _load_model_bundle(item["config_name"])
        model_payloads.append({"item": item, "payload": payload})

    make_diagnostics_figure(model_payloads)
    separate_paths = make_separate_diagnostics_figures(model_payloads)
    make_summary_figure(model_payloads)

    print("Saved figures:")
    print(FIG_DIR / "targeted_pls_key_models_diagnostics.png")
    print(FIG_DIR / "targeted_pls_key_models_diagnostics.pdf")
    for path in separate_paths:
        print(path)
    print(FIG_DIR / "targeted_pls_key_models_summary.png")
    print(FIG_DIR / "targeted_pls_key_models_summary.pdf")


if __name__ == "__main__":
    main()
