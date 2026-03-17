from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_io import (
    CIRCUIT_BASE_ROIS,
    TMP_DCM_ROOT,
    aggregate_matrix_by_base_roi,
    bh_fdr,
    ensure_dir,
    list_paired_dcm_subjects,
    load_dcm_labels,
    load_subject_dcm_matrix,
    load_subject_dcm_vector,
    paired_delta_stats,
    read_json,
    safe_slug,
    to_serializable,
    write_json,
)


def _reduce_to_circuit(matrix: np.ndarray, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    reduced, reduced_labels, _ = aggregate_matrix_by_base_roi(
        matrix,
        labels,
        include_base_rois=CIRCUIT_BASE_ROIS,
    )
    return reduced, reduced_labels


def _plot_circuit_matrices(
    off_mean: np.ndarray,
    on_mean: np.ndarray,
    delta_mean: np.ndarray,
    q_matrix: np.ndarray,
    labels: list[str],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    shared_lim = float(
        np.nanmax(np.abs(np.concatenate([off_mean.ravel(), on_mean.ravel()])))
    )
    delta_lim = float(np.nanmax(np.abs(delta_mean))) or 1.0
    panels = [
        (off_mean, "Circuit Mean OFF", shared_lim),
        (on_mean, "Circuit Mean ON", shared_lim),
        (delta_mean, "Circuit Mean ON - OFF", delta_lim),
    ]

    for ax, (matrix, title, vmax) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                mark = ""
                if title == "Circuit Mean ON - OFF" and np.isfinite(q_matrix[row, col]):
                    if q_matrix[row, col] < 0.05:
                        mark = "*"
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.2f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_dcm_medication_analysis(
    out_dir: Path,
    circuit_rois: list[str] | None = None,
) -> dict:
    out_dir = ensure_dir(Path(out_dir))
    circuit_rois = circuit_rois or CIRCUIT_BASE_ROIS
    dcm_labels = load_dcm_labels()
    group_summary = read_json(TMP_DCM_ROOT / "group_level" / "group_summary.json")
    subjects = list_paired_dcm_subjects()

    subject_rows = []
    comparison_rows = []
    off_mats = []
    on_mats = []
    delta_mats = []

    for subject in subjects:
        off_full = load_subject_dcm_matrix(subject, "dcm_session1_off.npy")
        on_full = load_subject_dcm_matrix(subject, "dcm_session2_on.npy")
        delta_full = load_subject_dcm_matrix(subject, "dcm_session2_minus_session1.npy")
        modulatory_full = load_subject_dcm_matrix(
            subject, "dcm_modulatory_session2_on_minus_session1_off.npy"
        )
        intrinsic_full = load_subject_dcm_matrix(subject, "dcm_intrinsic_session1_off.npy")
        constant_drive_full = load_subject_dcm_vector(subject, "dcm_constant_drive.npy")

        off, labels = _reduce_to_circuit(off_full, dcm_labels)
        on, _ = _reduce_to_circuit(on_full, dcm_labels)
        delta, _ = _reduce_to_circuit(delta_full, dcm_labels)
        modulatory, _ = _reduce_to_circuit(modulatory_full, dcm_labels)
        intrinsic, _ = _reduce_to_circuit(intrinsic_full, dcm_labels)
        off_mats.append(off)
        on_mats.append(on)
        delta_mats.append(delta)

        for target_idx, target_roi in enumerate(labels):
            for source_idx, source_roi in enumerate(labels):
                subject_rows.append(
                    {
                        "subject": subject,
                        "parameter_type": "coupling",
                        "target_roi": target_roi,
                        "source_roi": source_roi,
                        "off_parameter": float(off[target_idx, source_idx]),
                        "on_parameter": float(on[target_idx, source_idx]),
                        "delta_on_minus_off": float(delta[target_idx, source_idx]),
                        "modulatory_delta": float(modulatory[target_idx, source_idx]),
                        "intrinsic_off_parameter": float(intrinsic[target_idx, source_idx]),
                    }
                )

        constant_drive_matrix, constant_drive_labels = _reduce_to_circuit(
            np.diag(constant_drive_full), dcm_labels
        )
        for idx, roi in enumerate(constant_drive_labels):
            subject_rows.append(
                {
                    "subject": subject,
                    "parameter_type": "constant_drive",
                    "target_roi": roi,
                    "source_roi": "constant",
                    "off_parameter": float("nan"),
                    "on_parameter": float("nan"),
                    "delta_on_minus_off": float("nan"),
                    "modulatory_delta": float("nan"),
                    "intrinsic_off_parameter": float(constant_drive_matrix[idx, idx]),
                }
            )

    subject_df = pd.DataFrame(subject_rows)
    coupling_df = subject_df[subject_df["parameter_type"] == "coupling"].copy()

    for target_roi in circuit_rois:
        for source_roi in circuit_rois:
            edge_df = coupling_df[
                (coupling_df["target_roi"] == target_roi)
                & (coupling_df["source_roi"] == source_roi)
            ].sort_values("subject")
            delta = edge_df["delta_on_minus_off"].to_numpy()
            stats_dict = paired_delta_stats(delta)
            comparison_rows.append(
                {
                    "target_roi": target_roi,
                    "source_roi": source_roi,
                    "edge": f"{source_roi} -> {target_roi}",
                    "is_self_connection": bool(target_roi == source_roi),
                    "mean_off": float(edge_df["off_parameter"].mean()),
                    "mean_on": float(edge_df["on_parameter"].mean()),
                    "mean_modulatory_delta": float(edge_df["modulatory_delta"].mean()),
                    **stats_dict,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df["q_fdr"] = bh_fdr(comparison_df["p_signflip"])
    comparison_df = comparison_df.sort_values(
        ["q_fdr", "p_signflip", "is_self_connection", "mean_delta"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)

    subject_path = out_dir / "dcm_subject_parameters.csv"
    group_path = out_dir / "dcm_group_comparison.csv"
    group_fdr_path = out_dir / "dcm_group_comparison_fdr.csv"
    summary_path = out_dir / "dcm_circuit_summary.json"
    fig_path = out_dir / "dcm_circuit_off_on.png"

    subject_df.to_csv(subject_path, index=False)
    comparison_df.to_csv(group_path, index=False)
    comparison_df.sort_values(["q_fdr", "p_signflip", "mean_delta"], ascending=[True, True, False]).to_csv(
        group_fdr_path, index=False
    )

    off_mean = np.mean(np.stack(off_mats, axis=0), axis=0)
    on_mean = np.mean(np.stack(on_mats, axis=0), axis=0)
    delta_mean = np.mean(np.stack(delta_mats, axis=0), axis=0)
    q_matrix = np.full_like(delta_mean, np.nan, dtype=float)
    for _, row in comparison_df.iterrows():
        tgt = circuit_rois.index(row["target_roi"])
        src = circuit_rois.index(row["source_roi"])
        q_matrix[tgt, src] = row["q_fdr"]
    _plot_circuit_matrices(off_mean, on_mean, delta_mean, q_matrix, circuit_rois, fig_path)

    summary = {
        "analysis_note": (
            "This module re-summarizes the precomputed bilinear neuronal-state "
            "model saved under results/connectivity/tmp/dynamic_causal_modeling; "
            "it does not refit a classical hemodynamic DCM."
        ),
        "input_root": TMP_DCM_ROOT,
        "n_subjects": len(subjects),
        "circuit_rois": circuit_rois,
        "session_definition": group_summary.get("session_definition", {}),
        "matrix_convention": group_summary.get("matrix_convention"),
        "group_mean_abs_offdiag_off": group_summary.get("global_mean_abs_offdiag_off"),
        "group_mean_abs_offdiag_on": group_summary.get("global_mean_abs_offdiag_on"),
        "group_mean_abs_offdiag_delta_on_minus_off": group_summary.get(
            "global_mean_abs_offdiag_delta_on_minus_off"
        ),
        "n_edges_tested": int(comparison_df.shape[0]),
        "n_edges_fdr_below_0_05": int((comparison_df["q_fdr"] < 0.05).sum()),
        "top_edges_by_abs_mean_delta": to_serializable(
            comparison_df.assign(abs_mean_delta=comparison_df["mean_delta"].abs())
            .sort_values("abs_mean_delta", ascending=False)
            .head(10)[["edge", "mean_delta", "p_signflip", "q_fdr"]]
            .to_dict(orient="records")
        ),
        "output_files": {
            "dcm_subject_parameters": subject_path,
            "dcm_group_comparison": group_path,
            "dcm_group_comparison_fdr": group_fdr_path,
            "dcm_circuit_off_on": fig_path,
        },
    }
    write_json(summary_path, to_serializable(summary))

    return {
        "subjects": subjects,
        "circuit_rois": circuit_rois,
        "subject_parameters_path": subject_path,
        "group_comparison_path": group_path,
        "group_comparison_fdr_path": group_fdr_path,
        "summary_path": summary_path,
        "figure_path": fig_path,
    }
